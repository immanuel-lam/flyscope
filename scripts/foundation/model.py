"""MLX trainable counterpart of the explicitly source-wired CPU experiment."""
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[2]


class CircuitFoundation(nn.Module):
    def __init__(self, directory=None, reference=None):
        super().__init__()
        self.base = reference if reference is not None else load(str(directory or ROOT / 'data/foundation/smollm2-135m'))[0]
        w = dict(np.load(ROOT / 'data/graph-language/run-2/wiring.npz'))
        inputs, outputs, slots = w['inputs'], w['outputs'], w['slots']
        source = np.load(ROOT / 'data/language/graph.npz')['counts'] > 0
        if not np.all(source[outputs, inputs]) or np.any(w['mask'] & ~np.eye(512, dtype=bool) & ~source):
            raise ValueError('Every inter-cell operation must use a source edge')
        self._slots = mx.array(slots)
        self._inputs = mx.array(inputs)
        self._outputs = mx.array(outputs)
        input_gate = np.zeros(512, np.float32)
        input_gate[inputs] = 1
        relay_gate = np.zeros(512, np.float32)
        relay_gate[outputs] = 1
        relay_pre = np.zeros(512, np.int32)
        relay_pre[outputs] = inputs
        self._input_gate = mx.array(input_gate)
        self._relay_gate = mx.array(relay_gate)
        self._relay_pre = mx.array(relay_pre)
        self._mask = mx.array(w['mask'])

    def rope(self, x):
        half = x.shape[-1] // 2
        inverse = self.base.args.rope_theta ** (-mx.arange(half, dtype=mx.float32) * 2 / x.shape[-1])
        angles = self._slots[:, None] * inverse[None]
        cosine, sine = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
        first, second = x[..., :half], x[..., half:]
        return mx.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1).astype(x.dtype)

    def __call__(self, tokens, valid=None, ablated=False, attention_cut=False, return_states=False, all_logits=False):
        if tokens.shape[1] != 128:
            raise ValueError('Expected 128 explicit token slots')
        if valid is None:
            valid = mx.ones(tokens.shape, dtype=mx.bool_)
        embeddings = self.base.model.embed_tokens(tokens) * valid[..., None]
        x = embeddings[:, self._slots] * self._input_gate[None, :, None]
        if not ablated:
            x = x + x[:, self._relay_pre] * self._relay_gate[None, :, None]
        identity = mx.eye(512, dtype=mx.bool_)
        mask = (self._mask[None, None] & valid[:, None, None, self._slots]) | identity
        if ablated or attention_cut:
            mask = identity
        for layer in self.base.model.layers:
            a = layer.self_attn
            z = layer.input_layernorm(x)
            batch, cells, width = z.shape
            q = a.q_proj(z).reshape(batch, cells, a.n_heads, a.head_dim).transpose(0, 2, 1, 3)
            k = a.k_proj(z).reshape(batch, cells, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
            v = a.v_proj(z).reshape(batch, cells, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
            q, k = self.rope(q), self.rope(k)
            message = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
            x = x + a.o_proj(message.transpose(0, 2, 1, 3).reshape(batch, cells, width))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        readout = x[:, self._outputs] if all_logits else x[:, self._outputs[-1]]
        logits = self.base.model.embed_tokens.as_linear(self.base.model.norm(readout))
        return (logits, x) if return_states else logits

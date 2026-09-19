"""CPU-only pretrained-weight adaptation experiment; not the deployed FlyGPT.

All graph-mode token signals enter disjoint source input cells. A declared
identity-feature relay follows verified input-to-readout source edges, then all
attention communication follows the causal source mask. Local feature transforms
and these dynamics are engineered, not biological measurements.
"""
from pathlib import Path
import json
import struct
import time
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]


def load_weights(path):
    """Read numeric safetensors arrays without executing model code or pickle."""
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    raw = np.memmap(path, mode='r', dtype=np.uint8, offset=8 + length)
    result = {}
    for key, spec in header.items():
        if key == '__metadata__':
            continue
        start, end = spec['data_offsets']
        if not 0 <= start <= end <= len(raw):
            raise ValueError('Invalid tensor bounds')
        if spec['dtype'] == 'BF16':
            value = (raw[start:end].view('<u2').astype(np.uint32) << 16).view(np.float32)
        elif spec['dtype'] == 'F32':
            value = raw[start:end].view('<f4').copy()
        else:
            raise ValueError(f'Unsupported tensor dtype {spec["dtype"]}')
        if value.size != int(np.prod(spec['shape'])):
            raise ValueError('Invalid tensor shape')
        result[key] = value.reshape(spec['shape'])
    return result


class FoundationRuntime:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT / 'data/foundation/smollm2-135m')
        self.config = json.loads((self.directory / 'config.json').read_text())
        self.weights = load_weights(self.directory / 'model.safetensors')
        self.tokenizer = Tokenizer.from_file(str(self.directory / 'tokenizer.json'))
        wiring = dict(np.load(ROOT / 'data/graph-language/run-2/wiring.npz'))
        self.inputs, self.outputs = wiring['inputs'], wiring['outputs']
        self.slots, self.mask = wiring['slots'], wiring['mask']
        source = np.load(ROOT / 'data/language/graph.npz')['counts'] > 0
        if np.any(self.mask & ~np.eye(512, dtype=bool) & ~source):
            raise ValueError('Attention contains an off-source edge')
        if not np.all(source[self.outputs, self.inputs]) or set(self.inputs) & set(self.outputs):
            raise ValueError('Relay must follow source edges between disjoint cells')
        self.ids = [str(row[0]) for row in json.loads((ROOT / 'data/language/manifest.json').read_text())['neurons']]

    def norm(self, x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.config['rms_norm_eps']) * weight

    def rope(self, x, positions):
        half = x.shape[-1] // 2
        inverse = self.config['rope_theta'] ** (-np.arange(half, dtype=np.float32) * 2 / x.shape[-1])
        angles = positions[:, None].astype(np.float32) * inverse[None]
        cosine, sine = np.cos(angles)[None], np.sin(angles)[None]
        first, second = x[..., :half], x[..., half:]
        return np.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1)

    def next(self, prefix, graph=True, ablated=False, attention_cut=False, trace=False):
        prefix = list(prefix)[-128:]
        if not prefix:
            raise ValueError('A nonempty token context is required')
        w = self.weights
        if graph:
            tokens = np.zeros(128, np.int64)
            valid = np.zeros(128, bool)
            tokens[-len(prefix):], valid[-len(prefix):] = prefix, True
            x = np.zeros((512, self.config['hidden_size']), np.float32)
            x[self.inputs] = w['model.embed_tokens.weight'][tokens] * valid[:, None]
            injected = x.copy() if trace else None
            if not ablated:
                # Explicit first inter-cell update, along verified source pairs.
                x[self.outputs] += x[self.inputs]
            positions = self.slots
            mask = (self.mask & valid[self.slots][None, :]) | np.eye(512, dtype=bool)
            if ablated or attention_cut:
                mask = np.eye(512, dtype=bool)
            readout = int(self.outputs[-1])
        else:
            x = w['model.embed_tokens.weight'][prefix].copy()
            positions = np.arange(len(prefix))
            mask = np.tril(np.ones((len(prefix), len(prefix)), bool))
            if ablated or attention_cut:
                mask = np.eye(len(prefix), dtype=bool)
            readout = len(prefix) - 1
            injected = x.copy() if trace else None
        snapshots = [x.copy()] if trace else None
        heads = self.config['num_attention_heads']
        kv_heads = self.config['num_key_value_heads']
        width = self.config['hidden_size']
        head_dim = width // heads
        for layer in range(self.config['num_hidden_layers']):
            key = f'model.layers.{layer}.'
            normed = self.norm(x, w[key + 'input_layernorm.weight'])
            q = (normed @ w[key + 'self_attn.q_proj.weight'].T).reshape(len(x), heads, head_dim).transpose(1, 0, 2)
            k = (normed @ w[key + 'self_attn.k_proj.weight'].T).reshape(len(x), kv_heads, head_dim).transpose(1, 0, 2)
            v = (normed @ w[key + 'self_attn.v_proj.weight'].T).reshape(len(x), kv_heads, head_dim).transpose(1, 0, 2)
            q, k = self.rope(q, positions), self.rope(k, positions)
            k, v = np.repeat(k, heads // kv_heads, axis=0), np.repeat(v, heads // kv_heads, axis=0)
            scores = (q @ k.transpose(0, 2, 1)) * head_dim ** -.5
            scores = np.where(mask[None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= attention.sum(axis=-1, keepdims=True)
            message = (attention @ v).transpose(1, 0, 2).reshape(len(x), width)
            x += message @ w[key + 'self_attn.o_proj.weight'].T
            normed = self.norm(x, w[key + 'post_attention_layernorm.weight'])
            gate = normed @ w[key + 'mlp.gate_proj.weight'].T
            gate = gate / (1 + np.exp(-np.clip(gate, -80, 80)))
            up = normed @ w[key + 'mlp.up_proj.weight'].T
            x += (gate * up) @ w[key + 'mlp.down_proj.weight'].T
            if trace:
                snapshots.append(x.copy())
        logits = w['model.embed_tokens.weight'] @ self.norm(x[readout], w['model.norm.weight'])
        return logits, x, {'injected': injected, 'blocks': snapshots} if trace else None

    def generate(self, message, max_tokens=16, graph=True):
        prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        prompt += f'<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        prefix = self.tokenizer.encode(prompt).ids
        end = self.tokenizer.token_to_id('<|im_end|>')
        start = time.perf_counter()
        generated = []
        for _ in range(max_tokens):
            logits, _, _ = self.next(prefix + generated, graph=graph)
            logits[self.tokenizer.token_to_id('<|im_start|>')] = -np.inf
            token = int(np.argmax(logits))
            if token == end:
                break
            generated.append(token)
        elapsed = time.perf_counter() - start
        return {'text': self.tokenizer.decode(generated), 'tokenIds': generated,
                'elapsedSeconds': elapsed, 'tokensPerSecond': len(generated) / max(elapsed, 1e-9),
                'mode': 'source-graph-adaptation' if graph else 'ordinary-reference', 'engine': 'NumPy CPU'}

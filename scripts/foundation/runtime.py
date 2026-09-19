"""CPU-only pretrained-weight adaptation experiment; not the deployed FlyGPT.

All graph-mode token signals enter disjoint source input cells. A declared
identity-feature relay follows verified input-to-readout source edges, then all
attention communication follows the causal source mask. Local feature transforms
and these dynamics are engineered, not biological measurements.
"""
from pathlib import Path
import json
import hashlib
import re
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


def load_quantized_weights(directory):
    """Decode checked, per-row int8 storage to float32 CPU computation weights."""
    manifest = json.loads((directory / 'weights.json').read_text())
    if manifest.get('format') != 'flyscope-row-int8-v1':
        raise ValueError('Unsupported weight storage format')
    result = {}
    for key, spec in manifest['tensors'].items():
        filename = spec['file']
        if not re.fullmatch(r'weight-[0-9]{3}\.npz', filename):
            raise ValueError('Invalid weight filename')
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Weight checksum mismatch')
        with np.load(path, allow_pickle=False) as archive:
            values = archive['values']
            if list(values.shape) != spec['shape']:
                raise ValueError('Weight shape mismatch')
            if spec['storage'] == 'row-int8':
                scales = archive['scales']
                if (values.dtype != np.int8 or values.ndim != 2 or scales.dtype != np.float32
                        or scales.shape != (values.shape[0], 1) or not np.all(np.isfinite(scales))
                        or not np.all(scales > 0)):
                    raise ValueError('Invalid quantized weight')
                result[key] = values.astype(np.float32) * scales
            elif spec['storage'] == 'float32' and values.dtype == np.float32 and np.all(np.isfinite(values)):
                result[key] = values.copy()
            else:
                raise ValueError('Invalid weight encoding')
    return result


class FoundationRuntime:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT / 'data/foundation/smollm2-135m')
        self.config = json.loads((self.directory / 'config.json').read_text())
        self.weights = (load_quantized_weights(self.directory) if (self.directory / 'weights.json').exists()
                        else load_weights(self.directory / 'model.safetensors'))
        self.tokenizer = Tokenizer.from_file(str(self.directory / 'tokenizer.json'))
        wiring_file = self.directory / 'wiring.npz'
        wiring = dict(np.load(wiring_file if wiring_file.exists() else ROOT / 'data/graph-language/run-2/wiring.npz'))
        self.inputs, self.outputs = wiring['inputs'], wiring['outputs']
        self.slots, self.mask = wiring['slots'], wiring['mask']
        source_file = self.directory / 'source-mask.npy'
        source = (np.load(source_file, allow_pickle=False) if source_file.exists()
                  else np.load(ROOT / 'data/language/graph.npz')['counts'] > 0)
        if np.any(self.mask & ~np.eye(512, dtype=bool) & ~source):
            raise ValueError('Attention contains an off-source edge')
        if not np.all(source[self.outputs, self.inputs]) or set(self.inputs) & set(self.outputs):
            raise ValueError('Relay must follow source edges between disjoint cells')
        cells_file = self.directory / 'source-cells.json'
        cells_file = cells_file if cells_file.exists() else ROOT / 'data/language/manifest.json'
        self.ids = [str(row[0]) for row in json.loads(cells_file.read_text())['neurons']]

    def norm(self, x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.config['rms_norm_eps']) * weight

    def rope(self, x, positions):
        half = x.shape[-1] // 2
        inverse = self.config['rope_theta'] ** (-np.arange(half, dtype=np.float32) * 2 / x.shape[-1])
        angles = positions[:, None].astype(np.float32) * inverse[None]
        cosine, sine = np.cos(angles)[None], np.sin(angles)[None]
        first, second = x[..., :half], x[..., half:]
        return np.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1)

    def next(self, prefix, graph=True, ablated=False, attention_cut=False, trace=False, inspect=False):
        if inspect and not graph:
            raise ValueError('Source-cell inspection is only available for source-graph computation')
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
        selected = np.concatenate([self.inputs[-8:], self.outputs[-16:]]) if inspect else None
        inspection = None
        if inspect:
            inspection = {'schema': 'foundation-source-computation-v1', 'feature': 0,
                          'cells': [{'id': self.ids[int(i)], 'index': int(i), 'slot': int(self.slots[i]),
                                     'role': 'input' if i in self.inputs else 'readout'} for i in selected],
                          'relay': [{'sourceId': self.ids[int(i)], 'targetId': self.ids[int(o)],
                                     'featureContribution': float(w['model.embed_tokens.weight'][tokens[slot], 0])
                                     if valid[slot] and not ablated else 0.}
                                    for slot, (i, o) in enumerate(zip(self.inputs, self.outputs))],
                          'blocks': [], 'assumptions': 'One selected feature from 576 per cell. Every cell computes. Self and MLP updates are local operations; relay and intercell attention use source edges. Continuous model values, not biological spikes.'}
        heads = self.config['num_attention_heads']
        kv_heads = self.config['num_key_value_heads']
        width = self.config['hidden_size']
        head_dim = width // heads
        for layer in range(self.config['num_hidden_layers']):
            key = f'model.layers.{layer}.'
            before = x[selected, 0].copy() if inspect else None
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
            if inspect:
                # This is the actual signed feature-0 contribution after the
                # output projection, summed over all attention heads.
                projected = np.einsum('hnd,hd->hn', v, w[key + 'self_attn.o_proj.weight'][0].reshape(heads, head_dim))
                contributions = np.einsum('hij,hj->ij', attention[:, selected, :], projected)
                after_attention = x[selected, 0].copy()
            normed = self.norm(x, w[key + 'post_attention_layernorm.weight'])
            gate = normed @ w[key + 'mlp.gate_proj.weight'].T
            gate = gate / (1 + np.exp(-np.clip(gate, -80, 80)))
            up = normed @ w[key + 'mlp.up_proj.weight'].T
            x += (gate * up) @ w[key + 'mlp.down_proj.weight'].T
            if inspect:
                edges = [{'source': j, 'target': i, 'contribution': float(contributions[i, pre])}
                         for i, post in enumerate(selected) for j, pre in enumerate(selected)
                         if pre != post and self.mask[post, pre]]
                self_values = contributions[np.arange(len(selected)), selected]
                shown = np.zeros(len(selected), np.float32)
                for edge in edges:
                    shown[edge['target']] += edge['contribution']
                inspection['blocks'].append({'index': layer, 'before': before.tolist(),
                    'afterAttention': after_attention.tolist(), 'after': x[selected, 0].tolist(),
                    'selfContribution': self_values.tolist(), 'edges': edges,
                    'otherCellContribution': (contributions.sum(axis=-1) - self_values - shown).tolist(),
                    'mlpUpdate': (x[selected, 0] - after_attention).tolist()})
            if trace:
                snapshots.append(x.copy())
        logits = w['model.embed_tokens.weight'] @ self.norm(x[readout], w['model.norm.weight'])
        details = {'injected': injected, 'blocks': snapshots} if trace else {}
        if inspect:
            inspection['activityRms'] = np.sqrt(np.mean(x * x, axis=-1)).tolist()
            inspection['activityIds'] = self.ids
            top = np.argsort(logits)[-5:][::-1]
            probabilities = np.exp(logits.astype(np.float64) - np.max(logits))
            probabilities /= probabilities.sum()
            inspection['topTokens'] = [{'id': int(i), 'text': self.tokenizer.decode([int(i)]),
                                        'probability': float(probabilities[i])} for i in top]
            details['inspection'] = inspection
        return logits, x, details if details else None

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

"""Portable NumPy CPU inference for the experimental source-edge attention model."""
from pathlib import Path
import json
import time
import numpy as np
from tokenizers import Tokenizer


class GraphRuntime:
    def __init__(self, directory):
        directory = Path(directory)
        self.config = json.loads((directory / 'manifest.json').read_text())
        self.weights = dict(np.load(directory / 'runtime.npz'))
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.inputs = np.array(self.config['inputIndices'])
        self.outputs = np.array(self.config['outputIndices'])
        self.mask = self.weights.pop('attention_mask').astype(bool)
        self.heads = self.config['heads']
        self.layers = self.config['layers']

    @staticmethod
    def norm(x, weight):
        return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * weight

    def forward(self, tokens, ablated=False, trace=False):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.shape != (128,):
            raise ValueError('Expected exactly 128 token slots')
        w = self.weights
        x = w['cell_embedding.weight'].copy()
        x[self.inputs] += w['embedding.weight'][tokens] * (tokens != 0)[:, None]
        snapshots = [x.copy()] if trace else None
        mask = np.eye(512, dtype=bool) if ablated else self.mask
        width = x.shape[1]
        for layer in range(self.layers):
            prefix = f'blocks.{layer}.'
            qkv = self.norm(x, w[prefix + 'norm1.weight']) @ w[prefix + 'qkv.weight'].T
            q, k, v = [a.reshape(512, self.heads, width // self.heads).transpose(1, 0, 2)
                       for a in np.split(qkv, 3, axis=-1)]
            scores = (q @ k.transpose(0, 2, 1)) * (width // self.heads) ** -.5
            scores = np.where(mask[None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= attention.sum(axis=-1, keepdims=True)
            value = (attention @ v).transpose(1, 0, 2).reshape(512, width)
            x += value @ w[prefix + 'projection.weight'].T
            hidden = self.norm(x, w[prefix + 'norm2.weight']) @ w[prefix + 'up.weight'].T
            activated = .5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + .044715 * hidden ** 3)))
            x += activated @ w[prefix + 'down.weight'].T
            if trace:
                snapshots.append(x.copy())
        logits = self.norm(x[self.outputs], w['norm.weight']) @ w['embedding.weight'].T
        return logits, x, snapshots

    def next(self, prefix, ablated=False):
        # Right-align the current context. No generated or target future tokens enter it.
        window = np.zeros(128, dtype=np.int64)
        context = prefix[-128:]
        if context:
            window[-len(context):] = context
        logits, states, _ = self.forward(window, ablated)
        return logits[-1], states

    def generate(self, message, history=None, max_tokens=64, ablated=False):
        prompt = ''.join(f'<{m["role"]}> {m["content"]} <end>\n' for m in (history or [])[-8:])
        prompt += f'<user> {message} <end>\n<assistant>'
        prefix = self.tokenizer.encode(prompt).ids
        started = time.perf_counter()
        generated = []
        forbidden = [self.tokenizer.token_to_id(t) for t in ['<pad>', '<unk>', '<user>', '<assistant>']]
        end = self.tokenizer.token_to_id('<end>')
        for _ in range(max_tokens):
            logits, _ = self.next(prefix + generated, ablated)
            logits = logits.copy()
            logits[forbidden] = -np.inf
            token = int(np.argmax(logits))
            if token == end:
                break
            generated.append(token)
        elapsed = time.perf_counter() - started
        return {'text': self.tokenizer.decode(generated).strip(), 'tokenIds': generated,
                'elapsedSeconds': elapsed, 'tokensPerSecond': len(generated) / max(elapsed, 1e-9),
                'engine': 'NumPy CPU', 'ablated': ablated}

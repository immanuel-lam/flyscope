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
        self.cells = self.weights['cell_embedding.weight'].shape[0]

    def attention_mask(self, tokens, ablated=False):
        mask = np.eye(self.cells, dtype=bool) if ablated else self.mask
        if self.config.get('paddingMask', False):
            slots = np.asarray(self.config.get('slots', list(range(self.cells))))
            active = tokens[slots] != 0
            mask = (mask & active[None, :]) | np.eye(self.cells, dtype=bool)
        return mask

    @staticmethod
    def norm(x, weight):
        return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * weight

    def forward(self, tokens, ablated=False, trace=False, observer=None):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.shape != (128,):
            raise ValueError('Expected exactly 128 token slots')
        w = self.weights
        x = w['cell_embedding.weight'].copy()
        x[self.inputs] += w['embedding.weight'][tokens] * (tokens != 0)[:, None]
        snapshots = [x.copy()] if trace else None
        mask = self.attention_mask(tokens, ablated)
        width = x.shape[1]
        for layer in range(self.layers):
            prefix = f'blocks.{layer}.'
            qkv = self.norm(x, w[prefix + 'norm1.weight']) @ w[prefix + 'qkv.weight'].T
            q, k, v = [a.reshape(self.cells, self.heads, width // self.heads).transpose(1, 0, 2)
                       for a in np.split(qkv, 3, axis=-1)]
            scores = (q @ k.transpose(0, 2, 1)) * (width // self.heads) ** -.5
            scores = np.where(mask[None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= attention.sum(axis=-1, keepdims=True)
            value = (attention @ v).transpose(1, 0, 2).reshape(self.cells, width)
            attention_update = value @ w[prefix + 'projection.weight'].T
            x += attention_update
            hidden = self.norm(x, w[prefix + 'norm2.weight']) @ w[prefix + 'up.weight'].T
            activated = .5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + .044715 * hidden ** 3)))
            local_update = activated @ w[prefix + 'down.weight'].T
            x += local_update
            if observer is not None:
                observer(layer, attention, v, w[prefix + 'projection.weight'], attention_update, local_update)
            if trace:
                snapshots.append(x.copy())
        logits = self.norm(x[self.outputs], w['norm.weight']) @ w['embedding.weight'].T
        return logits, x, snapshots

    def inspect(self, prefix, ablated=False, feature=0):
        """Capture actual per-layer attention and one fixed feature, without replay."""
        if self.config.get('architecture') == 'ordinary-attention-control':
            raise ValueError('The control has no fly-cell inspection')
        width = self.weights['embedding.weight'].shape[1]
        if not 0 <= feature < width:
            raise ValueError('Feature index is outside the model width')
        selected = np.concatenate([self.inputs[-8:], self.outputs[-16:]])
        window = np.zeros(128, np.int64)
        context = list(prefix)[-128:]
        if context:
            window[-len(context):] = context
        edge_layers = []
        effective_mask = self.attention_mask(window, ablated)

        def observe(layer, attention, values, projection, attention_update, local_update):
            # Exact contribution to the chosen feature after output projection.
            # Local feed-forward and residual terms are reported separately below.
            projected = np.sum(values * projection[feature].reshape(self.heads, 1, -1), axis=-1)
            contributions = np.sum(attention * projected[:, None, :], axis=0)
            edges = []
            for post_index, post in enumerate(selected):
                for pre_index, pre in enumerate(selected):
                    if effective_mask[post, pre] and post != pre:
                        edges.append({'from': pre_index, 'to': post_index,
                                      'meanAttention': float(attention[:, post, pre].mean()),
                                      'featureContribution': float(contributions[post, pre])})
            edge_layers.append({'layer': layer, 'edges': edges,
                                'allAttentionFeatureUpdates': attention_update[selected, feature].tolist(),
                                'summedEdgeFeatureUpdates': contributions.sum(axis=1)[selected].tolist(),
                                'localFeedForwardFeatureUpdates': local_update[selected, feature].tolist(),
                                'localSelfFeatureUpdates': np.diag(contributions)[selected].tolist()})

        logits, final, snapshots = self.forward(window, ablated, trace=True, observer=observe)
        probabilities = np.exp(logits[-1].astype(np.float64) - logits[-1].max())
        probabilities /= probabilities.sum()
        top = np.argsort(probabilities)[-5:][::-1]
        states = [s[selected, feature].tolist() for s in snapshots]
        return {'architecture': 'source-edge-attention', 'featureIndex': feature, 'featureWidth': width,
                'cells': [{'id': str(self.config['neurons'][cell][0]), 'index': int(cell),
                           'input': bool(cell in self.inputs)} for cell in selected],
                'states': states, 'layers': edge_layers,
                'generationReadoutId': str(self.config['neurons'][self.outputs[-1]][0]),
                'finalFeatureRms': np.sqrt(np.mean(final * final, axis=1)).tolist(),
                'predictions': [{'tokenId': int(i), 'token': self.tokenizer.decode([int(i)], skip_special_tokens=False),
                                 'logit': float(logits[-1, i]), 'probability': float(probabilities[i])} for i in top],
                'labels': {'states': 'One engineered feature per source cell, captured before and after each graph block.',
                           'edges': 'Actual signed attention-update contributions to this feature; only selected source edges shown.',
                           'overlay': 'RMS over all final engineered features; continuous model values, not spikes.',
                           'readout': 'Vocabulary logits use all features of the final readout cell, not only the displayed feature.'}}

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
        return self.generate_tokens(prefix, max_tokens, ablated)

    def generate_tokens(self, prefix, max_tokens=64, ablated=False):
        prefix = list(prefix)
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

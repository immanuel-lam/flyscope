"""Assistant-only fine-tuning on real source replies, using CPU-shaped windows."""
import argparse
import hashlib
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import GraphLanguageModel, AttentionControl, ROOT
from dialogue_data import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run-1')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    data = ROOT / 'data/graph-language'
    run = data / args.run
    base = json.loads((run / 'training.json').read_text())
    mx.random.seed(163)
    rng = np.random.default_rng(163)
    wiring = dict(np.load(run / 'wiring.npz'))
    if base.get('architecture') == 'control':
        model = AttentionControl(width=base['arguments']['width'], layers=base['arguments']['layers'])
    else:
        model = GraphLanguageModel(width=base['arguments']['width'], layers=base['arguments']['layers'],
                                  wiring_data=tuple(wiring[k] for k in ['inputs','outputs','slots','mask','sourceMask']))
    model.load_weights(str(run / 'weights.safetensors'))
    base_sha = hashlib.sha256((run / 'weights.safetensors').read_bytes()).hexdigest()
    optimizer = optim.AdamW(learning_rate=.0001, weight_decay=.01)
    train = np.load(data / 'train.npy', mmap_mode='r')
    valid = np.load(data / 'validation.npy', mmap_mode='r')
    train_spans = np.load(data / 'train-reply-spans.npy')
    valid_spans = np.load(data / 'validation-reply-spans.npy')
    validation = tuple(mx.array(a) for a in sample(valid, valid_spans, np.random.default_rng(167), 64))

    def loss(m, x, y, mask):
        values = nn.losses.cross_entropy(m(x), y, reduction='none')
        return mx.sum(values * mask) / mx.maximum(mx.sum(mask), 1)

    grad = nn.value_and_grad(model, loss)

    def evaluate():
        values = []
        for start in range(0, 64, 8):
            x, y, mask = [v[start:start + 8] for v in validation]
            values.append(loss(model, x, y, mask))
        return float(mx.mean(mx.stack(values)))

    best = initial = evaluate()
    history = []
    started = time.perf_counter()
    model.save_weights(str(run / 'dialogue.safetensors'))
    print(json.dumps({'initialAssistantLoss': initial}), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.learning_rate = .0001 * min(step / 100, 1)
        batch = tuple(mx.array(a) for a in sample(train, train_spans, rng, args.batch))
        value, gradients = grad(model, *batch)
        gradients, _ = optim.clip_grad_norm(gradients, 1)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, value)
        if step == 1 or step % 200 == 0 or step == args.steps:
            val = evaluate()
            row = {'step': step, 'loss': float(value), 'validationLoss': val,
                   'seconds': time.perf_counter() - started}
            history.append(row)
            if val < best:
                best = val
                model.save_weights(str(run / 'dialogue.safetensors'))
            report = {'arguments': {**base['arguments'], **vars(args)}, 'seed': 163,
                      'architecture': base.get('architecture', 'graph'),
                      'parameters': base['parameters'], 'baseCheckpointSha256': base_sha,
                      'initialValidationLoss': initial, 'bestValidationLoss': best,
                      'history': history, 'assumptions': base['assumptions'],
                      'objective': 'Assistant token cross entropy, conversation-bounded left-padded windows; uniform reply and within-reply target sampling.',
                      'source': json.loads((data / 'dialogue-source.json').read_text())}
            (run / 'dialogue-training.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

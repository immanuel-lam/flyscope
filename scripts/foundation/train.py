"""Adapt attention projections to source wiring while retaining local language weights."""
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import CircuitFoundation, ROOT


def sample(tokens, spans, rng, batch):
    x = np.zeros((batch, 128), np.int32)
    y = np.zeros_like(x)
    valid = np.zeros(x.shape, bool)
    mask = np.zeros(x.shape, np.float32)
    for row, index in enumerate(rng.integers(0, len(spans), batch)):
        begin, reply, end = map(int, spans[index])
        target = int(rng.integers(reply, end))
        start = max(begin, target - 128)
        count = target - start
        x[row, -count:] = tokens[start:target]
        y[row, -count:] = tokens[start + 1:target + 1]
        valid[row, -count:] = True
        positions = np.arange(start + 1, target + 1)
        mask[row, -count:] = (positions >= reply) & (positions < end)
    return tuple(mx.array(a) for a in (x, y, valid, mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='adapt-1')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    root = ROOT / 'data/foundation'
    out = root / args.run
    out.mkdir(exist_ok=True)
    if (out / 'model.safetensors').exists():
        raise FileExistsError('Use a new directory to preserve existing runs')
    mx.random.seed(191)
    rng = np.random.default_rng(191)
    model = CircuitFoundation()
    model.set_dtype(mx.float32)
    model.freeze()
    for layer in model.base.model.layers:
        layer.self_attn.unfreeze()
    trainable = tree_flatten(model.trainable_parameters())
    if any('.self_attn.' not in key for key, _ in trainable):
        raise ValueError('Only attention projections should be trainable')
    frozen_embedding = model.base.model.embed_tokens.weight
    frozen_mlp = model.base.model.layers[0].mlp.gate_proj.weight
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=.01)
    train = np.load(root / 'corpus/train.npy', mmap_mode='r')
    train_spans = np.load(root / 'corpus/train-reply-spans.npy')
    validation = sample(np.load(root / 'corpus/validation.npy', mmap_mode='r'),
                        np.load(root / 'corpus/validation-reply-spans.npy'), np.random.default_rng(193), 16)

    def loss(m, x, y, valid, mask):
        values = nn.losses.cross_entropy(m(x, valid, all_logits=True), y, reduction='none')
        return mx.sum(values * mask) / mx.maximum(mx.sum(mask), 1)

    grad = nn.value_and_grad(model, loss)

    def evaluate():
        return float(mx.mean(mx.stack([loss(model, *(v[i:i + 1] for v in validation)) for i in range(16)])))

    for filename in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        shutil.copyfile(root / 'smollm2-135m' / filename, out / filename)
    shutil.copyfile(ROOT / 'data/graph-language/run-2/wiring.npz', out / 'wiring.npz')
    shutil.copyfile(ROOT / 'data/language/manifest.json', out / 'source-cells.json')
    initial = best = evaluate()
    started = time.perf_counter()
    history = []
    model.base.save_weights(str(out / 'model.safetensors'))
    print(json.dumps({'initialValidationLoss': initial, 'trainableParameters': sum(v.size for _, v in trainable)}), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.learning_rate = args.lr * min(step / 20, 1)
        value, gradients = grad(model, *sample(train, train_spans, rng, args.batch))
        gradients, norm = optim.clip_grad_norm(gradients, 1)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, value, norm)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise FloatingPointError('Training produced a nonfinite loss or gradient')
        if step == 1 or step % 100 == 0 or step == args.steps:
            val = evaluate()
            row = {'step': step, 'loss': float(value), 'gradientNorm': float(norm),
                   'validationLoss': val, 'seconds': time.perf_counter() - started,
                   'peakMemoryBytes': mx.get_peak_memory()}
            history.append(row)
            if val < best:
                best = val
                temporary = out / 'temporary.safetensors'
                model.base.save_weights(str(temporary))
                temporary.replace(out / 'model.safetensors')
            report = {'arguments': vars(args), 'seed': 191, 'initialValidationLoss': initial,
                      'bestValidationLoss': best, 'history': history,
                      'totalParameters': sum(v.size for _, v in tree_flatten(model.parameters())),
                      'trainableParameters': sum(v.size for _, v in trainable),
                      'trainableKeys': [key for key, _ in trainable],
                      'frozenEmbeddingUnchanged': bool(mx.all(model.base.model.embed_tokens.weight == frozen_embedding)),
                      'frozenMlpExampleUnchanged': bool(mx.all(model.base.model.layers[0].mlp.gate_proj.weight == frozen_mlp)),
                      'source': json.loads((root / 'corpus/source.json').read_text()),
                      'foundation': json.loads((ROOT / 'scripts/foundation/source.json').read_text()),
                      'assumptions': 'Source mask and verified relay fixed. Only attention projections adapt; embeddings and local MLP language weights frozen. Validation is held out from adaptation, not necessarily from foundation training.'}
            (out / 'training.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

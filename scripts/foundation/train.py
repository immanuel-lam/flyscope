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


CONTEXT_BANDS = [(1, 31), (32, 63), (64, 95), (96, None)]


def make_context_pools(spans, early_tokens=16):
    begin, reply, end = spans.T
    groups = []
    for early in [False, True]:
        stop = np.minimum(end, reply + early_tokens) if early else end
        bands = []
        for low, high in CONTEXT_BANDS:
            first = np.maximum(reply, begin + low)
            last = np.minimum(stop, begin + high + 1) if high is not None else stop
            indices = np.flatnonzero(first < last)
            if not len(indices):
                raise ValueError(f'No source targets for context band {low}:{high}, early={early}')
            bands.append((indices, first[indices], last[indices]))
        groups.append(bands)
    return groups


def sample(tokens, spans, rng, batch, early_fraction=0., early_tokens=16, context_pools=None):
    x = np.zeros((batch, 128), np.int32)
    y = np.zeros_like(x)
    valid = np.zeros(x.shape, bool)
    mask = np.zeros(x.shape, np.float32)
    indices = rng.integers(0, len(spans), batch) if context_pools is None else [None] * batch
    for row, index in enumerate(indices):
        if context_pools is not None:
            early = int(early_fraction > 0 and rng.random() < early_fraction)
            pool = context_pools[early][int(rng.integers(0, len(CONTEXT_BANDS)))]
            item = int(rng.integers(0, len(pool[0])))
            index = int(pool[0][item])
            target = int(rng.integers(pool[1][item], pool[2][item]))
        begin, reply, end = map(int, spans[index])
        if context_pools is None:
            target_end = min(end, reply + early_tokens) if early_fraction > 0 and rng.random() < early_fraction else end
            target = int(rng.integers(reply, target_end))
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
    parser.add_argument('--initial-run', help='Start a separate run from an existing checkpoint')
    parser.add_argument('--seed', type=int, default=191)
    parser.add_argument('--target', choices=['all', 'last'], default='all')
    parser.add_argument('--validation-examples', type=int, default=16)
    parser.add_argument('--early-fraction', type=float, default=0.)
    parser.add_argument('--distill', action='store_true', help='Add training-only foundation distribution and state targets')
    parser.add_argument('--balanced-context', action='store_true', help='Balance actual input-length bands using source targets')
    parser.add_argument('--teacher-run', default='smollm2-135m')
    parser.add_argument('--feature-weight', type=float, default=.05)
    parser.add_argument('--teacher-trim-padding', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.early_fraction <= 1 or (args.early_fraction and args.target != 'last'):
        parser.error('--early-fraction must be between zero and one and requires --target last')
    if args.distill and args.target != 'last':
        parser.error('--distill requires --target last')
    if args.feature_weight < 0:
        parser.error('--feature-weight must be nonnegative')
    root = ROOT / 'data/foundation'
    out = root / args.run
    out.mkdir(exist_ok=True)
    if (out / 'model.safetensors').exists():
        raise FileExistsError('Use a new directory to preserve existing runs')
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    initial_directory = root / args.initial_run if args.initial_run else root / 'smollm2-135m'
    initial_hash = hashlib.sha256((initial_directory / 'model.safetensors').read_bytes()).hexdigest()
    model = CircuitFoundation(initial_directory)
    model.set_dtype(mx.float32)
    reference = None
    teacher_metadata = None
    if args.distill:
        from mlx_lm import load
        from distillation import loss_terms
        teacher_directory = root / args.teacher_run
        if json.loads((teacher_directory / 'tokenizer.json').read_text()) != json.loads((root / 'smollm2-135m/tokenizer.json').read_text()):
            raise ValueError('Teacher tokenizer must match every student token ID and tokenizer operation')
        reference = load(str(teacher_directory))[0]
        reference.set_dtype(mx.float32)
        reference.freeze()
        with (teacher_directory / 'model.safetensors').open('rb') as handle:
            teacher_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        teacher_metadata = {'run': args.teacher_run, 'weightSha256': teacher_hash,
                            'parameters': sum(v.size for _, v in tree_flatten(reference.parameters())),
                            'trainingOnly': True, 'dtype': 'float32', 'tokenizerMatchesStudent': True}
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
    train_pools = make_context_pools(train_spans) if args.balanced_context else None
    validation_tokens = np.load(root / 'corpus/validation.npy', mmap_mode='r')
    validation_spans = np.load(root / 'corpus/validation-reply-spans.npy')
    validation_pools = make_context_pools(validation_spans) if args.balanced_context else None
    validation = sample(validation_tokens, validation_spans, np.random.default_rng(193), args.validation_examples)
    early_validation = (sample(validation_tokens, validation_spans,
                               np.random.default_rng(223), args.validation_examples, early_fraction=1.)
                        if args.early_fraction else None)
    balanced_validation = (sample(validation_tokens, validation_spans, np.random.default_rng(233),
                                  args.validation_examples, args.early_fraction, context_pools=validation_pools)
                           if args.balanced_context else None)

    def loss(m, x, y, valid, mask):
        if reference is not None:
            ce, kl, state_mse = loss_terms(m, reference, x, y[:, -1], valid,
                                         match_features=args.feature_weight > 0,
                                         trim_reference_padding=args.teacher_trim_padding)
            return .5 * ce + .5 * kl + args.feature_weight * state_mse
        if args.target == 'last':
            return nn.losses.cross_entropy(m(x, valid), y[:, -1], reduction='mean')
        values = nn.losses.cross_entropy(m(x, valid, all_logits=True), y, reduction='none')
        return mx.sum(values * mask) / mx.maximum(mx.sum(mask), 1)

    grad = nn.value_and_grad(model, loss)

    def evaluate():
        def measure(batch):
            # Finish each validation example before building the next graph;
            # a larger frozen teacher must not retain a whole validation set.
            values = [float(loss(model, *(v[i:i + 1] for v in batch))) for i in range(args.validation_examples)]
            return float(mx.mean(mx.array(values, dtype=mx.float32)))
        uniform = measure(validation)
        early = measure(early_validation) if early_validation is not None else None
        balanced = measure(balanced_validation) if balanced_validation is not None else None
        combined = (1 - args.early_fraction) * uniform + args.early_fraction * early if early is not None else uniform
        return {'validationLoss': balanced if balanced is not None else combined,
                'validationUniformLoss': uniform, 'validationEarlyLoss': early, 'validationBalancedLoss': balanced}

    for filename in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        shutil.copyfile(root / 'smollm2-135m' / filename, out / filename)
    shutil.copyfile(ROOT / 'data/graph-language/run-2/wiring.npz', out / 'wiring.npz')
    shutil.copyfile(ROOT / 'data/language/manifest.json', out / 'source-cells.json')
    initial_measures = evaluate()
    initial = best = initial_measures['validationLoss']
    started = time.perf_counter()
    history = []
    model.base.save_weights(str(out / 'model.safetensors'))
    print(json.dumps({'initialValidationLoss': initial, 'trainableParameters': sum(v.size for _, v in trainable)}), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.learning_rate = args.lr * min(step / 20, 1)
        value, gradients = grad(model, *sample(train, train_spans, rng, args.batch, args.early_fraction, context_pools=train_pools))
        gradients, norm = optim.clip_grad_norm(gradients, 1)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, value, norm)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise FloatingPointError('Training produced a nonfinite loss or gradient')
        if step == 1 or step % 100 == 0 or step == args.steps:
            measures = evaluate()
            val = measures['validationLoss']
            row = {'step': step, 'loss': float(value), 'gradientNorm': float(norm),
                   **measures, 'seconds': time.perf_counter() - started,
                   'peakMemoryBytes': mx.get_peak_memory()}
            history.append(row)
            if val < best:
                best = val
                temporary = out / 'temporary.safetensors'
                model.base.save_weights(str(temporary))
                temporary.replace(out / 'model.safetensors')
            report = {'arguments': vars(args), 'seed': args.seed, 'initialCheckpointSha256': initial_hash,
                      'initialValidationLoss': initial,
                      'initialValidationMeasures': initial_measures,
                      'contextSampling': {'balanced': args.balanced_context, 'bands': CONTEXT_BANDS,
                                          'trainPoolSizes': [[len(p[0]) for p in g] for g in train_pools] if train_pools else None,
                                          'validationPoolSizes': [[len(p[0]) for p in g] for g in validation_pools] if validation_pools else None},
                      'objective': {'crossEntropyWeight': .5 if args.distill else 1.,
                                    'teacherKlWeight': .5 if args.distill else 0.,
                                    'teacherFeatureMseWeight': args.feature_weight if args.distill else 0.,
                                    'teacherTrainingOnly': args.distill,
                                    'validationMeaning': 'Same declared objective and sampling mixture; not necessarily pure token cross entropy'},
                      'teacher': teacher_metadata,
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

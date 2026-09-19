"""Measure storage-induced prediction changes on fixed adaptation validation targets."""
import argparse
import hashlib
import json
import numpy as np
from runtime import FoundationRuntime, ROOT


def distribution(logits):
    z = logits.astype(np.float64) - np.max(logits)
    return z - np.log(np.exp(z).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--examples', type=int, default=32)
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    packed = directory.with_name(directory.name + '-q8')
    reference, candidate = FoundationRuntime(directory), FoundationRuntime(packed)
    np.testing.assert_array_equal(reference.mask, candidate.mask)
    np.testing.assert_array_equal(reference.inputs, candidate.inputs)
    np.testing.assert_array_equal(reference.outputs, candidate.outputs)
    assert reference.ids == candidate.ids
    rng = np.random.default_rng(211)
    tokens = np.load(ROOT / 'data/foundation/corpus/validation.npy', mmap_mode='r')
    spans = np.load(ROOT / 'data/foundation/corpus/validation-reply-spans.npy')
    rows = []
    for index in rng.integers(0, len(spans), args.examples):
        begin, reply, end = map(int, spans[index])
        target = int(rng.integers(reply, end))
        prefix = tokens[max(begin, target - 128):target].tolist()
        a, _, _ = reference.next(prefix)
        b, _, _ = candidate.next(prefix)
        la, lb = distribution(a), distribution(b)
        row = {'spanIndex': int(index), 'targetOffset': target, 'targetToken': int(tokens[target]),
               'referenceLoss': float(-la[tokens[target]]), 'quantizedLoss': float(-lb[tokens[target]]),
               'topTokenMatches': bool(np.argmax(a) == np.argmax(b)),
               'referenceToQuantizedKl': float(np.sum(np.exp(la) * (la - lb))),
               'logitRmsDifference': float(np.sqrt(np.mean((a - b) ** 2)))}
        rows.append(row)
    cut, _, _ = candidate.next([50, 80, 60], ablated=True)
    np.testing.assert_array_equal(cut, 0)
    report = {'run': args.run, 'examples': args.examples, 'seed': 211,
              'sourceWeightSha256': hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest(),
              'storageManifestSha256': hashlib.sha256((packed / 'weights.json').read_bytes()).hexdigest(),
              'referenceMeanLoss': float(np.mean([r['referenceLoss'] for r in rows])),
              'quantizedMeanLoss': float(np.mean([r['quantizedLoss'] for r in rows])),
              'meanKl': float(np.mean([r['referenceToQuantizedKl'] for r in rows])),
              'topTokenAgreement': float(np.mean([r['topTokenMatches'] for r in rows])),
              'cutMaxAbsoluteLogit': float(np.max(np.abs(cut))), 'rows': rows,
              'limitation': 'Teacher-forced next-token storage comparison on adaptation validation only; not conversational quality or deployment verification. Candidate remains rejected for chat quality.'}
    (packed / 'storage-evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()

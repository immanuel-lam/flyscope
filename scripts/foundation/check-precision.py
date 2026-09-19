"""Reproduce CPU normalization precision comparisons without relaxing tolerances."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import argparse
import copy
import hashlib
import json
import time
from unittest.mock import patch
import numpy as np
import mlx.core as mx
from model import CircuitFoundation
from runtime import FoundationRuntime, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--prompt', default='What is a cat?')
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    cpu = FoundationRuntime(directory)
    model = CircuitFoundation(directory)
    model.set_dtype(mx.float32)
    prefix = cpu.tokenizer.encode(args.prompt).ids[-128:]
    tokens = np.zeros((1, 128), np.int32)
    valid = np.zeros((1, 128), bool)
    tokens[0, -len(prefix):], valid[0, -len(prefix):] = prefix, True
    ml, ms = model(mx.array(tokens), mx.array(valid), return_states=True)
    mx.eval(ml, ms)
    ml, ms = np.array(ml)[0], np.array(ms)[0]
    old = copy.copy(cpu)
    old.norm = lambda x, w: x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + cpu.config['rms_norm_eps']) * w
    def higher_norm(x, w):
        value = x.astype(np.float64)
        return (value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True)
                                + cpu.config['rms_norm_eps']) * w).astype(x.dtype)
    cpu.norm = higher_norm
    results = {}
    values = {}
    for label, instance in [('originalFloat32Norm', old), ('higherPrecisionNorm', cpu)]:
        start = time.perf_counter()
        logits, states, _ = instance.next(prefix)
        elapsed = time.perf_counter() - start
        values[label] = (logits, states)
        results[label] = {'stateToleranceViolations': int(np.count_nonzero(~np.isclose(states, ms, atol=.003, rtol=.0003))),
                          'maxAbsoluteStateDifferenceFromMlx': float(np.max(np.abs(states - ms))),
                          'maxAbsoluteLogitDifferenceFromMlx': float(np.max(np.abs(logits - ml))),
                          'computeSeconds': elapsed}
    reference = copy.copy(cpu)
    reference.weights = {k: v.astype(np.float64) for k, v in cpu.weights.items()}
    zeros = np.zeros
    def higher_precision_state(shape, *positional, **named):
        if shape == (512, cpu.config['hidden_size']):
            return zeros(shape, dtype=np.float64)
        return zeros(shape, *positional, **named)
    with patch('runtime.np.zeros', side_effect=higher_precision_state):
        rl, rs, _ = reference.next(prefix)
    for label, (logits, states) in values.items():
        results[label]['maxCellRelativeL2VsFloat64State'] = float(np.max(np.linalg.norm(states - rs, axis=-1) / np.maximum(np.linalg.norm(rs, axis=-1), 1)))
        results[label]['maxLogitDifferenceVsFloat64State'] = float(np.max(np.abs(logits - rl)))
    report = {'run': args.run, 'prompt': args.prompt, 'results': results,
              'weightSha256': hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest(),
              'runtimeSha256': hashlib.sha256((ROOT / 'scripts/foundation/runtime.py').read_bytes()).hexdigest(),
              'stateTolerance': {'absolute': .003, 'relative': .0003},
              'referenceMeaning': 'Diagnostic CPU weights and states use float64. Rotary constants remain float32. Higher-precision normalization is an experimental variant, not production behavior. Production arithmetic and test tolerances are unchanged. Timings are single local calls, not a deployment benchmark.'}
    (directory / 'normalization-precision.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

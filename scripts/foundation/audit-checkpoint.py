"""Verify actual saved parameter changes against the pinned foundation."""
import argparse
import hashlib
import json
import numpy as np
from runtime import ROOT, load_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    original_file = ROOT / 'data/foundation/smollm2-135m/model.safetensors'
    saved_file = directory / 'model.safetensors'
    original, saved = load_weights(original_file), load_weights(saved_file)
    if original.keys() != saved.keys():
        raise ValueError('Checkpoint tensor names changed')
    rows = []
    for key, expected in original.items():
        actual = saved[key]
        if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
            raise ValueError(f'Invalid saved tensor: {key}')
        allowed = '.self_attn.' in key
        changed = int(np.count_nonzero(actual != expected))
        if changed and not allowed:
            raise ValueError(f'A frozen tensor changed: {key}')
        rows.append({'name': key, 'parameters': actual.size, 'allowedToChange': allowed,
                     'changedParameters': changed, 'maxAbsoluteChange': float(np.max(np.abs(actual - expected)))})
    wiring = directory / 'wiring.npz'
    cells = directory / 'source-cells.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(wiring) != sha(ROOT / 'data/graph-language/run-2/wiring.npz'):
        raise ValueError('Source wiring changed')
    if sha(cells) != sha(ROOT / 'data/language/manifest.json'):
        raise ValueError('Source cell manifest changed')
    report = {'run': args.run, 'referenceWeightSha256': sha(original_file), 'weightSha256': sha(saved_file),
              'wiringSha256': sha(wiring), 'sourceCellsSha256': sha(cells),
              'totalParameters': sum(r['parameters'] for r in rows),
              'allowedParameters': sum(r['parameters'] for r in rows if r['allowedToChange']),
              'changedParameters': sum(r['changedParameters'] for r in rows),
              'allFrozenTensorsUnchanged': True, 'sourceWiringAndCellsUnchanged': True, 'tensors': rows}
    (directory / 'checkpoint-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()

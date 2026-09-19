"""Export source-wired CPU weights with symmetric per-row int8 storage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from runtime import ROOT, load_weights


def quantize(directory, destination):
    destination.mkdir(exist_ok=False)
    original = directory / 'model.safetensors'
    weights = load_weights(original)
    manifest = {'format': 'flyscope-row-int8-v1', 'sourceWeightSha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'inference': 'Weights are decoded to float32 once at load. All inference uses NumPy CPU; this saves disk and bundle size, not resident weight memory.',
                'tensors': {}}
    for index, (key, values) in enumerate(sorted(weights.items())):
        if not np.all(np.isfinite(values)):
            raise ValueError('Cannot quantize nonfinite weights')
        filename = f'weight-{index:03d}.npz'
        if values.ndim == 2:
            scales = np.max(np.abs(values), axis=-1, keepdims=True) / 127
            scales = np.where(scales > 0, scales, 1).astype(np.float32)
            quantized = np.clip(np.rint(values / scales), -127, 127).astype(np.int8)
            np.savez(destination / filename, values=quantized, scales=scales)
            error = float(np.max(np.abs(quantized.astype(np.float32) * scales - values)))
            storage = 'row-int8'
        else:
            np.savez(destination / filename, values=values.astype(np.float32))
            error, storage = 0., 'float32'
        manifest['tensors'][key] = {'file': filename, 'shape': list(values.shape), 'storage': storage,
                                   'maxAbsoluteWeightError': error,
                                   'sha256': hashlib.sha256((destination / filename).read_bytes()).hexdigest()}
    for filename in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'wiring.npz', 'source-cells.json']:
        shutil.copyfile(directory / filename, destination / filename)
    source = np.load(ROOT / 'data/language/graph.npz')['counts'] > 0
    np.save(destination / 'source-mask.npy', source)
    manifest['sourceGraphSha256'] = hashlib.sha256((ROOT / 'data/language/graph.npz').read_bytes()).hexdigest()
    manifest['originalWeightBytes'] = original.stat().st_size
    manifest['storedTensorBytes'] = sum((destination / t['file']).stat().st_size for t in manifest['tensors'].values())
    (destination / 'weights.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    report = quantize(ROOT / 'data/foundation' / args.run, ROOT / 'data/foundation' / f'{args.run}-q8')
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2))

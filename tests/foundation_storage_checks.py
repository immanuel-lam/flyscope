"""Check int8 encoding error bounds and reject altered archive bytes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/foundation'))
from runtime import load_weights, load_quantized_weights


class StorageChecks(unittest.TestCase):
    def test_quantized_checkpoint_error_is_bounded_by_half_a_row_step(self):
        original = load_weights(ROOT / 'data/foundation/adapt-1/model.safetensors')
        restored = load_quantized_weights(ROOT / 'data/foundation/adapt-1-q8')
        self.assertEqual(set(original), set(restored))
        for key, value in original.items():
            if value.ndim == 2:
                step = np.max(np.abs(value), axis=-1, keepdims=True) / 127
                error = np.abs(value - restored[key])
                self.assertTrue(np.all(error <= step / 2 + 2e-6), key)
            else:
                np.testing.assert_array_equal(value, restored[key])

    def test_checksum_and_path_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'weight-000.npz'
            np.savez(path, values=np.ones(2, np.float32))
            spec = {'file': path.name, 'shape': [2], 'storage': 'float32',
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            report = {'format': 'flyscope-row-int8-v1', 'tensors': {'test': spec}}
            (root / 'weights.json').write_text(json.dumps(report))
            np.testing.assert_array_equal(load_quantized_weights(root)['test'], 1)
            spec['sha256'] = '0' * 64
            (root / 'weights.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_quantized_weights(root)
            spec['file'] = '../weight-000.npz'
            (root / 'weights.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'filename'):
                load_quantized_weights(root)


if __name__ == '__main__':
    unittest.main()

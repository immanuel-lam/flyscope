"""Verify balanced contexts retain real targets and conversation boundaries."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/foundation'))
from train import make_context_pools, sample


class ContextChecks(unittest.TestCase):
    def test_balanced_contexts_keep_source_targets(self):
        tokens = np.arange(1000, dtype=np.int32)
        spans = np.array([[0, 10, 150], [200, 240, 360], [400, 475, 600], [700, 810, 990]])
        pools = make_context_pools(spans)
        for fraction in [0., .5, 1.]:
            x, y, valid, mask = map(np.array, sample(tokens, spans, np.random.default_rng(239), 4000,
                                                    fraction, context_pools=pools))
            counts = np.bincount(np.digitize(valid.sum(axis=1), [32, 64, 96]), minlength=4)
            self.assertTrue(np.all((counts > 850) & (counts < 1150)), counts)
            for row in range(len(x)):
                target = int(y[row, -1])
                begin, reply, end = next(s for s in spans if s[1] <= target < s[2])
                np.testing.assert_array_equal(x[row, valid[row]], np.arange(max(begin, target - 128), target))
                self.assertEqual(mask[row, -1], 1)
                if fraction == 1.:
                    self.assertLess(target, reply + 16)

    def test_missing_band_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'No source targets'):
            make_context_pools(np.array([[0, 200, 300]]))


if __name__ == '__main__':
    unittest.main()

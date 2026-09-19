"""Ensure the diagnostic stays causal and does not claim fly-cell inspection."""
from pathlib import Path
import sys
import unittest
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/graph-language'))
from model import AttentionControl
from runtime import GraphRuntime


class ControlChecks(unittest.TestCase):
    def test_control_is_causal(self):
        mx.random.seed(181)
        model = AttentionControl(width=32, layers=2, heads=4)
        a = mx.full((1, 128), 75)
        b = a.at[:, 65:].add(100)
        x, y = model(a), model(b)
        mx.eval(x, y)
        np.testing.assert_allclose(np.array(x[:, :65]), np.array(y[:, :65]), atol=1e-5)
        self.assertGreater(float(mx.max(mx.abs(x[:, 65:] - y[:, 65:]))), .01)

    def test_control_cannot_present_fly_inspection(self):
        model = GraphRuntime(ROOT / 'data/graph-language/control-pilot/portable')
        with self.assertRaisesRegex(ValueError, 'no fly-cell'):
            model.inspect([2, 75, 4, 3])


if __name__ == '__main__':
    unittest.main()

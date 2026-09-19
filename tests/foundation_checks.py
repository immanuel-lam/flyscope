"""Check numerical correspondence and mandatory source-path dependence."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
from pathlib import Path
import sys
import unittest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/foundation'))
from model import CircuitFoundation
from runtime import FoundationRuntime


class FoundationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = FoundationRuntime()
        cls.model = CircuitFoundation()
        cls.model.set_dtype(mx.float32)

    def test_cpu_matches_source_model(self):
        prefix = self.cpu.tokenizer.encode('What is a cat?').ids
        tokens = np.zeros((1, 128), np.int32)
        valid = np.zeros((1, 128), bool)
        tokens[0, -len(prefix):], valid[0, -len(prefix):] = prefix, True
        expected, states = self.model(mx.array(tokens), mx.array(valid), return_states=True)
        mx.eval(expected, states)
        actual, cpu_states, _ = self.cpu.next(prefix)
        np.testing.assert_allclose(actual, np.array(expected)[0], atol=3e-3, rtol=3e-4)
        np.testing.assert_allclose(cpu_states, np.array(states)[0], atol=3e-3, rtol=3e-4)

    def test_ordinary_cpu_matches_original_library_model(self):
        prefix = self.cpu.tokenizer.encode('What is a cat?').ids
        expected = self.model.base(mx.array([prefix]))[0, -1]
        mx.eval(expected)
        actual, _, _ = self.cpu.next(prefix, graph=False)
        np.testing.assert_allclose(actual, np.array(expected), atol=3e-3, rtol=3e-4)

    def test_future_tokens_cannot_change_earlier_predictions(self):
        a = mx.full((1, 128), 75, dtype=mx.int32)
        b = a.at[:, 65:].add(100)
        x = self.model(a, all_logits=True)
        y = self.model(b, all_logits=True)
        mx.eval(x, y)
        np.testing.assert_allclose(np.array(x[:, :65]), np.array(y[:, :65]), atol=3e-3, rtol=3e-4)

    def test_cut_removes_all_prompt_information(self):
        a = self.cpu.tokenizer.encode('What is a cat?').ids
        b = self.cpu.tokenizer.encode('What is a tree?').ids
        x, states, _ = self.cpu.next(a, ablated=True)
        y, _, _ = self.cpu.next(b, ablated=True)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(states[self.cpu.outputs], 0)
        full, _, _ = self.cpu.next(a)
        changed, _, _ = self.cpu.next(b)
        self.assertGreater(float(np.max(np.abs(full - changed))), .01)
        # A relay alone must not be mistaken for context processing.
        relay_a, _, _ = self.cpu.next(a, attention_cut=True)
        relay_b, _, _ = self.cpu.next(b, attention_cut=True)
        np.testing.assert_array_equal(relay_a, relay_b)

    def test_relay_is_explicit_and_source_matched(self):
        _, _, trace = self.cpu.next([5, 10, 15], trace=True)
        initial = trace['injected']
        np.testing.assert_array_equal(initial[self.cpu.outputs], 0)
        np.testing.assert_array_equal(trace['blocks'][0][self.cpu.outputs], initial[self.cpu.inputs])
        self.assertEqual(len(trace['blocks']), self.cpu.config['num_hidden_layers'] + 1)

    def test_training_gradients_are_finite_with_empty_source_cells(self):
        model = CircuitFoundation()
        model.set_dtype(mx.float32)
        model.freeze()
        for layer in model.base.model.layers:
            layer.self_attn.unfreeze()
        tokens = mx.full((1, 128), 75, dtype=mx.int32)
        for valid in (mx.ones((1, 128), dtype=mx.bool_), mx.arange(128)[None] >= 120):
            def loss(m):
                return nn.losses.cross_entropy(m(tokens, valid), mx.array([100])).mean()
            value, gradients = nn.value_and_grad(model, loss)(model)
            mx.eval(value, gradients)
            self.assertTrue(np.isfinite(float(value)))
            for key, gradient in tree_flatten(gradients):
                self.assertTrue(bool(mx.all(mx.isfinite(gradient))), key)
            first = gradients['base']['model']['layers'][0]['self_attn']['v_proj']['weight']
            self.assertGreater(float(mx.max(mx.abs(first))), 0)


if __name__ == '__main__':
    unittest.main()

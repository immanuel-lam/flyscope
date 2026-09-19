"""Verify training-only reference targets and source-student gradients."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
from pathlib import Path
import sys
import unittest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/foundation'))
from model import CircuitFoundation
from distillation import reference_targets, loss_terms


class DistillationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference, cls.tokenizer = load(str(ROOT / 'data/foundation/smollm2-135m'))
        cls.reference.set_dtype(mx.float32)
        cls.prefix = cls.tokenizer.encode('What is a cat?')
        x = np.zeros((1, 128), np.int32)
        valid = np.zeros((1, 128), bool)
        x[0, -len(cls.prefix):], valid[0, -len(cls.prefix):] = cls.prefix, True
        cls.x, cls.valid = mx.array(x), mx.array(valid)

    def test_reference_padding_matches_original_unpadded_model(self):
        expected = self.reference(mx.array([self.prefix]))[:, -1]
        modified = mx.where(self.valid, self.x, mx.full(self.x.shape, 77))
        for trim in [False, True]:
            actual, _ = reference_targets(self.reference, self.x, self.valid, trim_padding=trim)
            padded, _ = reference_targets(self.reference, modified, self.valid, trim_padding=trim)
            mx.eval(actual, expected, padded)
            np.testing.assert_allclose(np.array(actual), np.array(expected), atol=.003, rtol=.0003)
            np.testing.assert_array_equal(np.array(actual), np.array(padded))

    def test_trim_rejects_noncontiguous_valid_tokens(self):
        valid = np.array(self.valid)
        valid[0, 0] = True
        with self.assertRaisesRegex(ValueError, 'left-padded'):
            reference_targets(self.reference, self.x, mx.array(valid), trim_padding=True)

    def test_reference_targets_have_no_parameter_gradient(self):
        self.reference.unfreeze()
        for trim in [False, True]:
            def objective(m):
                logits, features = reference_targets(m, self.x, self.valid, trim_padding=trim)
                return logits.sum() + features.sum()
            _, gradients = nn.value_and_grad(self.reference, objective)(self.reference)
            mx.eval(gradients)
            for key, value in tree_flatten(gradients):
                self.assertTrue(bool(mx.all(value == 0)), key)
        self.reference.freeze()

    def test_student_receives_finite_nonzero_gradients(self):
        student = CircuitFoundation()
        student.set_dtype(mx.float32)
        student.freeze()
        for layer in student.base.model.layers:
            layer.self_attn.unfreeze()
        for matching, trim in [(True, False), (False, True)]:
            def objective(m):
                ce, kl, feature = loss_terms(m, self.reference, self.x, mx.array([100]), self.valid,
                                             match_features=matching, trim_reference_padding=trim)
                return .5 * ce + .5 * kl + .05 * feature
            loss, gradients = nn.value_and_grad(student, objective)(student)
            mx.eval(loss, gradients)
            self.assertTrue(np.isfinite(float(loss)))
            for key, value in tree_flatten(gradients):
                self.assertTrue(bool(mx.all(mx.isfinite(value))), key)
            self.assertGreater(float(mx.max(mx.abs(gradients['base']['model']['layers'][0]['self_attn']['v_proj']['weight']))), 0)


if __name__ == '__main__':
    unittest.main()

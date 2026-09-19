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
from train import sample


class FoundationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('FOUNDATION_CHECKPOINT')
        cls.cpu = FoundationRuntime(directory)
        cls.model = CircuitFoundation(directory)
        cls.model.set_dtype(mx.float32)

    def test_cpu_matches_source_model(self):
        cases = ['What is a cat?',
                 '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n',
                 'A tiny fly follows an odour trail. ' * 32]
        for text in cases:
            with self.subTest(text=text[:50]):
                prefix = self.cpu.tokenizer.encode(text).ids[-128:]
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

    def test_sample_target_is_assistant_and_context_stays_in_conversation(self):
        tokens = np.arange(500, dtype=np.int32)
        spans = np.array([[20, 30, 70], [200, 230, 400]], dtype=np.int64)
        x, y, valid, mask = map(np.array, sample(tokens, spans, np.random.default_rng(13), 64))
        for row in range(len(x)):
            target = int(y[row, -1])
            begin, reply, end = next(s for s in spans if s[1] <= target < s[2])
            expected = np.arange(max(begin, target - 128), target)
            np.testing.assert_array_equal(x[row, valid[row]], expected)
            np.testing.assert_array_equal(y[row, valid[row]], expected + 1)
            self.assertEqual(mask[row, -1], 1)
            self.assertTrue(np.all(mask[row, ~valid[row]] == 0))

    def test_early_reply_sampling_retains_target_alignment(self):
        tokens = np.arange(500, dtype=np.int32)
        spans = np.array([[20, 30, 70], [200, 230, 400]], dtype=np.int64)
        x, y, valid, mask = map(np.array, sample(tokens, spans, np.random.default_rng(13), 128, early_fraction=1.))
        for row in range(len(x)):
            target = int(y[row, -1])
            begin, reply, end = next(s for s in spans if s[1] <= target < s[2])
            self.assertLess(target, min(end, reply + 16))
            self.assertEqual(x[row, -1], target - 1)
            self.assertEqual(mask[row, -1], 1)
            self.assertGreaterEqual(int(x[row, valid[row]][0]), begin)

    def test_compact_inspection_reconstructs_actual_updates(self):
        prefix = self.cpu.tokenizer.encode('What is a cat?').ids
        logits, states, _ = self.cpu.next(prefix)
        actual, observed, details = self.cpu.next(prefix, trace=True, inspect=True)
        np.testing.assert_array_equal(actual, logits)
        np.testing.assert_array_equal(observed, states)
        view = details['inspection']
        selected = np.array([c['index'] for c in view['cells']])
        np.testing.assert_allclose(view['activityRms'], np.sqrt(np.mean(states * states, axis=-1)), rtol=1e-6)
        for i, block in enumerate(view['blocks']):
            np.testing.assert_array_equal(block['before'], details['blocks'][i][selected, 0])
            np.testing.assert_array_equal(block['after'], details['blocks'][i + 1][selected, 0])
            contribution = np.array(block['selfContribution']) + np.array(block['otherCellContribution'])
            for edge in block['edges']:
                pre, post = selected[edge['source']], selected[edge['target']]
                self.assertNotEqual(pre, post)
                self.assertTrue(self.cpu.mask[post, pre])
                contribution[edge['target']] += edge['contribution']
            np.testing.assert_allclose(np.array(block['before']) + contribution,
                                       block['afterAttention'], atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(np.array(block['afterAttention']) + block['mlpUpdate'],
                                       block['after'], atol=3e-4, rtol=3e-4)
        # Independently reconstruct the largest displayed first-block edge.
        edge = max(view['blocks'][0]['edges'], key=lambda e: abs(e['contribution']))
        pre, post = selected[edge['source']], selected[edge['target']]
        w, key = self.cpu.weights, 'model.layers.0.'
        z = self.cpu.norm(details['blocks'][0], w[key + 'input_layernorm.weight'])
        q = (z @ w[key + 'self_attn.q_proj.weight'].T).reshape(512, 9, 64).transpose(1, 0, 2)
        k = (z @ w[key + 'self_attn.k_proj.weight'].T).reshape(512, 3, 64).transpose(1, 0, 2)
        v = (z @ w[key + 'self_attn.v_proj.weight'].T).reshape(512, 3, 64).transpose(1, 0, 2)
        q, k = self.cpu.rope(q, self.cpu.slots), self.cpu.rope(k, self.cpu.slots)
        allowed = self.cpu.mask[post] & (self.cpu.slots >= 128 - len(prefix))
        allowed[post] = True
        expected = 0.
        for head in range(9):
            score = k[head // 3].astype(np.float64) @ q[head, post].astype(np.float64) / 8
            score[~allowed] = -np.inf
            probability = np.exp(score - score.max())
            probability /= probability.sum()
            projected = np.dot(v[head // 3, pre], w[key + 'self_attn.o_proj.weight'][0, head * 64:(head + 1) * 64])
            expected += float(probability[pre] * projected)
        self.assertAlmostEqual(expected, edge['contribution'], places=4)

    def test_compact_inspection_distinguishes_relay_and_attention_cuts(self):
        _, _, full = self.cpu.next([50, 80, 60], inspect=True)
        _, _, cut = self.cpu.next([50, 80, 60], ablated=True, inspect=True)
        _, _, local = self.cpu.next([50, 80, 60], attention_cut=True, inspect=True)
        self.assertTrue(any(r['featureContribution'] != 0 for r in full['inspection']['relay']))
        self.assertTrue(all(r['featureContribution'] == 0 for r in cut['inspection']['relay']))
        self.assertEqual(full['inspection']['relay'], local['inspection']['relay'])
        for run in [cut, local]:
            self.assertTrue(all(e['contribution'] == 0 for b in run['inspection']['blocks'] for e in b['edges']))
        with self.assertRaisesRegex(ValueError, 'Source-cell'):
            self.cpu.next([5], graph=False, inspect=True)


if __name__ == '__main__':
    unittest.main()

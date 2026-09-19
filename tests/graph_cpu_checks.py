"""Behavioral controls for a separately exported graph-language candidate."""
from pathlib import Path
import sys
import os
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/graph-language'))
from runtime import GraphRuntime


class GraphCpuChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GraphRuntime(ROOT / 'data/graph-language' / os.environ.get('GRAPH_LANGUAGE_RUN', 'run-2') / os.environ.get('GRAPH_LANGUAGE_EXPORT', 'portable'))

    def test_all_context_slots_reach_generation_cell(self):
        reach = np.eye(512, dtype=bool)
        for _ in range(self.model.layers):
            reach = (self.model.mask.astype(np.int32) @ reach.astype(np.int32)) > 0
        self.assertTrue(np.all(reach[self.model.outputs[-1], self.model.inputs]))

    def test_future_tokens_cannot_change_past_predictions(self):
        a = np.full(128, 75)
        b = a.copy()
        b[65:] = 614
        x, _, _ = self.model.forward(a)
        y, _, _ = self.model.forward(b)
        np.testing.assert_allclose(x[:65], y[:65], atol=1e-5)
        self.assertGreater(float(np.max(np.abs(x[65:] - y[65:]))), 1e-3)

    def test_source_cut_removes_prompt_information(self):
        a, _, _ = self.model.forward(np.full(128, 75), ablated=True)
        b, _, _ = self.model.forward(np.full(128, 614), ablated=True)
        np.testing.assert_array_equal(a, b)

    def test_trace_is_computed_cell_features(self):
        _, final, trace = self.model.forward(np.full(128, 75), trace=True)
        self.assertEqual(len(trace), self.model.layers + 1)
        np.testing.assert_array_equal(final, trace[-1])
        # Output cells begin with identity features only, not input embeddings.
        np.testing.assert_array_equal(trace[0][self.model.outputs],
                                      self.model.weights['cell_embedding.weight'][self.model.outputs])
        self.assertGreater(float(np.max(np.abs(trace[-1] - trace[0]))), .01)

    def test_inspection_uses_actual_layer_updates_and_predictions(self):
        prefix = [2, 75, 614, 4, 3]
        inspection = self.model.inspect(prefix)
        logits, states = self.model.next(prefix)
        np.testing.assert_allclose(inspection['finalFeatureRms'], np.sqrt(np.mean(states * states, axis=1)), atol=1e-6)
        for i, layer in enumerate(inspection['layers']):
            before = np.array(inspection['states'][i])
            after = np.array(inspection['states'][i + 1])
            attention = np.array(layer['allAttentionFeatureUpdates'])
            local = np.array(layer['localFeedForwardFeatureUpdates'])
            np.testing.assert_allclose(layer['summedEdgeFeatureUpdates'], attention, atol=1e-5)
            np.testing.assert_allclose(before + attention + local, after, atol=1e-5)
            for edge in layer['edges']:
                pre = inspection['cells'][edge['from']]['index']
                post = inspection['cells'][edge['to']]['index']
                self.assertTrue(self.model.mask[post, pre])
        for prediction in inspection['predictions']:
            self.assertAlmostEqual(prediction['logit'], float(logits[prediction['tokenId']]), places=5)
        cut = self.model.inspect(prefix, ablated=True)
        self.assertTrue(all(not layer['edges'] for layer in cut['layers']))


if __name__ == '__main__':
    unittest.main()

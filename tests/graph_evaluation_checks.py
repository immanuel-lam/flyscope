"""Check held-out sampling and numerical loss without training or generation."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/graph-language'))
from evaluate import examples, token_loss


class EvaluationChecks(unittest.TestCase):
    def test_selection_keeps_one_reply_per_conversation_and_excludes_answers(self):
        tokens = np.arange(600)
        spans = np.array([[0, 20, 30], [0, 60, 70], [100, 120, 125], [200, 360, 380]])
        selected = examples(tokens, spans, 3)
        conversations = []
        for row in selected:
            start, reply, end = spans[row['spanIndex']]
            conversations.append(start)
            self.assertEqual(row['prefix'][-1], reply - 1)
            self.assertEqual(row['target'][0], reply)
            self.assertEqual(row['target'][-1], end - 1)
            self.assertLessEqual(len(row['prefix']), 128)
            self.assertGreaterEqual(row['prefix'][0], start)
        self.assertEqual(len(set(conversations)), 3)
        self.assertEqual(selected, examples(tokens, spans, 3))

    def test_loss_is_stable_for_large_logits(self):
        self.assertAlmostEqual(token_loss(np.array([10000., 10000.]), 0), np.log(2), places=10)


if __name__ == '__main__':
    unittest.main()

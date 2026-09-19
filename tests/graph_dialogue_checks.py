"""Verify reply windows expose prior context, never the predicted token."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/graph-language'))
from dialogue_data import windows


class DialogueWindowsChecks(unittest.TestCase):
    def test_short_prompt_is_left_padded_and_target_excluded(self):
        tokens = np.array([999, 2, 75, 4, 3, 100, 101, 4])
        spans = np.array([[1, 5, 8]])
        x, y, mask = windows(tokens, spans, [0], [5])
        np.testing.assert_array_equal(x[0, -4:], [2, 75, 4, 3])
        self.assertTrue(np.all(x[0, :-4] == 0))
        self.assertEqual(y[0, -1], 100)
        self.assertEqual(mask.sum(), 1)
        self.assertNotIn(999, x[0])
        self.assertNotIn(100, x[0])

    def test_reply_loss_includes_end_and_excludes_user_tokens(self):
        tokens = np.array([2, 75, 4, 3, 100, 101, 4])
        x, y, mask = windows(tokens, np.array([[0, 4, 7]]), [0], [6])
        np.testing.assert_array_equal(y[mask.astype(bool)], [100, 101, 4])
        np.testing.assert_array_equal(x[0, -6:], tokens[:6])

    def test_long_context_uses_exact_last_128_tokens(self):
        tokens = np.arange(300) + 10
        x, y, mask = windows(tokens, np.array([[10, 100, 280]]), [0], [270])
        np.testing.assert_array_equal(x[0], tokens[142:270])
        np.testing.assert_array_equal(y[0], tokens[143:271])
        self.assertTrue(np.all(mask == 1))


if __name__ == '__main__':
    unittest.main()

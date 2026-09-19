"""Conversation-bounded assistant windows with the runtime's left padding."""
import numpy as np


def windows(tokens, spans, choices, targets):
    x = np.zeros((len(choices), 128), np.int32)
    y = np.zeros_like(x)
    mask = np.zeros(x.shape, np.float32)
    for row, (choice, target) in enumerate(zip(choices, targets)):
        conversation_start, reply_start, reply_end = map(int, spans[choice])
        target = int(target)
        if not reply_start <= target < reply_end:
            raise ValueError('Target must be inside the selected assistant reply')
        start = max(conversation_start, target - 128)
        count = target - start
        if count < 1:
            raise ValueError('Prediction needs a preceding token')
        x[row, -count:] = tokens[start:target]
        y[row, -count:] = tokens[start + 1:target + 1]
        positions = np.arange(start + 1, target + 1)
        mask[row, -count:] = (positions >= reply_start) & (positions < reply_end)
    return x, y, mask


def sample(tokens, spans, rng, size):
    choices = rng.integers(0, len(spans), size)
    targets = np.array([rng.integers(spans[i, 1], spans[i, 2]) for i in choices])
    return windows(tokens, spans, choices, targets)

"""Training-only targets from the pinned foundation; never imported by CPU inference."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def reference_targets(reference, tokens, valid, trim_padding=False):
    if trim_padding:
        features, logits = [], []
        for row, mask in enumerate(np.array(valid)):
            length = int(mask.sum())
            if length < 1 or not np.array_equal(mask, np.arange(len(mask)) >= len(mask) - length):
                raise ValueError('Reference targets require nonempty left-padded token windows')
            hidden = reference.model(tokens[row:row + 1, -length:])
            logits.append(reference.model.embed_tokens.as_linear(hidden)[:, -1])
            features.append(hidden[:, -1])
        return mx.stop_gradient(mx.concatenate(logits)), mx.stop_gradient(mx.concatenate(features))
    width = tokens.shape[1]
    mask = (mx.tril(mx.ones((width, width), dtype=mx.bool_))[None, None]
            & valid[:, None, None, :]) | mx.eye(width, dtype=mx.bool_)[None, None]
    state = reference.model.embed_tokens(tokens) * valid[..., None]
    for layer in reference.model.layers:
        state = layer(state, mask=mask)
    features = reference.model.norm(state[:, -1])
    logits = reference.model.embed_tokens.as_linear(features)
    return mx.stop_gradient(logits), mx.stop_gradient(features)


def loss_terms(student, reference, tokens, targets, valid, match_features=True, trim_reference_padding=False):
    if match_features:
        logits, states = student(tokens, valid, return_states=True)
    else:
        logits = student(tokens, valid)
    target_logits, target_features = reference_targets(reference, tokens, valid, trim_padding=trim_reference_padding)
    target_logp = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
    student_logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ce = nn.losses.cross_entropy(logits, targets, reduction='mean')
    kl = mx.mean(mx.sum(mx.exp(target_logp) * (target_logp - student_logp), axis=-1))
    state_mse = mx.array(0., dtype=mx.float32)
    if match_features:
        features = student.base.model.norm(states[:, student._outputs[-1]])
        if features.shape != target_features.shape:
            raise ValueError('Feature matching requires equal feature dimensions; use zero feature weight for a larger reference')
        state_mse = mx.mean((features - target_features) ** 2)
    return ce, kl, state_mse

"""Training-only targets from the pinned foundation; never imported by CPU inference."""
import mlx.core as mx
import mlx.nn as nn


def reference_targets(reference, tokens, valid):
    width = tokens.shape[1]
    mask = (mx.tril(mx.ones((width, width), dtype=mx.bool_))[None, None]
            & valid[:, None, None, :]) | mx.eye(width, dtype=mx.bool_)[None, None]
    state = reference.model.embed_tokens(tokens) * valid[..., None]
    for layer in reference.model.layers:
        state = layer(state, mask=mask)
    features = reference.model.norm(state[:, -1])
    logits = reference.model.embed_tokens.as_linear(features)
    return mx.stop_gradient(logits), mx.stop_gradient(features)


def loss_terms(student, reference, tokens, targets, valid):
    logits, states = student(tokens, valid, return_states=True)
    target_logits, target_features = reference_targets(reference, tokens, valid)
    features = student.base.model.norm(states[:, student._outputs[-1]])
    target_logp = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
    student_logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ce = nn.losses.cross_entropy(logits, targets, reduction='mean')
    kl = mx.mean(mx.sum(mx.exp(target_logp) * (target_logp - student_logp), axis=-1))
    state_mse = mx.mean((features - target_features) ** 2)
    return ce, kl, state_mse

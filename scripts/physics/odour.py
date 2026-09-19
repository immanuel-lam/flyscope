"""Defined concentration field sampled at simulated antenna positions.

This is a scalar test field, not a turbulent fluid or receptor model.
"""
import numpy as np


def sample(antennae, source, spread=5.0):
    positions = np.asarray(antennae, dtype=float)
    source = np.asarray(source, dtype=float)
    if positions.shape != (2, 3) or source.shape != (2,) or spread <= 0:
        raise ValueError('Expected two 3D antenna positions, a 2D source and positive spread')
    if not np.isfinite(positions).all() or not np.isfinite(source).all():
        raise ValueError('Odour coordinates must be finite')
    distance_squared = ((positions[:, :2] - source) ** 2).sum(axis=1)
    return np.exp(-distance_squared / (2 * spread ** 2))


def steering(concentration):
    values = np.asarray(concentration, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected two finite nonnegative concentrations')
    # Negative turn drives toward the anatomical left in the hybrid controller.
    return float(np.clip(-8 * (values[0] - values[1]) / (values.sum() + .002), -.6, .6))

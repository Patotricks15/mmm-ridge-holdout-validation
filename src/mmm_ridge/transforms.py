"""Carry-over and diminishing-returns transforms for weekly channel spend."""

import numpy as np


def geometric_adstock(x, decay):
    """Return a_t = x_t + decay * a_{t-1} with a_{-1} = 0."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1)")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    carry = 0.0
    for t, value in enumerate(x):
        carry = value + decay * carry
        out[t] = carry
    return out


def hill_saturation(a, alpha, gamma):
    """Return s(a) = a^alpha / (a^alpha + gamma^alpha); s(gamma) = 0.5."""
    if alpha <= 0 or gamma <= 0:
        raise ValueError("alpha and gamma must be positive")
    a = np.asarray(a, dtype=float)
    if np.any(a < 0):
        raise ValueError("adstock values must be non-negative")
    a_alpha = np.power(a, alpha)
    return a_alpha / (a_alpha + gamma**alpha)


def normalise_by_train_max(a, train_idx):
    """Scale by the maximum over training rows only, so test rows never inform the scale."""
    a = np.asarray(a, dtype=float)
    scale = a[np.asarray(train_idx)].max()
    if scale <= 0:
        raise ValueError("training adstock must have a positive maximum")
    return a / scale

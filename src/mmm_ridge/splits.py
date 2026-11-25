"""Time-ordered holdout and rolling-origin splits. No shuffling."""

import numpy as np


def holdout_split(n_periods, holdout_fraction=0.15):
    """Return (train_idx, test_idx): the last `holdout_fraction` of periods form the test set."""
    if n_periods < 2:
        raise ValueError("need at least two periods")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")
    n_test = max(1, round(n_periods * holdout_fraction))
    cut = n_periods - n_test
    return np.arange(cut), np.arange(cut, n_periods)


def rolling_origin_splits(n_periods, n_folds=3, horizon=13):
    """Expanding-window folds. Fold k trains on [0, s + k*h) and tests on the next h periods."""
    start = n_periods - n_folds * horizon
    if start < 2:
        raise ValueError("not enough periods for the requested folds and horizon")
    folds = []
    for k in range(n_folds):
        cut = start + k * horizon
        folds.append((np.arange(cut), np.arange(cut, cut + horizon)))
    return folds

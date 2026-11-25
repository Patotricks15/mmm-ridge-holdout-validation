"""Holdout error metrics and the naive baseline."""

import numpy as np


def mape(actual, predicted):
    """Mean absolute percentage error, in percent."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def naive_last_value_forecast(y, train_idx, test_idx):
    """Forecast every test period with the last observed training value (no test data used)."""
    y = np.asarray(y, dtype=float)
    return np.full(len(test_idx), y[np.asarray(train_idx)][-1])

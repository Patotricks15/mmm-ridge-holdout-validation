"""Closed-form ridge regression with an unpenalised intercept and standardised predictors."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeFit:
    coef: np.ndarray
    intercept: float
    kappa: float


def fit_ridge(X, y, kappa):
    """Minimise ||y - Xb||^2 + kappa * ||b_std||^2, with predictors standardised on the fit rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    if np.any(x_std == 0):
        raise ValueError("a predictor is constant on the fit rows")
    Z = (X - x_mean) / x_std
    y_mean = y.mean()
    beta_std = np.linalg.solve(Z.T @ Z + kappa * np.eye(Z.shape[1]), Z.T @ (y - y_mean))
    coef = beta_std / x_std
    return RidgeFit(coef=coef, intercept=float(y_mean - x_mean @ coef), kappa=float(kappa))


def predict_ridge(fit, X):
    return fit.intercept + np.asarray(X, dtype=float) @ fit.coef

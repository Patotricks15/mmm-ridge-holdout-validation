import numpy as np
from sklearn.linear_model import Ridge

from mmm_ridge.ridge import fit_ridge, predict_ridge


def test_ridge_matches_sklearn_on_standardised_predictors():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3)) * np.array([1.0, 5.0, 0.2])
    y = X @ np.array([0.5, -0.3, 2.0]) + rng.normal(scale=0.1, size=80)
    kappa = 2.5

    fit = fit_ridge(X, y, kappa)

    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    sk = Ridge(alpha=kappa, fit_intercept=True).fit(Z, y)
    np.testing.assert_allclose(fit.coef * X.std(axis=0), sk.coef_, rtol=1e-8)
    np.testing.assert_allclose(predict_ridge(fit, X), sk.predict(Z), rtol=1e-8)


def test_larger_penalty_shrinks_coefficients():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(scale=0.2, size=60)
    norms = [np.linalg.norm(fit_ridge(X, y, k).coef) for k in (0.01, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]

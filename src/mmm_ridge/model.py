"""Log-linear MMM fitted with ridge penalty, chosen by rolling-origin validation (MM02)."""

import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from mmm_ridge.metrics import mape, naive_last_value_forecast
from mmm_ridge.ridge import fit_ridge, predict_ridge
from mmm_ridge.splits import holdout_split, rolling_origin_splits
from mmm_ridge.transforms import geometric_adstock, hill_saturation, normalise_by_train_max

DEFAULT_GRID = {
    "kappa": (0.01, 0.1, 1.0, 10.0),
    "decay": (0.0, 0.3, 0.5, 0.7),
    "alpha": (0.5, 1.0, 2.0),
    "gamma": (0.5, 1.0),
}


def build_design(df, channels, controls, decay, alpha, gamma, fit_idx):
    """Saturated adstock per channel; the scale uses only `fit_idx` rows."""
    columns = {}
    for channel in channels:
        a = geometric_adstock(df[channel].to_numpy(), decay)
        a = normalise_by_train_max(a, fit_idx)
        columns[f"{channel}_sat"] = hill_saturation(a, alpha, gamma)
    for control in controls:
        columns[control] = df[control].to_numpy(dtype=float)
    return pd.DataFrame(columns, index=df.index)


def _score_config(df, target, channels, controls, config, fit_idx, eval_idx):
    """Fit on `fit_idx` and return the MAPE on `eval_idx` for one hyperparameter set."""
    X = build_design(df, channels, controls, config["decay"], config["alpha"], config["gamma"], fit_idx)
    y = np.log(df[target].to_numpy(dtype=float))
    fit = fit_ridge(X.iloc[fit_idx].to_numpy(), y[fit_idx], config["kappa"])
    pred = np.exp(predict_ridge(fit, X.iloc[eval_idx].to_numpy()))
    return mape(df[target].to_numpy()[eval_idx], pred)


def _configs(grid):
    keys = ("kappa", "decay", "alpha", "gamma")
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def select_config(df, target, channels, controls, grid, train_idx, n_folds, horizon):
    """Pick the hyperparameters with the lowest mean MAPE across rolling-origin folds of the training window."""
    folds = rolling_origin_splits(len(train_idx), n_folds=n_folds, horizon=horizon)
    best = None
    for config in _configs(grid):
        scores = []
        for fold_train, fold_test in folds:
            fit_idx = train_idx[fold_train]
            eval_idx = train_idx[fold_test]
            scores.append(_score_config(df, target, channels, controls, config, fit_idx, eval_idx))
        mean_score = float(np.mean(scores))
        if best is None or mean_score < best["rolling_mape"]:
            best = {**config, "rolling_mape": mean_score, "fold_mapes": scores}
    return best


def fit_full(df, target, channels, controls, config, train_idx):
    """Refit on the full training window and return the design, the fit, and channel log-contributions."""
    X = build_design(df, channels, controls, config["decay"], config["alpha"], config["gamma"], train_idx)
    y = np.log(df[target].to_numpy(dtype=float))
    fit = fit_ridge(X.iloc[train_idx].to_numpy(), y[train_idx], config["kappa"])
    coefs = dict(zip(X.columns, fit.coef))
    contributions = pd.DataFrame(
        {ch: X[f"{ch}_sat"].to_numpy() * coefs[f"{ch}_sat"] for ch in channels},
        index=X.index,
    )
    return X, fit, contributions


def contribution_stability(df, target, channels, controls, base_config, kappas, train_idx):
    """Channel ranking by mean absolute log-contribution, for each kappa, with other hyperparameters fixed."""
    rows = {}
    for kappa in kappas:
        config = {**base_config, "kappa": kappa}
        _, _, contrib = fit_full(df, target, channels, controls, config, train_idx)
        rows[kappa] = contrib.iloc[train_idx].abs().mean()
    table = pd.DataFrame(rows)
    table.columns = [f"kappa={k:g}" for k in kappas]
    ranks = table.rank(ascending=False).astype(int)
    stable = bool((ranks.nunique(axis=1) == 1).all())
    return table, ranks, stable


def variance_inflation(X, channels):
    cols = [f"{c}_sat" for c in channels]
    values = X[cols].to_numpy(dtype=float)
    return pd.Series(
        [variance_inflation_factor(values, i) for i in range(len(cols))],
        index=channels, name="VIF",
    )


def run_mm02(df, target, channels, controls=(), grid=DEFAULT_GRID,
             holdout_fraction=0.15, n_folds=3, horizon=13):
    """Full MM02 pipeline: holdout split, rolling-origin tuning on training data, refit, holdout MAPE, stability."""
    n = len(df)
    train_idx, test_idx = holdout_split(n, holdout_fraction)
    best = select_config(df, target, channels, controls, grid, train_idx, n_folds, horizon)
    config = {k: best[k] for k in ("kappa", "decay", "alpha", "gamma")}

    X, fit, contributions = fit_full(df, target, channels, controls, config, train_idx)
    pred = np.exp(predict_ridge(fit, X.iloc[test_idx].to_numpy()))
    actual = df[target].to_numpy()[test_idx]
    naive = naive_last_value_forecast(df[target].to_numpy(), train_idx, test_idx)

    table, ranks, stable = contribution_stability(
        df, target, channels, controls, config, grid["kappa"], train_idx
    )

    return {
        "config": config,
        "rolling_mape": best["rolling_mape"],
        "fold_mapes": best["fold_mapes"],
        "fit": fit,
        "design": X,
        "contributions": contributions,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "holdout_mape": mape(actual, pred),
        "naive_mape": mape(actual, naive),
        "test_prediction": pred,
        "stability_table": table,
        "stability_ranks": ranks,
        "ranking_stable": stable,
        "vif": variance_inflation(X.iloc[train_idx], channels),
    }

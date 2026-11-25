# mmm-ridge-holdout-validation (MM02)

**Question:** Are the marketing channel contributions stable under ridge regularisation and a time holdout?

## Portfolio

| ID | Repository | Focus |
|---|---|---|
| MM01 | [mmm-adstock-saturation](https://github.com/Patotricks15/mmm-adstock-saturation) | Log-linear model with adstock and Hill saturation |
| **MM02** | **mmm-ridge-holdout-validation** (this repo) | Ridge penalties and rolling-origin validation |
| MM03 | [mmm-budget-reallocation](https://github.com/Patotricks15/mmm-budget-reallocation) | Constrained budget optimisation |

## Status

**No dataset is verified yet.** Phase 0 (verify one real weekly dataset with channel spend) is open. See [data/README.md](data/README.md). No results are reported until a dataset is attached. The notebook has not been executed because the data is not on this machine.

## Data

- Required: weekly sales (or conversions) and spend by channel, at least 100 periods, real observations (not synthetic), and a license that allows derived notebooks and figures.
- Access: read in place inside a Kaggle Notebook at `/kaggle/input/<slug>/`. Nothing is downloaded or committed.
- Holdout: the last 15% of periods, time-ordered. Dates are filled in once a dataset is verified.

## Method

- Features: adstock and Hill saturation as in MM01 (`src/mmm_ridge/transforms.py`), with the scale taken from training rows only.
- Model: `log y` on saturated adstock, fitted by closed-form ridge (`src/mmm_ridge/ridge.py`). Predictors are standardised on the fit rows. The intercept is not penalised. Objective: `‖y − Xβ‖² + κ‖β_std‖²`.
- Grid: `κ` in {0.01, 0.1, 1, 10}, `λ` (adstock decay), `α`, `γ`.
- Selection: rolling-origin validation (3 expanding folds of 13 periods) inside the training window only. The holdout is not used for selection.
- Holdout: refit on the full training window, then score the holdout once against the last-value naive baseline.
- Stability: channel ranking by mean absolute log-contribution for each `κ`, with the other settings fixed. The notebook reports whether the ranking is identical across penalties.
- Collinearity: VIF on the saturated-adstock columns.

## Decision supported

Use a channel ranking only if it holds across the penalty grid and the holdout MAPE beats the naive baseline. Otherwise report the channels as not separately identified, and use ridge coefficients only as a regularised summary.

## Limitation

Ridge shrinks correlated channels toward each other, so a channel can look important only because it is shrunk less. Stability across penalties does not show that the channel effects are causal.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests use tiny hand-built fixtures inside the test files. They check that the closed-form ridge matches `sklearn.linear_model.Ridge` on standardised predictors, that larger penalties shrink coefficients, that rolling-origin folds only train on past periods, that the holdout comes after every validation fold, and the adstock decay.

## Notebook

`notebooks/mm02_ridge_holdout.py` (jupytext percent format) runs top to bottom once `DATA_SLUG` is set and the dataset is attached on Kaggle. Figures are written to `reports/figures/`. They were not executed locally because the data is not on this machine.

# %% [markdown]
# # MM02: Ridge penalties and rolling-origin validation
#
# Question: are the channel contributions stable under regularisation and a time holdout?
#
# Run on Kaggle: attach the verified dataset, set `DATA_SLUG`, then Run All.
# Channel contributions are associations, not causal effects, unless an experiment supports them.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm_ridge.data import load_weekly
from mmm_ridge.model import run_mm02

np.random.seed(42)

# Set to the verified Kaggle slug recorded in data/README.md. Left unset until Phase 0 is done.
DATA_SLUG = None
DATA_FILE = "weekly.csv"
DATE_COL = "date"
TARGET_COL = "sales"
CHANNEL_COLS = []
CONTROL_COLS = []
DATASET_NAME = "verified weekly dataset (TBD)"
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

if DATA_SLUG is None:
    raise RuntimeError("Set DATA_SLUG to the verified Kaggle slug (see data/README.md).")

DATA_PATH = Path("/kaggle/input") / DATA_SLUG / DATA_FILE
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Attach the dataset to the notebook.")

# %% [markdown]
# ## 1. Load and validate

# %%
df = load_weekly(DATA_PATH, DATE_COL, TARGET_COL, CHANNEL_COLS)
print(f"{len(df)} weekly periods, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

# %% [markdown]
# ## 2. Tune on rolling origins inside the training window, then score the holdout once

# %%
result = run_mm02(df, TARGET_COL, CHANNEL_COLS, controls=CONTROL_COLS)
test_idx = result["test_idx"]
print("Selected configuration:", result["config"])
print(f"Rolling-origin MAPE (mean over folds): {result['rolling_mape']:.2f}%")
print("Fold MAPEs:", [round(m, 2) for m in result["fold_mapes"]])
print(f"Holdout: {df[DATE_COL].iloc[test_idx[0]].date()} to {df[DATE_COL].iloc[test_idx[-1]].date()}"
      f" ({len(test_idx)} periods)")
print(f"Holdout MAPE: {result['holdout_mape']:.2f}%  |  naive (last training value): {result['naive_mape']:.2f}%")

# %% [markdown]
# ## 3. Contribution stability across penalties

# %%
display(result["stability_table"].round(3))
display(result["stability_ranks"])
print("Channel ranking identical across kappa:", result["ranking_stable"])
if not result["ranking_stable"]:
    print("Ranking changes with kappa. Report the instability rather than a single ranking.")

# %% [markdown]
# ## 4. Collinearity

# %%
display(result["vif"].round(2))

# %% [markdown]
# ## 5. Figures

# %%
fig, ax = plt.subplots(figsize=(8, 4))
result["stability_table"].T.plot(ax=ax, marker="o")
ax.set_xlabel("Ridge penalty kappa")
ax.set_ylabel("Mean |log contribution|")
ax.set_title(f"MM02 contribution stability across penalties: {DATASET_NAME}")
fig.savefig(FIG_DIR / "mm02_stability.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))
test_dates = df[DATE_COL].iloc[test_idx]
ax.plot(test_dates, df[TARGET_COL].iloc[test_idx], label="actual")
ax.plot(test_dates, result["test_prediction"], label="ridge model")
ax.set_title(f"MM02 holdout fit: {DATASET_NAME} (MAPE {result['holdout_mape']:.2f}%)")
ax.legend()
fig.savefig(FIG_DIR / "mm02_holdout.png", dpi=150, bbox_inches="tight")
plt.show()

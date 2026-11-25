"""Load and validate a weekly marketing dataset."""

import pandas as pd

MIN_PERIODS = 100


def load_weekly(path, date_col, target_col, channel_cols):
    """Read a CSV, sort by date, and check the rules in the plan's data requirements."""
    df = pd.read_csv(path, parse_dates=[date_col])
    missing = [c for c in [date_col, target_col, *channel_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.sort_values(date_col).reset_index(drop=True)
    if len(df) < MIN_PERIODS:
        raise ValueError(f"need at least {MIN_PERIODS} periods, got {len(df)}")
    if (df[target_col] <= 0).any():
        raise ValueError("target must be strictly positive")
    if (df[list(channel_cols)] < 0).any().any():
        raise ValueError("channel spend must be non-negative")
    return df

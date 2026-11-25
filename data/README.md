# Data

Status: **no dataset verified yet.** Phase 0 of the MMM plan (verify one real weekly dataset with channel spend) is open. This repo uses the same dataset as MM01 once that is verified.

## Search log

- 2026-11-25: queried the public Kaggle dataset listing (`api/v1/datasets/list`, metadata only, nothing downloaded) for "marketing mix" and "marketing-mix".
- Candidate `veer06b/marrket-mix-dataset`: rejected. License listed as "Unknown", no description, file size 2,793 bytes (far below the 100-period minimum).
- No candidate passed all five rules in the plan.

## Intended source (once verified)

| Field | Value |
|---|---|
| Kaggle slug | TBD (not yet verified) |
| License | TBD |
| Read date | 2026-11-25 (search only) |
| Frequency | weekly, with spend by channel |

## Access method

Data is read in place inside a Kaggle Notebook at `/kaggle/input/<slug>/`. It is not downloaded to a local machine and never committed. `DATA_SLUG` in `notebooks/` must be set to the verified slug.

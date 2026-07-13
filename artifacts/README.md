# Artifacts (prospective structure)

This directory is scaffolding for a future convention, not a implemented system. No script currently writes here.

## Intended purpose

- `diagnostics/` — saved diagnostic outputs (e.g. calibration tables, residual checks) for a specific run, when worth keeping beyond the terminal.
- `experiments/` — one subdirectory per experiment run, named by run id.
- `splits/` — saved train/test (or other) split definitions, if a split needs to be pinned and reused across multiple experiment runs beyond what `data/processed/game_split.csv` already covers.

## Future structured run-artifact schema

The intended shape for a single experiment run, once this is actually
implemented:

```text
artifacts/experiments/<run-id>/
├── config.json        # feature set, model hyperparameters, split reference
├── metrics.json        # accuracy, log loss, calibration, etc.
├── data_summary.json    # row counts, class balance, date range
├── run_info.json       # timestamp, script, git commit, seed(s)
└── predictions.parquet  # per-row predicted probabilities, for re-analysis
```

The goal: these structured files become the **authoritative numerical source**. Any future CSV results ledger (e.g. an all-experiments summary table) should be _generated from_ these files, not maintained by hand alongside them — avoiding the drift risk of two independently-updated sources of the same numbers.

This is documentation of intent only. No artifact framework, run-id convention, or writer code has been implemented yet — see `notes/current-state.md` and `notes/decisions.md` for what's actually in place today.

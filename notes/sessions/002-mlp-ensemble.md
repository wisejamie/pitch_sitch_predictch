# Session 002 — MLP probability ensemble

**Status:** completed session narrative. Nothing here is an accepted
decision unless it is also in `notes/decisions.md`. Numbers are from
`artifacts/experiments/20260713-mlp-seed-ensemble-v1/` (see run_info.json
for exact git commit and package versions), not re-derived from memory.

Session 1 left the MLP's adoption undecided: it beat the logistic
`stand x strikes` baseline on every one of 10 seeds, but individual
seeds varied, and whether it ever predicted SL/OTHER had not been
checked. This session tested whether averaging the 10 seeds'
predicted probabilities resolves both concerns, using the
already-selected architecture frozen rather than re-tuned, and
produced the project's first structured experiment-artifact run.

## The story

The ensemble (arithmetic mean of the 10 seeds' probability vectors,
not a label vote) was evaluated against the logistic baseline, each
individual seed, and a single fixed reference seed (seed 0, chosen
before looking at any result). It beat every individual seed on log
loss and Brier score, and beat the mean and 9 of 10 seeds on accuracy
-- but not the single luckiest seed. This is the expected signature of
averaging under a proper scoring rule: log loss/Brier improve reliably
from ensembling; top-one accuracy doesn't necessarily beat the best
individual draw.

Checking each seed's predictions individually resolved the open SL/OTHER
question from session 1: every individual MLP seed does occasionally
predict SL (0.10%-1.10% of development pitches, depending on seed) --
something the logistic model never does. Averaging washes this back
down to near-zero for the ensemble (0.06%), closer to but not identical
to the logistic model's behavior. No model -- no seed, no ensemble, no
logistic -- ever predicts OTHER.

The result is not a uniform win. Per-class calibration and
discrimination broke down into a mixed picture: the ensemble has the
best calibration for FS, SL, and OTHER, but the logistic model still
has better FF calibration; the ensemble has the best ROC-AUC for FF,
FS, and SL, but the logistic model still has better OTHER ROC-AUC. A
game-level bootstrap (n=1000) confirmed the log-loss gain over both the
logistic baseline and the reference seed is highly stable across
resamples of the development games (95% CI never crosses zero in
either comparison); the accuracy gain is similarly robust vs. the
logistic baseline but slightly less certain vs. the single reference
seed (95.4% of resamples favor the ensemble, not 100%).

The experiment was run twice: once on a dirty working tree while the
implementation was still uncommitted (kept as
`artifacts/experiments/20260713-mlp-seed-ensemble-v1-validation/`,
not committed to git), and once from the clean implementation commit
(`artifacts/experiments/20260713-mlp-seed-ensemble-v1/`, committed).
The two runs' `metrics.json` and `predictions.parquet` were verified
identical before anything here was written.

## Results

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
| Logistic baseline (`stand x strikes`) | 0.5688 | 0.9061 | 0.5365 |
| MLP individual seeds -- mean +/- std | 0.5819 +/- 0.0043 | 0.8930 +/- 0.0030 | -- |
| MLP individual seeds -- range | [0.5772, 0.5908] | [0.8860, 0.8964] | -- |
| MLP reference seed (seed 0, fixed) | 0.5772 | 0.8964 | 0.5282 |
| **10-seed probability ensemble** | **0.5853** | **0.8830** | **0.5189** |

Per-class calibration (mean |gap| across 5 bins, lower better):

| class | logistic | reference seed | ensemble |
|---|---|---|---|
| FF | **0.0097** | 0.0293 | 0.0204 |
| FS | 0.0168 | 0.0189 | **0.0114** |
| SL | 0.0160 | 0.0139 | **0.0075** |
| OTHER | 0.0045 | 0.0053 | **0.0034** |

Per-class ROC-AUC:

| class | logistic | reference seed | ensemble |
|---|---|---|---|
| FF | 0.6426 | 0.6592 | **0.6769** |
| FS | 0.7030 | 0.7155 | **0.7257** |
| SL | 0.7744 | 0.7758 | **0.7816** |
| OTHER | **0.6967** | 0.6755 | 0.6905 |

Full metrics, confidence-vs-coverage tables, and per-pitch predictions:
`artifacts/experiments/20260713-mlp-seed-ensemble-v1/`.

## Confirmed results

- The 10-seed probability ensemble beats the logistic baseline on
  accuracy, log loss, and Brier score, with the log-loss gain confirmed
  stable under game-level bootstrap resampling of the development set
  (95% CI [-0.0296, -0.0165], never crosses zero).
- Every individual MLP seed occasionally predicts SL as its top class;
  the logistic model never does. No model tested predicts OTHER.
- The ensemble's SL prediction rate (0.06%) is lower than every
  individual seed's (0.10%-1.10%) -- averaging suppresses each seed's
  idiosyncratic swings toward the minority class.
- Per-class calibration and discrimination are not uniformly better for
  the ensemble: the logistic model has better FF calibration and better
  OTHER ROC-AUC than the ensemble.

## Methodological decisions

See `decisions.md` for the full entry. Pointer: the 10-seed probability
ensemble is adopted as the current leading predictive model; the
logistic `stand x strikes` model is retained as the interpretable
baseline, not replaced.

## Interpretations and hypotheses (not confirmed)

- The ensemble's accuracy/log-loss gain looks like genuine probability
  averaging benefiting from proper-scoring-rule behavior, not just
  variance reduction masking a worse model -- but why FF calibration
  and OTHER discrimination specifically resist this while FS/SL/OTHER
  calibration and FF/FS/SL discrimination all improve has not been
  investigated.

## Negative / mixed results

- FF calibration: logistic (0.0097) beats the ensemble (0.0204).
- OTHER ROC-AUC: logistic (0.6967) beats the ensemble (0.6905).
- Ensemble accuracy does not beat the single luckiest individual seed
  (seed 1, 0.5908 vs. ensemble's 0.5853) -- only the mean and 9 of 10
  seeds.

## Unresolved limitations and open questions

- Still evaluated entirely on the same 33 development games -- see the
  standing note in session 001 and `current-state.md`.
- Why FF and OTHER specifically resist the ensembling benefit that
  helps every other class/metric is unexplained.
- The reference seed (seed 0) is one fixed draw; its individual numbers
  are a reproducibility anchor, not a claim about "the" MLP's typical
  behavior.
- This session's artifact is the first use of the structured
  experiment-artifact convention. It was committed to git as a
  one-time research record (see `notes/decisions.md`), a deliberate,
  explicit exception to the general "don't commit generated artifacts"
  guidance, not a change to that guidance itself.

## Where we are now

Working models: the 10-seed MLP probability ensemble (accuracy 0.5853,
log loss 0.8830) is the current leading predictive model; the logistic
`stand x strikes` model (accuracy 0.5688, log loss 0.9061) is retained
as the interpretable baseline. Neither model predicts OTHER; the
ensemble very rarely predicts SL, the logistic model never does.

## Future directions

Unchanged from session 001's list -- see `notes/sessions/001-gausman-first-models.md`.

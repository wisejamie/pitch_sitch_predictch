# Decisions

This file contains accepted methodological decisions.

Ideas, hypotheses, and tentative plans belong in the research log until
they are explicitly accepted.

---

## 2026-07-12 — Data source

Pitch-level data will be sourced from Baseball Savant (Statcast) via the
`pybaseball` package.

**Why:** it is the only public source that provides pitch type, location,
and sequencing together for the Statcast era, which fully covers Kevin
Gausman's career. See `notes/research-log.md` for the fuller comparison
against Retrosheet, the MLB Stats API, and third-party mirrors.

**Not yet decided:** target definition, feature set, season range for
modelling, train/test split strategy, model family.

---

## 2026-07-12 — Gausman pitch-type target: collapse to FF / FS / SL / OTHER

For Gausman's Blue Jays-era data (2022-01-01 to 2026-07-12), the pitch-type
target collapses any `pitch_type` outside {FF, FS, SL} into a single
`OTHER` class (SI, CH, ST). Rows with no resolved `pitch_type`, `plate_x`,
`plate_z`, or `zone` (112 of 14,885 pitches, ~0.75%) are dropped rather
than imputed. Implemented in `pitch_sitch/labels.py`.

**Why:** FF/FS/SL account for ~98% of pitches across the period; the
remaining types are individually rare and vary in prevalence by season.
The 112 dropped rows include pitch-clock violations with no pitch thrown
(no data to label) and a small number of pitches Statcast's classifier
failed to label, concentrated in 10 games; both are treated the same way
for simplicity given the negligible row count.

**Known cost of this decision:** SI (sinker) was not uniformly rare — it
was ~6% of 2024 pitches (181/2,920) before nearly disappearing in 2025.
Collapsing it into OTHER obscures that this was a real, temporary
repertoire addition, not noise. If repertoire change over time becomes a
focus of analysis, this collapse may need revisiting.

**Not yet decided:** whether this collapse should also apply if the
project extends beyond Gausman or beyond this date range; how OTHER
should be handled in the location target (still undecided).

---

## 2026-07-12 — Train/test split: random game-level

The first experiment will use a random game-level split: all pitches
from a given game stay on one side of the split, and games are assigned
to train/test at random (not chronologically, not by season).

**Why:** the feature set under discussion includes recent-pitch-history
features (`prev_1..3_pitch_type`, `_plate_x`, `_plate_z`, ...). A random
pitch-level split would let adjacent pitches from the same plate
appearance land on both sides of the split, letting the model partially
see the exact PA it's being evaluated on via neighboring rows. Game-level
splitting avoids this by keeping each PA, and its surrounding history,
entirely within one side.

**Explicitly not decided here:** whether a chronological split should
also be run later to test stability over time; season range for the
modelling dataset; how many games/pitches are needed (data-sufficiency
question, to be resolved once the feature set is fixed).

---

## 2026-07-12 — Two feature-set exclusions

While the overall feature set remains an open draft (see research log),
two specific items are settled:

- `batter_id` will not be used as a model input for now, but will be
  kept as an identifier column in the working data so it remains
  available for a lookup join if batter-specific features are added
  later.
- `starter_or_reliever` is dropped entirely rather than kept as a
  constant column, since Gausman is used exclusively as a starter in
  this data.

**Why:** avoids re-deriving batter linkage later while not pretending
either field is a meaningful feature for a single-pitcher model today.

---

## 2026-07-12 — Move from lookup tables to multinomial logistic regression

The count+game-situation conditional-frequency lookup table performed
*worse* than the no-feature marginal baseline (63% of combinations had
exactly one training example). The project moves to multinomial
(softmax) logistic regression as the next model class, since it shares
statistical strength across feature values instead of requiring exact
combination matches. Adds `scikit-learn` as a dependency.

Encoding choices for the count_game feature group
(`pitch_sitch/design_matrix.py`):
- count state (balls, strikes): one-hot over the 12 fixed combinations,
  not raw numeric, since the lookup table showed this relationship is
  clearly non-additive (e.g. the two-strike splitter shift).
- outs_when_up, inning_topbot: one-hot.
- score_diff: numeric, clipped to +/-6 to limit the leverage of rare
  blowout-game rows.
- inning, on_1b/2b/3b: numeric/binary, unchanged.

**Not yet decided:** whether inning should eventually be bucketed
instead of linear; regularization/hyperparameter tuning; how this
encoding approach extends to future feature groups (pitcher workload,
matchup, pitch history).

---

## 2026-07-12 — No class weighting; overall performance and calibration remain the objective

Despite the class-weight sweep showing weighting can raise SL recall
from 0% toward 58%, the project will not use class weighting and will
not optimize specifically for slider recall. The model stays unweighted
(`class_weight=None`).

**Why:** the sweep showed this trade-off is not free -- log loss
worsened monotonically with weighting (0.937 to 1.278), and calibration
broke down sharply above alpha=0.5 (confidence-vs-coverage gaps grew to
+0.9 at high thresholds, meaning the model's most "confident"
predictions became actively wrong under heavy weighting). Since
calibrated probability and confidence-vs-coverage behavior are
explicit goals of this project (see README), that cost was judged not
worth the recall gain.

**How to apply:** future model comparisons should be evaluated on
overall accuracy and log loss / calibration, not on per-class recall
for the minority classes. SL/OTHER's poor recall is an accepted,
understood limitation of the current approach, not a bug to chase.

---

## 2026-07-13 — Batter handedness (`stand`) joins the working baseline

The working baseline is now: richest sequencing feature set (+prev3_class)
+ batter handedness, unweighted logistic regression, same game-level
split. `pitch_sitch/design_matrix.build_batter_hand_onehot`.

**Why:** log loss improved from 0.9371 to 0.9098 (the largest single-
feature gain found so far), accuracy change (-0.007) was within noise,
and it was added to fix a specific, measured problem: pooled FS
calibration at 0-0/1-0 looked fine (+0.011/-0.036) only because
opposite-signed handedness-specific errors (-0.086/+0.100 at 0-0,
-0.136/+0.087 at 1-0) were canceling out. Adding `stand` cut those
gaps by 35-63%, though a residual gap remains (largest at 1-0/L,
-0.089) that handedness alone doesn't explain.

**Not yet decided:** what's causing the residual gap; workload features
(pitch count, times through order, matchup history) are being tested
next, one group at a time, against this baseline.

---

## 2026-07-13 — Workload features tested and not adopted; found a real count x handedness interaction

Tested four current-game workload features (`game_pitch_count`,
`times_through_order`, `inning_pitch_count`, `prior_pa_vs_batter`;
`pitch_sitch/workload_features.py`), each added in isolation to the
handedness baseline. None improved accuracy or log loss beyond noise
(log loss ranged 0.9088-0.9101 vs. baseline 0.9098; accuracy within
+/-0.4pt against a ~0.9pt noise floor). **None of these four features
are part of the working model.**

Follow-up residual check found the FS calibration gap by strikes level
flips sign between hands: at 0 strikes, left-handed batters are
underpredicted (-0.046) and right-handed overpredicted (+0.042); at 2
strikes this reverses (L +0.031, R -0.043). An additive model (count +
stand, no interaction term) cannot represent a sign flip like this --
it can only shift each hand's prediction by a constant offset across
all counts. This is a real, fairly large, well-sampled (438-595 pitches
per cell) signature of a genuine count x handedness interaction: the
two-strike shift toward FS is apparently proportionally larger for
right-handed batters than the current model captures.

The analogous check for count x previous-pitch-class found no
comparably strong or consistent pattern -- gaps were mostly under 0.03
with no consistent sign trend, aside from one small, unreliable cell
(n=17).

**Not yet decided:** whether/how to add a count x handedness
interaction term (e.g. separate count coefficients per hand). This is
the strongest lead so far for the next accuracy/log-loss improvement,
but it hasn't been implemented -- doing so would be the first
departure from a purely additive feature set.

---

## 2026-07-13 — `stand x strikes` interaction is the new linear baseline

The working baseline is now: richest sequencing + stand (additive) +
stand x strikes interaction (6 pure interaction columns, 56 features
total), unweighted logistic regression, same split.
`pitch_sitch/design_matrix.build_stand_strikes_interaction`.

**Why:** vs. the additive baseline, log loss improved 0.9098 -> 0.9061
and accuracy 0.5627 -> 0.5688. A game-level bootstrap (n=1000) showed
the log-loss gain is the most stable result found so far (95% CI
[-0.006, -0.001], never crosses zero). It fixed the specific
count-by-handedness sign-flip pattern at 0 and 2 strikes. A finer
`stand x full-count` variant (24 cells) got marginally better accuracy
but a less certain log-loss gain and several thinly-populated cells, so
the coarser 6-cell version was preferred.

**Known limitation, not fixed by this:** the strikes=1/right-handed FS
calibration gap (~+0.03-0.04) is untouched by this interaction and
remains an open, unexplained residual.

---

## 2026-07-13 — Adopt the 10-seed MLP probability ensemble as the leading model; retain logistic regression as the interpretable baseline

The working "leading predictive model" is now the arithmetic-mean
probability ensemble of the 10 already-trained MLP seeds (frozen
architecture: 2 hidden layers, 32/16 units, alpha=0.001, 19 epochs --
see `pitch_sitch/mlp.py SELECTED_CONFIG`). The logistic `stand x
strikes` model is **retained**, not replaced, as the interpretable
baseline for cases where that matters more than the ensemble's metric
edge.

**Why:** vs. the logistic baseline, the ensemble improves accuracy
(0.5688 -> 0.5853), log loss (0.9061 -> 0.8830), and Brier score
(0.5365 -> 0.5189). The log-loss gain is confirmed stable under
game-level bootstrap resampling of the development games (95% CI
[-0.0296, -0.0165], vs. both the logistic baseline and a fixed
reference seed). This also resolves the two things blocking adoption
at the end of session 1: individual-seed variance (the ensemble beats
the mean and 9 of 10 seeds, though not the single luckiest seed) and
the unverified SL/OTHER question (every individual seed does
occasionally predict SL, unlike the logistic model; the ensemble
suppresses this back to near-zero; nothing tested ever predicts
OTHER).

**Known cost of this decision:** the ensemble is not uniformly better.
The logistic model has better FF-class calibration (mean |gap| 0.0097
vs. the ensemble's 0.0204) and better OTHER-class ROC-AUC (0.6967 vs.
0.6905). Neither model predicts OTHER at all, and the ensemble only
very rarely predicts SL (0.06% of development pitches, vs. 0% for the
logistic model).

**How to apply:** use the ensemble as the default reported model going
forward; use the logistic model when interpretability of individual
feature effects is required, or as a sanity-check baseline. Continue
evaluating both, since neither dominates the other in every respect.

**Also decided:** the artifact directory this decision is based on
(`artifacts/experiments/20260713-mlp-seed-ensemble-v1/`) was committed
to git as a one-time research record, in a commit separate from the
implementation code -- a deliberate, explicit exception to the general
"don't commit generated artifacts" guidance for this specific
small (~2.8MB), documented experiment, not a change to that guidance.
An earlier dirty-worktree run of the same experiment was kept locally
as `...v1-validation` (verified to reproduce identical metrics and
predictions) but was not committed.

**Not yet decided:** why FF calibration and OTHER discrimination
specifically resist the ensembling benefit; the strikes=1/right-handed
residual gap from session 1.

**Provisional to the development set:** as with every decision in this
file so far, this is based entirely on the repeatedly-used 33-game
development set, not a held-out test set.

---

## 2026-07-14 — Broad bat-tracking feature bundle tested and not adopted; more targeted conditional analysis remains open

Tested adding 8 raw Statcast bat-tracking measurements at prev_1
(bat speed, swing length, attack angle/direction, swing path tilt, two
contact-point offset fields, miss distance;
`pitch_sitch.design_matrix.build_prev_bat_tracking_numeric`) to the
working 56-feature baseline. On the 131/33 split, logistic and a
single-reference-seed MLP both moved in the positive direction on
accuracy and log loss, but every delta stayed inside the noise floor
already established by earlier feature experiments. **Not part of the
working model.**

A parallel, separately-scoped side project (`feature/estimated-swing-
timing` branch, paused) explored whether a genuine pitch-level swing-
timing label could be constructed via weak supervision against
Baseball Savant's aggregated Swing Timing metric, since no pitch-level
timing label is publicly available. It found a stable late-swing (not
early-swing) signal under held-out-pitcher and 2025 temporal
validation, but blinded video review found ~30% of a "swing"-labeled
sample were actually bunts/check-swings, concentrated in the
estimator's most extreme buckets -- a real Statcast `description`-
field data-quality gap. This branch is paused, not abandoned.

**Why not adopted (bat-tracking bundle):** the fields are populated
only when the previous pitch was swung at (~30% of rows) and only from
2023 onward, with `miss_distance` populated only on whiffs (~7%) --
most rows fall back to the imputed mean, likely limiting the signal
available to a bundle-level test.

**Not yet decided / worth exploring later:**
- Whether a more targeted, conditional treatment of individual
  bat-tracking fields (rather than one bundle) performs differently.
- Whether resuming the estimated-swing-timing side project is
  worthwhile once a better swing/no-swing detection method addresses
  the bunt/check-swing contamination -- plausibly a doable, worthwhile
  deeper exploration, but deliberately not the current focus.

See `notes/sessions/003-richness-bat-tracking-and-timing-pause.md` for
the full session narrative, including the also-negative prev_2/prev_3
richness test from the same session.

# Current State

Replaced each session, not appended to. For history and reasoning, see `notes/research-log.md` (index) and `notes/sessions/`. For accepted decisions and why, see `notes/decisions.md`.

## Active research problem

Predicting Kevin Gausman's next pitch type from pre-pitch context (count, game situation, recent pitch sequence, batter handedness). Location/zone prediction, other pitchers, and multi-pitcher generalization are not yet started.

## Data and target

Statcast pitch log via `pybaseball`, Gausman's Blue Jays tenure (2022-01-01 to 2026-07-12): 14,885 pitches, 164 games, 3,794 plate appearances. Target collapses pitch type to {FF, FS, SL, OTHER}; 112 pitches (0.75%) with no resolved pitch type/location are dropped. Split is random, game-level (131 train / 33 development games).

## The 33-game split is development/validation data, not a final test set

It has been evaluated against repeatedly across this session while iterating on features and models. Nothing reported anywhere in this project is an out-of-sample generalization claim yet — that requires a fresh, never-touched test set that doesn't exist yet.

## Current interpretable baseline

Multinomial logistic regression, unweighted, 56 features: count, game situation, recent pitch sequence (previous 1-3 pitches), batter handedness, and a `stand x strikes` interaction. See session 001 for the full feature list.

**Accuracy 0.5688, log loss 0.9061** on the 33 development games.

## Current leading candidate (not adopted)

A small regularized MLP (2 hidden layers, 32/16 units) on the same 56 features beat the logistic baseline on both metrics across all 10 random-initialization seeds tested (accuracy 0.5819 +/- 0.0043, log loss 0.8930 +/- 0.0030). It also introduced new calibration problems the logistic model didn't have (worse FF-specific calibration, new overconfidence at the 0.80-confidence threshold). Whether it ever predicts SL/OTHER has not been checked. Adoption is undecided.

## Strongest confirmed findings

- A full-combination frequency lookup table breaks down past ~2 features (63% of cells had exactly 1 training example; log loss hit 4.18) — the reason the project moved to logistic regression.
- Accuracy and log loss can disagree about which model is better (batter handedness: accuracy dropped within noise, log loss improved substantially).
- Every logistic model tested (unweighted) never predicts SL or OTHER, despite carrying real discriminative signal for SL (ROC-AUC 0.65-0.77) — the signal never gets large enough to beat FF/FS under an unweighted objective.
- A real count x batter-handedness interaction exists (FS calibration gap flips sign between 0 and 2 strikes); adding it improved log loss more robustly (under game-level resampling of the development set) than any other single change tested.

## Current limitations

- `OTHER` is a heterogeneous bucket (sinker + changeup + sweeper); no model tested predicts it well.
- The strikes=1/right-handed FS calibration gap is unexplained.
- No untouched final test set exists yet.
- MLP adoption is an open decision (metric gain vs. new calibration issues vs. losing interpretability).

## Most relevant next directions

See session 001's "Future directions" for the full, unscoped list. The most immediately relevant, not yet decided:

- Deciding whether to adopt the MLP, and checking its SL/OTHER behavior.
- Investigating the unexplained strikes=1/right-handed residual gap.
- Establishing a genuine held-out test set before treating any result here as a generalization claim.

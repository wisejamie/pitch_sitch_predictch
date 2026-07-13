# Current State

Replaced each session, not appended to. For history and reasoning, see
`notes/research-log.md` (index) and `notes/sessions/`. For accepted
decisions and why, see `notes/decisions.md`.

## Active research problem

Predicting Kevin Gausman's next pitch type from pre-pitch context
(count, game situation, recent pitch sequence, batter handedness).
Location/zone prediction, other pitchers, and multi-pitcher
generalization are not yet started.

## Data and target

Statcast pitch log via `pybaseball`, Gausman's Blue Jays tenure
(2022-01-01 to 2026-07-12): 14,885 pitches, 164 games, 3,794 plate
appearances. Target collapses pitch type to {FF, FS, SL, OTHER}; 112
pitches (0.75%) with no resolved pitch type/location are dropped. Split
is random, game-level (131 train / 33 development games).

## The 33-game split is development/validation data, not a final test set

It has been evaluated against repeatedly across two sessions while
iterating on features and models. Nothing reported anywhere in this
project is an out-of-sample generalization claim yet.

## Current leading predictive model

A 10-seed probability ensemble of a small regularized MLP (2 hidden
layers, 32/16 units, frozen architecture -- see session 002), on the
same 56 features as the logistic baseline. Ensemble = arithmetic mean
of the 10 seeds' predicted probabilities.

**Accuracy 0.5853, log loss 0.8830, Brier score 0.5189** on the 33
development games. Beats every individual seed on log loss/Brier, and
the mean/9-of-10 seeds on accuracy (not the single best seed). Full
run artifact: `artifacts/experiments/20260713-mlp-seed-ensemble-v1/`.

## Current interpretable baseline (retained, not replaced)

Multinomial logistic regression, unweighted, same 56 features: count,
game situation, recent pitch sequence (previous 1-3 pitches), batter
handedness, and a `stand x strikes` interaction.

**Accuracy 0.5688, log loss 0.9061, Brier score 0.5365.**

## Strongest confirmed findings

- A full-combination frequency lookup table breaks down past ~2
  features (63% of cells had exactly 1 training example; log loss hit
  4.18) -- the reason the project moved to logistic regression.
- Accuracy and log loss can disagree about which model is better.
- Every individual MLP seed occasionally predicts SL as its top class
  (0.10%-1.10% of pitches); the logistic model never does. The ensemble
  suppresses this back down to 0.06%. **No model tested -- logistic,
  any MLP seed, or the ensemble -- ever predicts OTHER.**
- A real count x batter-handedness interaction exists in the logistic
  model (FS calibration gap flips sign between 0 and 2 strikes).
- The ensemble is not uniformly better than the logistic model: the
  logistic model has better FF calibration and better OTHER ROC-AUC.

## Current limitations

- `OTHER` is a heterogeneous bucket (sinker + changeup + sweeper); no
  model tested -- logistic or MLP -- predicts it well.
- FF calibration and OTHER discrimination remain better under the
  logistic model than the ensemble; unexplained why those two
  specifically resist the ensembling benefit seen everywhere else.
- The strikes=1/right-handed FS calibration gap (logistic model) is
  still unexplained.
- No untouched final test set exists yet.

## Most relevant next directions

See session 001's "Future directions" for the full, unscoped list. The
most immediately relevant, not yet decided:
- Investigating why FF calibration and OTHER discrimination favor the
  logistic model despite the ensemble winning everywhere else.
- Investigating the unexplained strikes=1/right-handed residual gap.
- Establishing a genuine held-out test set before treating any result
  here as a generalization claim.

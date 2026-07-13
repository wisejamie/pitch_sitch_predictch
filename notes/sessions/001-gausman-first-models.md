# Session 001 — Gausman: first models

**Status:** completed session narrative. Nothing here is an accepted decision unless it is also in `notes/decisions.md` — this file follows the same rule as the rest of the research log. Numbers are from saved script outputs, not re-derived from memory.

Session 1 is paused here. This is a narrative record of what was done and learned.

## The story

Session 1 started from one narrow question: what pitch-level data exists to represent a pre-pitch situation and observe the resulting pitch? That audit led to choosing Baseball Savant (via `pybaseball`) as the data source, and Kevin Gausman's Blue Jays tenure (2022-01-01 to 2026-07-12: 14,885 pitches, 164 games, 3,794 plate appearances) as the concrete case study. Two cleaning decisions followed directly from the audit: 112 pitches (0.75%) lacking a resolved pitch type or location were dropped, and pitch types outside {FF, FS, SL} were collapsed into an `OTHER` class — a decision later found to obscure a real, temporary 2024 sinker addition (~6% of that season) that the `OTHER` label hides.

A random **game-level** train/test split (131/33 games) was chosen over a random pitch-level split specifically to avoid a mechanical leakage risk: with recent-pitch-history features in the plan, a pitch-level split would let a model partially "see" a plate appearance through its neighboring rows.

The first modeling attempt was a simple frequency lookup table, following the project's "start simple" principle. It worked cleanly for count alone (12 well-populated cells, a real if modest lift over the no-feature baseline) but collapsed once extended to the full game-situation feature group: with ~11,700 training pitches spread across ~5,700 realized feature combinations, 63% of cells had exactly one training example, and log loss exploded to 4.18 — worse than doing nothing. This was the first concrete argument for moving to a parametric model that shares statistical strength across situations instead of requiring exact combination matches.

Multinomial logistic regression fixed the sparsity problem immediately and became the base model for the rest of the session. Adding recent-pitch-history features (previous pitch class/result/location, extending to the previous three pitches) produced a small, mostly plateauing improvement, concentrated almost entirely in the first history feature (previous pitch class); depth beyond that added little.

A closer look at this logistic model's predictions turned up its most important limitation: it never predicted SL or OTHER at all, on any development-set pitch. Follow-up diagnostics showed this wasn't an absence of signal — P(SL) reliably separated true sliders from other pitches (ROC-AUC 0.65-0.77 across the logistic models, well above chance) — but the achieved separation was too small in magnitude to ever beat FF/FS under an unweighted decision rule, given how imbalanced the classes are. A controlled class-weight sweep confirmed this could be "fixed" mechanically (SL recall reached 58% at full balancing) but only by trading away log loss and, more importantly, calibration — at high weighting, the model's confident predictions became actively wrong (though the most extreme of these gaps were measured on very small high-confidence buckets, as small as 4 pitches — see Negative Results). Since calibrated probability was set as the explicit priority over minority-class recall, class weighting was tested and rejected.

A pitch-by-pitch walkthrough of one representative development-set game surfaced a second, more useful lead: the model appeared to underpredict the splitter early in counts. A full development-set diagnostic sharpened this into something more precise — the pooled undercount was actually two much larger, opposite-signed errors by batter handedness canceling out (lefties get far more splitters at 0-0/1-0 than the model, lacking any handedness feature, could represent). Adding batter handedness produced the largest single-feature log-loss improvement of the session, despite a small, noise-level drop in raw accuracy — a clean illustration of accuracy and log loss disagreeing about which model is
better.

Four current-game workload features (pitch count, times through the order, in-inning pitch count, prior same-game matchup count) were tested next and, individually, none improved on the handedness baseline beyond noise — a clean negative result. A follow-up residual check found something more specific than "workload matters": the handedness effect on splitter usage isn't just a constant offset per hand, it interacts with the count — the calibration gap's sign flips between 0 and 2 strikes. Adding a `stand x strikes` interaction term captured this and produced the most stable improvement of the session under game-level bootstrap resampling of the development games (its 95% CI on the log-loss gain never crossed zero); a finer `stand x full-count` version tested slightly better on accuracy but with more thinly-populated cells and a less certain log-loss gain, so the coarser interaction became the new baseline. This bootstrap speaks to how stable the gain is across which development games happen to be resampled — it is not an out-of-sample generalization check, since no data outside the existing 131/33 game split was involved.

Finally, a small regularized MLP (two hidden layers, selected via a validation split carved out of the training games only, never the development set) beat the logistic interaction model on both accuracy and log loss, consistently across ten random initializations. Diagnostics suggest this is mostly genuine improvement in probability quality (wider, similarly- or better-calibrated high-confidence coverage, better per-class discrimination for FF/FS) rather than simply sharper wrong guesses — but it introduced new, localized problems of its own (worse FF-specific calibration, new overconfidence at the 0.80 threshold, on a moderate n=116 pitches) that the logistic model didn't have. Whether the MLP also never predicts SL/OTHER was not checked in the saved output — that specific claim, confirmed for every logistic model tested, is unverified for the MLP. Whether to adopt the MLP is an
open decision.

## Cumulative results

All numbers are from the saved script outputs (`scripts/run_baseline.py`, `run_logistic_baseline.py`, `run_sequence_features.py`, `run_batter_hand_experiment.py`, `run_interaction_experiment.py`, `run_mlp_experiment.py`), evaluated on the same 33 development games throughout.

| Model                                                | Description                                                                           | Accuracy          | Log loss          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------- | ----------------- |
| Global marginal                                      | No features; always predicts train's most common class (FF)                           | 0.5207            | 1.0080            |
| Count-only lookup                                    | Empirical frequency table conditioned on (balls, strikes)                             | 0.5562            | 0.9559            |
| Count + game-situation lookup                        | Same lookup approach extended to 9 features — **failed**                              | 0.5045            | 4.1806            |
| Count + game-situation logistic regression           | First parametric model                                                                | 0.5630            | 0.9515            |
| + Recent pitch history (richest sequencing)          | + previous 1-3 pitch classes, previous-pitch result & location                        | 0.5694            | 0.9371            |
| + Batter handedness                                  | + one-hot batter stand                                                                | 0.5627            | 0.9098            |
| **+ stand x strikes interaction (current baseline)** | + 6 pure interaction columns                                                          | 0.5688            | 0.9061            |
| (tested, not adopted) stand x full-count interaction | 24-cell interaction; slightly better accuracy, less certain log-loss gain, thin cells | 0.5707            | 0.9072            |
| Small MLP (32,16), same features as current baseline | Regularized 2-hidden-layer MLP; mean +/- std across 10 seeds                          | 0.5819 +/- 0.0043 | 0.8930 +/- 0.0030 |

Not shown as separate rows (see below for why): the intermediate sequencing steps (+prev1_class 0.5662/0.9375, +prev1_result 0.5669/0.9384, +prev2_class 0.5675/0.9373 — all between the logistic and richest-sequencing rows above), the 5-point class-weight sweep (alpha 0.0 to 1.0, rejected), and the 4 individually-tested workload features (all within noise of the handedness baseline).

## Current full feature set (56 features — the `stand x strikes` baseline)

All one-hot blocks use fixed, hardcoded category lists (not just whatever appears in a given split), so train/test always produce identically-shaped matrices. "NONE" categories mark missing history at a plate appearance's start (or immediately after an unresolved pitch).

**Count / plate-appearance state**

- count one-hot (12): balls x strikes, e.g. `count_0-0` ... `count_3-2`

**Game situation**

- outs one-hot (3): `outs_0`, `outs_1`, `outs_2`
- inning half one-hot (2): `half_Top`, `half_Bot`
- inning (numeric, scaled)
- score differential (numeric, scaled; pitcher's-team perspective, clipped to +/-6 to limit the leverage of rare blowout games)
- runners on base (3 binary flags): `on_1b`, `on_2b`, `on_3b`

**Recent pitch sequence**

- previous pitch class, one-hot (5): FF / FS / SL / OTHER / NONE
- previous pitch result, one-hot (7): ball / called_strike / swinging_strike / foul / in_play / other / NONE
- previous pitch location (4): `prev_1_plate_x`, `prev_1_plate_z` (numeric, scaled, mean-imputed at missing history) plus a has-value flag for each
- pitch class two pitches back, one-hot (5)
- pitch class three pitches back, one-hot (5)

**Batter / matchup**

- batter handedness, one-hot (2): `stand_L`, `stand_R`
- batter handedness x strikes interaction, one-hot (6):
  `standXstrikes_L_0` ... `standXstrikes_R_2`

**Explicitly not included:** batter identity (`batter_id` is kept in the working data for a possible future lookup join, but is not itself a feature — see `decisions.md`), pitcher identity/handedness and starter-or-reliever (all constant for this single-pitcher case study), and all four workload features tested and rejected (game pitch count, times through the order, in-inning pitch count, prior same-game matchup count).

## Confirmed results

- Statcast via `pybaseball` provides complete pitch-level data for Gausman's Blue Jays tenure; 112 of 14,885 pitches (0.75%) lack a resolved pitch type or location.
- Collapsing pitch types outside {FF, FS, SL} into `OTHER` leaves it at ~1.8% of pitches overall, but it was ~6% in 2024 alone (a temporary sinker addition) — a real information cost of that decision.
- A full-combination frequency lookup table breaks down once conditioned on more than ~2 features: 63% of realized feature combinations had exactly one training example, and log loss exploded to 4.18.
- Accuracy and log loss can disagree about which model is better: adding batter handedness lowered accuracy slightly (within noise) while substantially improving log loss, by fixing a calibration problem argmax-based accuracy can't see.
- **Every logistic regression model tested with the unweighted objective never predicted SL or OTHER at all**, on any development-set pitch — a robust, repeated finding across that model family specifically. Whether this also holds for the MLP was not checked in the saved output — see Unresolved limitations.
- The logistic models carry real discriminative signal for SL independent of argmax (ROC-AUC 0.65-0.77, always above chance) — the signal exists but never reaches the magnitude needed to outrank FF/FS.
- A specific count x handedness interaction is real and well-sampled: pooled FS calibration gaps flip sign between 0 strikes and 2 strikes, a pattern an additive model can't represent; adding the interaction term closed most of this gap.
- None of four tested workload features improved accuracy or log loss beyond the noise floor.
- A small regularized MLP beat the best logistic model on both accuracy and log loss across all 10 random-initialization seeds tested.

## Methodological decisions

Full detail and rationale for each of these is in `decisions.md`; this
is only a pointer list:

1. Data source: Baseball Savant via `pybaseball`.
2. Target: collapse to FF/FS/SL/OTHER; drop the 112 unresolved rows.
3. Split: random **game-level**, not pitch-level or chronological.
4. `batter_id` kept for lookup but not used as a feature; `starter_or_reliever` dropped (constant for this pitcher).
5. Move from lookup tables to multinomial logistic regression (adds `scikit-learn` as a dependency).
6. No class weighting; overall accuracy/log loss/calibration remain the objective, not minority-class recall.
7. Batter handedness (`stand`) joined the working baseline.
8. `stand x strikes` interaction is the current working baseline.

## Interpretations and hypotheses (not confirmed)

- SL/OTHER's blindness in the logistic models looks like a class-imbalance/objective-function issue rather than a missing-feature issue — supported by richer sequencing features not moving recall at all, but not proven, since not every possible feature has been tried.
- The MLP's gain looks more like genuine improvement in probability estimation than sharper-but-wrong decisions, based on widened and similarly-or-better-calibrated confidence coverage at most thresholds — but this is qualified, not settled, given its new FF-calibration and 0.80-threshold problems.
- The residual strikes=1/right-handed FS calibration gap remains unexplained; no hypothesis has been ventured for it yet.

## Negative results

- Count + game-situation lookup table: log loss 4.18, worse than the no-feature baseline.
- Class-weight sweep: raised SL recall to 58% at full balancing, but log loss degraded monotonically (0.94 to 1.28) and calibration broke down at high thresholds (gaps up to +0.93) — rejected given the project's calibration-first objective. Note the largest of these gaps were measured on very small high-confidence buckets (as few as 4 pitches at the most extreme threshold under full balancing), so their exact magnitude is imprecise even though the qualitative pattern (calibration degrading with weight) held consistently across thresholds.
- Four workload features (game pitch count, times through the order, in-inning pitch count, prior same-game matchup count), tested ndividually: none beat the handedness baseline beyond noise.
- `stand x full-count` (24-cell) interaction: a small accuracy edge over the 6-cell version, but a less statistically certain log-loss gain and several very thin cells (n as low as 11) — not adopted.

## Unresolved limitations and open questions

- **No untouched final test set exists.** Everything above comes from repeated use of the same 33 development games — see the note below.
- `OTHER` remains a heterogeneous bucket (sinker + changeup + sweeper) that nothing tested predicts well; unclear whether this is fixable within the current target definition.
- The strikes=1/right-handed FS gap, and the MLP's FF-calibration and 0.80-threshold issues, remain unexplained.
- **Whether the MLP ever predicts SL or OTHER has not been checked.** The "never predicts SL/OTHER" finding is confirmed only for the logistic model family; the MLP experiment script did not report predicted-class frequencies.
- Whether to adopt the MLP over the logistic interaction model is undecided — the trade-off between a uniform accuracy/log-loss gain and new localized calibration problems (plus losing linear interpretability) hasn't been resolved.
- Season range, location-target definition, and multi-pitcher generalization remain entirely unexplored, as originally scoped out at the start of the session.

## The 33 development games are not an untouched final test set

Every result in this session — including the ones labeled "confirmed" — was produced by repeatedly evaluating on the same 33 development games while iterating on features and models. That is development/validation use, not a final generalization check. Game-level bootstrap results in this session describe stability across resamples of these same 33 games, not confirmation of out-of-sample generalization. Before any claim here is treated as a statement about true out-of-sample performance, it should be re-checked against a fresh, never-touched
test set.

## Where we are now

The working model is multinomial logistic regression on the richest sequencing feature set, plus batter handedness and a `stand x strikes` interaction (accuracy 0.5688, log loss 0.9061 on the development set). A small MLP beats it on both metrics but introduces new calibration quirks and hasn't been adopted. SL and OTHER remain unpredicted by every unweighted logistic model tried; that's an accepted, understood limitation given the project's calibration-first objective, not an open bug. Whether the same is true of the MLP is unverified.

## Future directions (not decisions — preserved as ideas only)

These have not been accepted, scoped, or committed to. They're recorded so they aren't lost, not because they've been chosen as next steps:

1. Ensemble the MLPs (the 10 seeds already trained behave differently enough that averaging might help).
2. Add richer batter information beyond handedness (identity-based history, swing/take tendencies).
3. Better represent previous-pitch outcomes and sequences (deeper history, alternative encodings beyond one-hot class/result).
4. Train a general model across many pitchers and fine-tune to a specific pitcher.
5. Learn pitcher and batter embeddings rather than hand-designing categorical features.
6. Eventually model entire plate appearances as sequences, using a larger multi-pitcher dataset.

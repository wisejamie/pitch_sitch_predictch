# Session 003 — Sequencing richness, a swing-timing side branch (paused), and raw bat-tracking features

**Status:** completed session narrative. Nothing here is an accepted
decision unless it is also in `notes/decisions.md`. Numbers are
re-derived by re-running the relevant scripts this session, not
recalled from memory.

Session 002 left the MLP ensemble adopted as the leading model, with
"richer batter information" and "better representing previous pitches"
named as open tinkering directions. This session worked through both,
plus a substantial side excursion into estimating a pitch-level swing
timing label, which was explored, validated, and then explicitly
paused in favor of a more direct approach.

## The story

**1. Prev_2/prev_3 richness.** Session 001 found "depth beyond prev_1
adds little," but that comparison was asymmetric: prev_1 carried class
+ result + location, while prev_2/prev_3 only ever carried class. This
session gave prev_2 and prev_3 the same richness and re-tested against
the 56-feature baseline, using the historical 131/33 split and a
lightweight logistic + single-reference-seed-MLP comparison. Result:
clean negative. No candidate (prev_2 alone, prev_3 alone, or both)
beat baseline consistently across both models and both metrics; all
deltas were within the noise floor already established by the earlier
workload-feature experiment. Not adopted.

**2. Estimated swing-timing side branch (`feature/estimated-swing-timing`,
paused).** A separate line of work investigated whether batter
swing-timing information could be added as a feature. Baseball
Savant's official "Swing Timing" leaderboard turned out to be
aggregated-only (pitcher-season / batter-season / pitcher-season-by-
pitch-type) -- no pitch-level timing label is publicly downloadable.
The branch built a weak-supervision estimate instead: a shared,
cross-pitcher pitch-level model fit to match Savant's aggregate rates,
validated via held-out-pitcher splits, a constant-baseline comparison,
formal leave-one-feature-out ablation, and frozen-model temporal
validation on 2025 data. A blinded visual review against real MLB Film
Room clips followed, with the answer key kept out of git until review
completion. The review surfaced a real Statcast data-quality gap:
`description` doesn't reliably separate bunts/check-swings from full
swings, and this contamination concentrated in the model's most
extreme score buckets. Given that open problem and the cost of
resolving it, the engineer chose to pause this branch entirely rather
than continue -- see "Future directions" below for revisiting it. All
code is committed on `feature/estimated-swing-timing`; no
review/answer-key data was ever committed, by design.

**3. Raw bat-tracking features on `main`.** Instead of continuing the
timing-label branch, the session pivoted to testing Statcast's raw
bat-tracking measurements directly as prev_1 history features, on the
existing baseline pipeline -- no timing interpretation, no weak
supervision, just the raw fields. `add_history()` gained a
`bat_tracking_depths` parameter (`pitch_sitch/sequence_features.py`)
and `design_matrix.build_prev_bat_tracking_numeric()` applies the same
has-value-flag + train-mean-imputation treatment already used for
prev_k location. Eight fields were added at prev_1: `bat_speed`,
`swing_length`, `attack_angle`, `attack_direction`, `swing_path_tilt`,
the two contact-point offset fields, and `miss_distance`. Result:
another clean negative, same shape as the prev_2/3 test -- all four
deltas (logistic/MLP-reference-seed x accuracy/log_loss) moved in the
positive direction but stayed inside the established noise floor.
These fields are populated only when the *previous* pitch was swung at
(~30% of rows, 0% before 2023), and `miss_distance` only on whiffs
(~7%), so most rows fall back to the imputed mean -- the likely reason
for the weak effect. Not adopted as a broad bundle.

## Results

Prev_2/prev_3 richness (`scripts/run_prev_depth_richness_experiment.py`,
131/33 split):

| candidate | n_feat | logistic acc | logistic ll | MLP(seed 0) acc | MLP(seed 0) ll |
|---|---|---|---|---|---|
| baseline | 56 | 0.5688 | 0.9061 | 0.5772 | 0.8964 |
| + prev2 result+location | 67 | 0.5685 (-0.0003) | 0.9067 (+0.0006) | 0.5814 (+0.0042) | 0.8950 (-0.0014) |
| + prev3 result+location | 67 | 0.5682 (-0.0006) | 0.9059 (-0.0002) | 0.5762 (-0.0010) | 0.9016 (+0.0052) |
| + prev2 & prev3 | 78 | 0.5701 (+0.0013) | 0.9063 (+0.0003) | 0.5717 (-0.0055) | 0.9040 (+0.0077) |

Raw bat-tracking bundle (`scripts/run_bat_tracking_experiment.py`,
same split):

| candidate | n_feat | logistic acc | logistic ll | MLP(seed 0) acc | MLP(seed 0) ll |
|---|---|---|---|---|---|
| baseline | 56 | 0.5688 | 0.9061 | 0.5772 | 0.8964 |
| + prev1 bat tracking | 72 | 0.5727 (+0.0039) | 0.9056 (-0.0005) | 0.5778 (+0.0006) | 0.8947 (-0.0017) |

Baseline row reproduces the recorded historical numbers exactly in
both tables, confirming the pipeline wiring.

## Confirmed results

- Giving prev_2/prev_3 the same feature richness as prev_1 (result +
  location, not just pitch class) does not improve on the baseline
  beyond noise, on either model.
- Adding a broad bundle of 8 raw prev_1 bat-tracking measurements does
  not improve on the baseline beyond noise, on either model.
- Baseball Savant's public Swing Timing metric is aggregated-only;
  confirmed by direct investigation of the leaderboard's endpoints and
  split options. No pitch-level timing label exists publicly.
- A weakly-supervised, cross-pitcher, pitch-level late-swing score can
  be fit and shows a stable signal for late swings (not early swings)
  under held-out-pitcher and 2025 temporal validation.
- Blinded human video review found ~30% of a "swing"-labeled sample
  were actually bunts or check-swings, disproportionately concentrated
  in the estimator's most extreme score buckets -- a real Statcast
  `description`-field data-quality gap, not a flaw specific to the
  estimator's modeling choices.

## Methodological decisions

None from this session require a `decisions.md` entry beyond the
tested-and-not-adopted bundle noted there (see pointer below). No
model, feature set, or split methodology changed from what session 002
established.

## Interpretations and hypotheses (not confirmed)

- The bat-tracking bundle's weak effect is most likely explained by
  its severe compounding sparsity (previous-pitch-was-a-swing x
  post-2023 x, for `miss_distance`, previous-pitch-was-a-whiff) rather
  than the fields being genuinely uninformative -- a more targeted
  test (e.g. conditioning on cases where the field is actually
  populated, or testing fields individually rather than as one bundle)
  might behave differently. Not tested this session.
- The bunt/check-swing contamination found in the blinded review may
  be fixable with a more targeted swing/no-swing definition (e.g.
  incorporating swing-length or bat-speed thresholds rather than
  `description` alone) if that side project is picked up again.

## Negative / mixed results

- Prev_2/prev_3 richness: no consistent win on any metric/model
  combination; not adopted.
- Raw bat-tracking bundle: small, noise-floor-sized gains only; not
  adopted.

## Unresolved limitations and open questions

- Still evaluated entirely on the same 33 development games -- see the
  standing note in session 001 and `current-state.md`.
- Whether a more targeted (non-bundle) treatment of bat-tracking
  fields would do better is untested.
- The estimated-swing-timing branch's data-quality gap (bunt/check-
  swing contamination) is unresolved; the branch is paused, not
  abandoned.

## Where we are now

Working models are unchanged from session 002: the 10-seed MLP
probability ensemble (accuracy 0.5853, log loss 0.8830) remains the
leading model; the logistic `stand x strikes` model (accuracy 0.5688,
log loss 0.9061) remains the interpretable baseline. Two candidate
feature additions (prev_2/3 richness, raw bat-tracking bundle) were
tested and not adopted this session.

## Future directions

Unchanged from session 001's list, plus:
- A more targeted (non-bundle, or conditional) treatment of raw
  bat-tracking features, if revisited.
- Resuming the estimated-swing-timing side project once a better
  swing/no-swing detection method addresses the bunt/check-swing
  contamination found in the blinded review -- plausibly worthwhile
  and doable, but deliberately not the current focus.

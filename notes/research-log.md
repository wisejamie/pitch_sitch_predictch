# Research Log

This file records the evolving reasoning behind Pitch Sitch Predictch.

It may contain hypotheses, candidate formulations, intuitions, possible experiments, and unresolved questions. Nothing in this file should be treated as an accepted methodological decision unless it is later recorded in `decisions.md`.

---

## 2026-07-12 — Initial project framing

### Broad idea

Pitch Sitch Predictch is an exploratory machine-learning project about modelling a pitcher’s next-pitch decision from the situation immediately before the pitch is thrown.

A pre-pitch situation could include information such as:

- count;
- inning and outs;
- baserunners;
- score and game state;
- pitcher and batter handedness;
- pitcher and batter identities;
- recent pitch sequence;
- previous pitch results;
- pitcher workload;
- repertoire and historical tendencies;
- prior interactions between the pitcher and batter.

The model would map this context to some representation of the next pitch.

At the highest level:

$[x_t = \text{information available before pitch } t]$

$[
y_t = \text{pitch thrown at time } t
]$

The project should be framed around the general problem of modelling a pitcher, while Kevin Gausman serves as the first concrete case study.

---

## Why this problem may be valuable

Pitchers do not choose pitches independently. Pitch selection may depend on the count, recent sequencing, batter characteristics, game situation, pitcher repertoire, and strategic tendencies.

A successful model could potentially provide value in several ways.

### Opponent preparation

A batter or analyst could estimate which pitches are most likely in a given situation.

The practical value may not come from predicting every pitch correctly. It may come from identifying a smaller set of situations in which a pitcher becomes highly predictable.

For example:

> When the model assigns at least 80% probability to one pitch type, how often is that pitch actually thrown?

A useful result may therefore involve both:

- predictive accuracy or probability quality;
- coverage, meaning how often the model reaches a useful confidence level.

### Pitcher self-scouting

A team could identify situations in which its own pitcher becomes unusually predictable.

Possible findings might include:

- repeated pitch sequences;
- strong count-based tendencies;
- changes with runners on base;
- predictable behaviour after a miss or swinging strike;
- differences against left- and right-handed batters;
- changes later in games.

Predictability would not automatically imply poor strategy. Some pitches may remain effective even when expected. The model would first describe behaviour, not determine whether the behaviour is optimal.

### Faster understanding of new pitchers

A broader model trained across many pitchers might eventually learn general pitch-selection structure and then adapt to an individual pitcher using limited pitcher-specific data.

This could be relevant for:

- young pitchers;
- recent call-ups;
- pitchers returning from injury;
- pitchers who changed their repertoire;
- newly acquired players;
- pitchers with limited major-league data.

This remains a longer-term direction rather than part of the initial experiment.

### Foundation for pitch strategy modelling

Predicting what a pitcher will throw is different from recommending what a pitcher should throw.

A future system might separate:

$[
P(\text{pitch choice} \mid \text{pre-pitch context})
]$

from:

$[
P(\text{outcome} \mid \text{context}, \text{pitch choice}, \text{location})
]$

The first model describes historical decision-making. The second would estimate the result of possible pitch choices.

Only after both are understood could the project begin to investigate counterfactual or strategic questions.

---

## Initial case study: Kevin Gausman

Kevin Gausman is the likely first pitcher to investigate.

Current intuition:

- he has a large recent pitch sample;
- his repertoire is concentrated around a small number of major pitches;
- his four-seam fastball and splitter create an intuitive pitch-selection problem;
- the concentrated repertoire creates a strong naive baseline;
- the model would therefore need to identify meaningful contextual signal rather than gain accuracy by handling many rare pitch classes.

Gausman should remain a case study rather than becoming embedded into the general project definition.

The first data audit should determine:

- available seasons;
- total pitch count;
- pitch-type frequencies;
- number of games and plate appearances;
- missing data;
- changes in repertoire or usage over time;
- whether the available sample supports the proposed modelling questions.

---

# Candidate machine-learning formulations

Several formulations are possible. These should be investigated rather than assumed.

---

## 1. Pitch-type classification

The simplest initial target is the next pitch type.

$[
P(Y_{\text{type}} = k \mid x_t)
]$

Possible target classes might include:

- _individual Statcast pitch labels;_ - I think this is the way to go.
- broader pitch families such as fastball, breaking ball, and off-speed.

### Questions to resolve

- Should classification reflect the official Statcast label or a broader functional category?
- How stable are pitch labels across seasons?

### Possible advantages

- clear supervised-learning task;
- interpretable baselines;
- easy to compare model families;
- probabilities can be directly inspected;
- suitable first step before location modelling.

### Possible limitations

- two pitches of the same type may have very different intended locations;
- pitch labels do not capture execution quality;
- rare pitches create imbalance;
- the observed label may not fully capture the strategic decision.

---

## 2. Discrete location prediction

The area around home plate could be divided into zones.

The model could predict:

$[
P(Y_{\text{zone}} = z \mid x_t)
]$

or:

$[
P(Y_{\text{zone}} = z \mid x_t, Y_{\text{type}})
]$

Possible location representations include:

- nine strike-zone cells;
- strike-zone cells plus chase regions;
- a larger fixed grid;
- broader regions such as high, low, inside, and outside.

### Questions to resolve

- How many zones preserve meaningful information without creating excessive sparsity?
- Should zone boundaries be fixed or normalized to the batter’s strike zone?
- Should pitch type be known when predicting location?
- Should pitch type and location be predicted jointly?
- Does actual plate location represent the pitcher’s decision or execution error?

### Important limitation

Observed location is not necessarily intended location.

The model may be learning a combination of:

- target choice;
- command;
- execution error;
- measurement noise.

The resulting claim may need to be:

> Predict where the pitch is likely to arrive.

rather than:

> Predict where the pitcher intended to throw it.

---

## 3. Continuous location prediction

Instead of location zones, the model could predict continuous plate coordinates:

$[
(\hat{x}, \hat{z})
]$

A simple model might minimize squared error:

$[
L_{\text{location}} =
(x-\hat{x})^2 + (z-\hat{z})^2
]$

However, a single point prediction may be inappropriate if several locations are plausible.

For example, the pitcher may commonly throw a fastball either high and inside or high and outside. Predicting the average location could place the estimate near the centre of the plate, even if that location is rarely intended or observed.

Possible alternatives include:

- predicting a two-dimensional probability distribution;
- mixture-density networks;
- discretized spatial probability maps;
- Gaussian or mixture-of-Gaussian outputs;
- quantile or uncertainty-aware regression.

### Questions to resolve

- Is point prediction useful?
- Should the output represent expected location or a probability surface?
- How should uncertainty be evaluated?
- Should the location model be conditional on predicted or actual pitch type?
- How much location error would be meaningful in baseball terms?

---

## 4. Joint pitch-type and location prediction

The full target could be represented as:

$[
P(Y_{\text{type}}, Y_{\text{location}} \mid x_t)
]$

One decomposition is:

$[
P(Y_{\text{type}} \mid x_t)
\cdot
P(Y_{\text{location}} \mid Y_{\text{type}}, x_t)
]$

This reflects the intuition that location strategy depends heavily on pitch type.

Possible modelling approaches include:

- two-stage models;
- one shared encoder with separate output heads;
- joint discrete classes such as `FASTBALL_HIGH`;
- hierarchical classifiers;
- conditional density models.

### Questions to resolve

- Would joint labels create too many low-frequency combinations?
- Should both outputs share the same contextual representation?
- How should errors in pitch-type prediction propagate into location prediction?
- Is it better to establish pitch-type performance first?

---

## 5. Pitch-characteristic prediction

A model might eventually predict:

- velocity;
- spin rate;
- horizontal movement;
- vertical movement;
- release point.

These may be easier to predict after conditioning on pitcher identity and pitch type, but they may be less directly connected to strategic pitch choice.
These feel like more like excuetion outcomes rather than decisions.

### Questions to resolve

- Are they useful for detecting fatigue, pitch-shape changes, or deception?
- Should they be part of the initial target or treated as later auxiliary tasks?
- Maybe fine for grouping (like by velocity and break) - but probably not worth exploring.

---

# Candidate feature representation

The model needs a representation of the pre-pitch situation.

The first major design question is whether to use manually constructed tabular features, learned sequence representations, or both.

---

## 1. Count and plate-appearance state

Candidate features:

- balls;
- strikes;
- pitch number in the plate appearance;
- number of foul balls;
- (some derived ones:)
  - whether the pitcher is ahead or behind;
  - whether the count contains two strikes;
  - whether the count contains three balls.

Questions:

- Should balls and strikes be treated as separate numerical variables or one categorical count state?
- Are derived features useful, or should the model learn them?
- How should unusual counts or automatic balls and strikes be handled?

---

## 2. Game state

Candidate features:

- inning;
- inning half;
- outs;
- score differential;
- runners on first, second, and third;
- number of runners on base;
- runner in scoring position;
- leverage-related variables.

Questions:

- Which of these contain actual predictive value?
- Should score and inning be represented continuously or categorically?
- Would an externally calculated leverage index introduce hidden assumptions?
- Are all game-state variables available before the pitch in the selected data source?

---

## 3. Pitcher context

Candidate features:

- pitcher identity;
- pitcher handedness;
- starter or reliever;
- historical pitch usage;
- typical repertoire;
- pitch count in the game;
- pitches thrown in the inning;
- batters faced;
- times through the order;
- rest since previous appearance;
- current-game pitch usage;
- current-game velocity trends.

Questions:

- Should historical tendencies be provided explicitly or learned from pitcher identity?
- How can historical statistics be calculated without using future data?
- Should current-game behaviour be allowed to update the model’s representation of the pitcher?
- How should repertoire changes across seasons be represented?

For a single-pitcher model, some stable pitcher features will be constant and therefore irrelevant. They become important if the project expands to multiple pitchers.

---

## 4. Batter and matchup context

Candidate features:

- batter identity;
- batter handedness;
- batting-order position;
- same-handed or opposite-handed matchup;
- batter pitch-type tendencies;
- batter swing and chase rates;
- batter hot/cold zones;
- prior plate appearances against the pitcher;
- pitches already seen from the pitcher that game;
- times through the matchup.

Questions:

- Does batter identity provide generalizable information or encourage memorization?
- Should batter statistics be career-level, season-level, or rolling?
- How much data is needed before batter-specific effects are reliable?
- Does including batter skill help predict the pitcher’s decision, or does it complicate the first question unnecessarily?

---

## 5. Recent pitch sequence

Candidate sequence information:

- previous pitch types;
- previous pitch locations;
- previous pitch velocities;
- previous pitch results;
- swing or take;
- contact, foul, or miss;
- whether the pitch was in the zone;
- number of consecutive repeated pitch types;
- pitch-type counts within the plate appearance;
- pitch sequence from earlier matchups in the game.

Possible representations:

- fixed features for the previous one, two, or three pitches;
- summary statistics over the current plate appearance;
- Markov transition features;
- recurrent neural networks;
- Transformers;
- learned pitch embeddings.

Questions:

- How much recent history is useful?
- Does sequence information beyond the current plate appearance matter?
- Should sequences reset at the start of each batter?
- How should missing history at the beginning of a plate appearance be represented?
- Is a sequence model justified by the available sample size?
- Does a simple previous-pitch transition table capture most of the available signal?

---

## 6. Categorical representations and embeddings

Categorical variables could be represented using:

- one-hot encoding;
- ordinal or integer encoding;
- frequency encoding;
- target-independent historical statistics;
- learned embeddings.

Potential embedding targets include:

- pitcher identity;
- batter identity;
- pitch type;
- count state;
- game situation;
- recent pitch sequence.

A possible neural representation is:

$[
h_t = f_\theta(x_t)
]$

where $(h_t)$ is a learned situation embedding.

This embedding could represent similarity between situations in a way that is useful for predicting the next pitch.

Potential future analysis:

- visualize learned situation embeddings;
- identify clusters of similar pitch situations;
- compare embeddings across pitchers;
- test whether situations cluster by count, handedness, or pitch sequence.

Questions:

- Are embeddings useful with only one pitcher?
- Would learned representations outperform carefully designed tabular features?
- Can the learned space be interpreted?
- Is “situation similarity space” an actual analytic goal or simply a useful mental model?

---

# Model-output options

The model’s output should match the intended baseball use.

---

## Hard classification

The model returns one predicted pitch:

$[
\hat{y} = \arg\max_k P(Y=k\mid x)
]$

This is easy to score with accuracy but discards uncertainty.

Example:

> Predicted pitch: splitter.

This may be insufficient for strategic interpretation.

---

## Probability distribution over pitch types

The model returns:

$[
P(Y=k\mid x)
]$

Example:

- four-seam fastball: 54%;
- splitter: 39%;
- slider: 7%.

This output may be more useful.

It supports:

- calibration analysis;
- high-confidence filtering;
- confidence-versus-coverage curves;
- expected-value calculations;
- comparing multiple plausible pitch options.

Current intuition is that probability quality may matter more than only top-one classification accuracy, but this remains to be tested.

---

## Joint probability over pitch and location

A model could return probabilities such as:

- fastball, upper zone: 31%;
- fastball, middle zone: 18%;
- splitter, below zone: 29%;
- splitter, in zone: 12%;
- other outcomes: 10%.

This may be closer to the information a batter would care about, but it creates a more difficult and sparse prediction task.

Questions:

- Is a joint output interpretable?
- Does it require too much data?
- Should pitch type and location be evaluated separately?
- How many combined classes would be viable?

---

## Conditional spatial distribution

The model could produce:

$[
P(\text{location} \mid \text{pitch type}, x)
]$

This could be shown as a heat map for each pitch type.

Example:

> If the next pitch is a splitter, most of the probability mass lies below the strike zone.

This may better reflect the batter’s actual decision problem than a single point location.

---

# Objective functions, evaluation metrics, and baseball utility

The phrase “value function” may refer to several different things and should be separated carefully.

---

## Training objective

The training loss determines what the model learns.

Possible losses include:

### Pitch classification

Cross-entropy:

\[
L\_{\text{type}}
=
-\sum_k y_k \log p_k
\]

Possible modifications:

- class weighting;
- focal loss;
- label smoothing;
- hierarchical loss for related pitch types.

Questions:

- Should rare pitches receive extra weight?
- Would class weighting improve minority recall while harming calibration?
- Is distinguishing every class equally valuable?

### Discrete location

Cross-entropy over zones.

Potential issue:

Predicting an adjacent zone is penalized the same as predicting the opposite side of the plate unless the loss reflects spatial distance.

Possible alternatives:

- distance-aware class loss;
- ordinal or structured classification;
- soft labels over neighbouring zones.

### Continuous location

Possible losses:

- mean squared error;
- mean absolute error;
- negative log-likelihood;
- mixture-density loss;
- probabilistic spatial scoring rules.

### Multi-task learning

A combined objective could be:

\[
L =
\lambda*{\text{type}} L*{\text{type}}

- \lambda*{\text{location}} L*{\text{location}}
  \]

Questions:

- How should the weights be selected?
- Does joint learning improve shared representations?
- Does one task dominate the other?

---

## Predictive evaluation

Possible pitch-type metrics:

- overall accuracy;
- balanced accuracy;
- per-class precision and recall;
- macro F1;
- top-two accuracy;
- log loss;
- Brier score;
- expected calibration error;
- confusion matrix.

Possible location metrics:

- zone accuracy;
- distance between predicted and observed location;
- negative log-likelihood;
- probability contained within a chosen radius;
- calibration of spatial confidence regions.

Important principle:

A model should be compared against strong and relevant baselines, not only the majority pitch.

Potential baselines include:

- always predict the pitcher’s most common pitch;
- sample from the pitcher’s overall usage distribution;
- most common pitch by count;
- most common pitch by count and batter handedness;
- previous-pitch transition frequencies;
- simple multinomial logistic regression;
- simple tree-based model.

---

## Confidence and coverage

A model may be most useful when it recognizes situations in which one pitch is unusually likely.

For a threshold (q), act only when:

$[
\max_k P(Y=k\mid x) \ge q
]$

Then evaluate:

### Precision at threshold

How often is the predicted pitch correct when the model exceeds the threshold?

### Coverage

What fraction of all pitches exceed the threshold?

This creates a confidence-versus-coverage curve.

Example result:

> The model was correct on 82% of pitches when its top probability exceeded 75%, and those situations represented 24% of all pitches.

This may be more meaningful than one overall accuracy value.

Questions:

- What confidence level would be actionable?
- What coverage is large enough to matter?
- Should thresholds vary by count or batter?
- Does the model remain calibrated at high confidence?

---

## Baseball utility or decision value

Predictive accuracy does not directly equal batter advantage.

A batter deciding to sit on a pitch faces asymmetric outcomes:

- benefit from guessing correctly;
- cost from guessing incorrectly;
- value of remaining neutral;
- dependence on count, pitch type, and batter skill.

A simplified expected-value model might be:

\[
EV(\text{sit on pitch})
=
pA - (1-p)B
\]

where:

- (p) is the probability the predicted pitch occurs;
- (A) is the benefit of correctly anticipating it;
- (B) is the cost of anticipating the wrong pitch.

The prediction becomes actionable when:

$[
p > \frac{B}{A+B}
]$

This framing suggests that the required confidence may be substantially above 50% when an incorrect guess is costly.

However, (A) and (B) should eventually be estimated from baseball outcomes rather than assumed.

Possible future utility measurements:

- expected change in run value;
- expected change in weighted on-base average;
- swing-decision value;
- batter performance when anticipating a pitch family;
- simulation of correct and incorrect pitch anticipation;
- comparison with natural pitch-recognition ability.

This is a later stage. The first model should avoid claiming batter advantage solely from prediction accuracy.

---

# Data sufficiency

The project should not assume that a fixed number of pitches is sufficient.

Possible approaches to evaluating data sufficiency:

## Learning curves

Train on increasing numbers of pitch events:

$[
N \in
{250, 500, 1000, 2000, 4000, \ldots}
]$

Evaluate each model on the same held-out set.

Questions:

- Does performance continue to improve?
- Does calibration improve?
- Do rare-pitch metrics stabilize?
- Does performance vary substantially across random seeds?
- Does older data help or hurt because the pitcher has changed?

## Per-class sample counts

A model may have many total pitches but few examples of a rare pitch type.

Relevant quantities include:

- total number of pitches;
- training examples per class;
- examples per count;
- examples per handedness matchup;
- examples in high-confidence subgroups;
- number of independent games and plate appearances.

## Effective rather than theoretical situation count

The space of exact pitch situations may be enormous.

This does not necessarily require observing every combination. Machine-learning models can share information across related situations.

The key empirical question is:

> Does model performance plateau with the available data under the selected representation and model class?

---

# Generalization and dataset splitting

The appropriate split depends on the research question.

Possible approaches include:

## Random individual-pitch split

Likely risk:

- pitches from the same game or plate appearance may appear in multiple sets;
- correlated pitch sequences may leak across training and evaluation.

This option may be inappropriate.

## Random game-level split

All pitches from one game remain together, while games are randomly assigned to train, validation, and test sets.

This evaluates:

> Can the model generalize to unseen games drawn from the same broad period?

## Season-balanced random game split

Games are randomized within season so each dataset split contains representation from each season.

This reduces accidental season imbalance.

## Chronological split

Train on earlier games and evaluate on later games.

This evaluates:

> Can historical behaviour predict a pitcher’s future behaviour?

It is sensitive to repertoire drift and strategic change.

## Leave-one-season-out evaluation

Train on some seasons and evaluate on a separate season.

This can reveal changes across years but may not be the primary evaluation.

No split strategy has yet been accepted.

The first investigation should clarify what claim each split supports and which contamination risks apply.

---

# Candidate model progression

The project should begin with the simplest models capable of answering the current question.

Possible progression:

1. empirical frequency baselines;
2. count-conditioned lookup models;
3. multinomial logistic regression;
4. tree-based models;
5. small multilayer perceptron;
6. sequence models;
7. multi-pitcher representation learning;
8. pitcher-specific fine-tuning.

This is not a fixed roadmap.

A more complex model should be introduced only when:

- the simpler model has established the baseline;
- the dataset supports the added complexity;
- the architecture tests a specific hypothesis;
- the expected gain is not merely cosmetic.

---

# Longer-term transfer-learning idea

A future extension could train a model across many pitchers and then adapt it to a specific pitcher.

The general model could learn:

- common count effects;
- handedness effects;
- typical sequencing patterns;
- relationships among pitch families;
- general responses to game state.

The pitcher-specific adaptation could then learn:

- individual repertoire;
- usage rates;
- personal sequencing tendencies;
- matchup-specific preferences.

Possible experiment:

- train a pitcher-specific model from scratch using (N) pitches;
- train a general model on other pitchers;
- evaluate the general model zero-shot;
- fine-tune the general model using the same (N) pitcher-specific pitches;
- compare performance as (N) increases.

A meaningful result could be:

> The adapted general model reaches a given level of performance using substantially fewer pitcher-specific examples than a model trained from scratch.

This remains a future direction and should not shape the first implementation prematurely.

---

# Initial intuitions

These are current intuitions rather than accepted conclusions.

- A probability distribution over pitch types may be more useful than only a hard predicted class.
- High-confidence accuracy and coverage may be more meaningful than overall accuracy alone.
- Recent pitch sequence may contain important predictive information.
- A concentrated repertoire creates a stronger baseline and therefore a more demanding initial case study.
- Actual pitch location may combine strategic intent and execution error.
- Simple tabular models may provide strong first baselines.
- The model should first describe pitcher behaviour before attempting to recommend strategy.
- The amount of required data should be determined empirically with learning curves.
- Gausman appears to be a promising first case but should be validated through a data audit.

---

# Questions currently open

## Data

- Which data source should be used?
- What pitch-level fields are available?
- Which fields are reliably known before the pitch?
- How complete and stable are Statcast pitch labels?
- How much usable Gausman data exists?
- How much has his repertoire changed across seasons?
- How should unusual or missing pitch events be handled?

## Target

- Predict pitch type only?
- Predict broad pitch families or exact pitch labels?
- Include an `OTHER` class?
- Predict discrete location zones?
- Predict continuous location?
- Predict a joint pitch-type/location distribution?
- Include velocity, movement, or spin as later auxiliary targets?

## Features

- Which game-state variables matter?
- How much previous pitch history should be included?
- Should batter identity be included?
- Should historical tendencies be explicit features?
- Should the representation be tabular, sequential, or both?
- Can a meaningful situation embedding be learned?

## Model output

- Hard class?
- Probability distribution?
- Top-two prediction?
- Spatial probability map?
- Joint distribution over pitch type and location?

## Evaluation

- What are the strongest honest baselines?
- Which metrics match the intended use?
- What confidence and coverage would be practically meaningful?
- How should class imbalance be handled?
- How should calibration be measured?
- What constitutes a successful first result?

## Splitting and generalization

- Random game-level split?
- Season-balanced game-level split?
- Chronological evaluation?
- Multiple split strategies for different claims?
- How many random seeds or repeated splits are needed?

## Baseball interpretation

- At what probability would a batter rationally sit on one pitch?
- How costly is an incorrect pitch guess?
- Does knowing pitch type without location provide enough value?
- Are some counts or pitch types more actionable than others?
- Can predictability be translated into run-value or batting-outcome changes?

---

# Immediate next investigation

The first investigation should remain narrow:

> What pitch-level data is available for representing a general pre-pitch situation and observing the resulting pitch?

The investigation should:

1. identify candidate data sources;
2. list available fields;
3. separate pre-pitch context from post-pitch outcomes;
4. inspect how pitches, games, and plate appearances are identified;
5. identify likely leakage risks;
6. determine the minimum code required for a first data audit;
7. leave target, feature, and model choices open unless the data clearly constrains them.

After that, the first Gausman-specific audit can investigate:

- pitch volume by season;
- pitch labels and usage;
- missingness;
- repertoire stability;
- class balance;
- number of games and plate appearances;
- candidate sample sizes for initial experiments.

---

# Working reminder

The purpose of the early repository is not to prove that the initial idea is correct.

It is to create a disciplined process for discovering:

- what the actual modelling problem should be;
- what the data can support;
- what performance would be meaningful;
- which parts of the problem are predictable;
- which claims remain unsupported.

---

# 2026-07-13 — Session 1 summary: from data audit to a working baseline

Session 1 is paused here. This section is a narrative record of what was
done and learned. It follows the same rule as the rest of this file:
nothing here is an accepted decision unless it is also in
`decisions.md`.

## The story

Session 1 started from one narrow question: what pitch-level data
exists to represent a pre-pitch situation and observe the resulting
pitch? That audit led to choosing Baseball Savant (via `pybaseball`) as
the data source, and Kevin Gausman's Blue Jays tenure (2022-01-01 to
2026-07-12: 14,885 pitches, 164 games, 3,794 plate appearances) as the
concrete case study. Two cleaning decisions followed directly from the
audit: 112 pitches (0.75%) lacking a resolved pitch type or location
were dropped, and pitch types outside {FF, FS, SL} were collapsed into
an `OTHER` class — a decision later found to obscure a real, temporary
2024 sinker addition (~6% of that season) that the `OTHER` label hides.

A random **game-level** train/test split (131/33 games) was chosen over
a random pitch-level split specifically to avoid a mechanical leakage
risk: with recent-pitch-history features in the plan, a pitch-level
split would let a model partially "see" a plate appearance through its
neighboring rows.

The first modeling attempt was a simple frequency lookup table,
following the project's "start simple" principle. It worked cleanly for
count alone (12 well-populated cells, a real if modest lift over the
no-feature baseline) but collapsed once extended to the full
game-situation feature group: with ~11,700 training pitches spread
across ~5,700 realized feature combinations, 63% of cells had exactly
one training example, and log loss exploded to 4.18 — worse than doing
nothing. This was the first concrete argument for moving to a
parametric model that shares statistical strength across situations
instead of requiring exact combination matches.

Multinomial logistic regression fixed the sparsity problem immediately
and became the base model for the rest of the session. Adding
recent-pitch-history features (previous pitch class/result/location,
extending to the previous three pitches) produced a small, mostly
plateauing improvement, concentrated almost entirely in the first
history feature (previous pitch class); depth beyond that added little.

A closer look at this model's predictions turned up its most important
limitation: it never predicted SL or OTHER at all, on any test pitch.
Follow-up diagnostics showed this wasn't an absence of signal — P(SL)
reliably separated true sliders from other pitches (ROC-AUC 0.65-0.77
across models, well above chance) — but the achieved separation was too
small in magnitude to ever beat FF/FS under an unweighted decision rule,
given how imbalanced the classes are. A controlled class-weight sweep
confirmed this could be "fixed" mechanically (SL recall reached 58% at
full balancing) but only by trading away log loss and, more importantly,
calibration — at high weighting, the model's confident predictions
became actively wrong. Since calibrated probability was set as the
explicit priority over minority-class recall, class weighting was
tested and rejected.

A pitch-by-pitch walkthrough of one representative held-out game
surfaced a second, more useful lead: the model appeared to underpredict
the splitter early in counts. A full development-set diagnostic
sharpened this into something more precise — the pooled undercount was
actually two much larger, opposite-signed errors by batter handedness
canceling out (lefties get far more splitters at 0-0/1-0 than the model,
lacking any handedness feature, could represent). Adding batter
handedness produced the largest single-feature log-loss improvement of
the session, despite a small, noise-level drop in raw accuracy — a clean
illustration of accuracy and log loss disagreeing about which model is
better.

Four current-game workload features (pitch count, times through the
order, in-inning pitch count, prior same-game matchup count) were tested
next and, individually, none improved on the handedness baseline beyond
noise — a clean negative result. A follow-up residual check found
something more specific than "workload matters": the handedness effect
on splitter usage isn't just a constant offset per hand, it interacts
with the count — the calibration gap's sign flips between 0 and 2
strikes. Adding a `stand x strikes` interaction term captured this and
produced the most statistically robust improvement of the session
(confirmed via game-level bootstrap resampling); a finer
`stand x full-count` version tested slightly better on accuracy but with
more thinly-populated cells and a less certain log-loss gain, so the
coarser interaction became the new baseline.

Finally, a small regularized MLP (two hidden layers, selected via a
validation split carved out of the training games only, never the
development set) beat the logistic interaction model on both accuracy
and log loss, consistently across ten random initializations.
Diagnostics suggest this is mostly genuine improvement in probability
quality (wider, similarly- or better-calibrated high-confidence
coverage, better per-class discrimination for FF/FS) rather than simply
sharper wrong guesses — but it introduced new, localized problems of its
own (worse FF-specific calibration, new overconfidence at the 0.80
threshold) that the logistic model didn't have. Whether to adopt it is
an open decision.

## Cumulative results

All numbers are from the saved script outputs (`scripts/run_baseline.py`,
`run_logistic_baseline.py`, `run_sequence_features.py`,
`run_batter_hand_experiment.py`, `run_interaction_experiment.py`,
`run_mlp_experiment.py`), evaluated on the same 33-game holdout throughout.

| Model | Description | Accuracy | Log loss |
|---|---|---|---|
| Global marginal | No features; always predicts train's most common class (FF) | 0.5207 | 1.0080 |
| Count-only lookup | Empirical frequency table conditioned on (balls, strikes) | 0.5562 | 0.9559 |
| Count + game-situation lookup | Same lookup approach extended to 9 features — **failed** | 0.5045 | 4.1806 |
| Count + game-situation logistic regression | First parametric model | 0.5630 | 0.9515 |
| + Recent pitch history (richest sequencing) | + previous 1-3 pitch classes, previous-pitch result & location | 0.5694 | 0.9371 |
| + Batter handedness | + one-hot batter stand | 0.5627 | 0.9098 |
| **+ stand x strikes interaction (current baseline)** | + 6 pure interaction columns | 0.5688 | 0.9061 |
| (tested, not adopted) stand x full-count interaction | 24-cell interaction; slightly better accuracy, less certain log-loss gain, thin cells | 0.5707 | 0.9072 |
| Small MLP (32,16), same features as current baseline | Regularized 2-hidden-layer MLP; mean +/- std across 10 seeds | 0.5819 +/- 0.0043 | 0.8930 +/- 0.0030 |

Not shown as separate rows (see below for why): the intermediate
sequencing steps (+prev1_class 0.5662/0.9375, +prev1_result 0.5669/0.9384,
+prev2_class 0.5675/0.9373 — all between the logistic and richest-sequencing
rows above), the 5-point class-weight sweep (alpha 0.0 to 1.0, rejected),
and the 4 individually-tested workload features (all within noise of the
handedness baseline).

## Current full feature set (56 features — the `stand x strikes` baseline)

All one-hot blocks use fixed, hardcoded category lists (not just
whatever appears in a given split), so train/test always produce
identically-shaped matrices. "NONE" categories mark missing history at
a plate appearance's start (or immediately after an unresolved pitch).

**Count / plate-appearance state**
- count one-hot (12): balls x strikes, e.g. `count_0-0` ... `count_3-2`

**Game situation**
- outs one-hot (3): `outs_0`, `outs_1`, `outs_2`
- inning half one-hot (2): `half_Top`, `half_Bot`
- inning (numeric, scaled)
- score differential (numeric, scaled; pitcher's-team perspective,
  clipped to +/-6 to limit the leverage of rare blowout games)
- runners on base (3 binary flags): `on_1b`, `on_2b`, `on_3b`

**Recent pitch sequence**
- previous pitch class, one-hot (5): FF / FS / SL / OTHER / NONE
- previous pitch result, one-hot (7): ball / called_strike /
  swinging_strike / foul / in_play / other / NONE
- previous pitch location (4): `prev_1_plate_x`, `prev_1_plate_z`
  (numeric, scaled, mean-imputed at missing history) plus a has-value
  flag for each
- pitch class two pitches back, one-hot (5)
- pitch class three pitches back, one-hot (5)

**Batter / matchup**
- batter handedness, one-hot (2): `stand_L`, `stand_R`
- batter handedness x strikes interaction, one-hot (6):
  `standXstrikes_L_0` ... `standXstrikes_R_2`

**Explicitly not included:** batter identity (`batter_id` is kept in
the working data for a possible future lookup join, but is not itself
a feature — see `decisions.md`), pitcher identity/handedness and
starter-or-reliever (all constant for this single-pitcher case study),
and all four workload features tested and rejected (game pitch count,
times through the order, in-inning pitch count, prior same-game
matchup count).

## Confirmed results

- Statcast via `pybaseball` provides complete pitch-level data for
  Gausman's Blue Jays tenure; 112 of 14,885 pitches (0.75%) lack a
  resolved pitch type or location.
- Collapsing pitch types outside {FF, FS, SL} into `OTHER` leaves it at
  ~1.8% of pitches overall, but it was ~6% in 2024 alone (a temporary
  sinker addition) — a real information cost of that decision.
- A full-combination frequency lookup table breaks down once conditioned
  on more than ~2 features: 63% of realized feature combinations had
  exactly one training example, and log loss exploded to 4.18.
- Accuracy and log loss can disagree about which model is better: adding
  batter handedness lowered accuracy slightly (within noise) while
  substantially improving log loss, by fixing a calibration problem
  argmax-based accuracy can't see.
- The model never predicted SL or OTHER at all, under any feature set or
  model tested with the unweighted objective — a robust, repeated
  finding.
- The model does carry real discriminative signal for SL independent of
  argmax (ROC-AUC 0.65-0.77, always above chance) — the signal exists
  but never reaches the magnitude needed to outrank FF/FS.
- A specific count x handedness interaction is real and well-sampled:
  pooled FS calibration gaps flip sign between 0 strikes and 2 strikes,
  a pattern an additive model can't represent; adding the interaction
  term closed most of this gap.
- None of four tested workload features improved accuracy or log loss
  beyond the noise floor.
- A small regularized MLP beat the best logistic model on both accuracy
  and log loss across all 10 random-initialization seeds tested.

## Methodological decisions

Full detail and rationale for each of these is in `decisions.md`; this
is only a pointer list:

1. Data source: Baseball Savant via `pybaseball`.
2. Target: collapse to FF/FS/SL/OTHER; drop the 112 unresolved rows.
3. Split: random **game-level**, not pitch-level or chronological.
4. `batter_id` kept for lookup but not used as a feature;
   `starter_or_reliever` dropped (constant for this pitcher).
5. Move from lookup tables to multinomial logistic regression
   (adds `scikit-learn` as a dependency).
6. No class weighting; overall accuracy/log loss/calibration remain the
   objective, not minority-class recall.
7. Batter handedness (`stand`) joined the working baseline.
8. `stand x strikes` interaction is the current working baseline.

## Interpretations and hypotheses (not confirmed)

- SL/OTHER's blindness looks like a class-imbalance/objective-function
  issue rather than a missing-feature issue — supported by richer
  sequencing features not moving recall at all, but not proven, since
  not every possible feature has been tried.
- The MLP's gain looks more like genuine improvement in probability
  estimation than sharper-but-wrong decisions, based on widened and
  similarly-or-better-calibrated confidence coverage at most thresholds
  — but this is qualified, not settled, given its new FF-calibration and
  0.80-threshold problems.
- The residual strikes=1/right-handed FS calibration gap remains
  unexplained; no hypothesis has been ventured for it yet.

## Negative results

- Count + game-situation lookup table: log loss 4.18, worse than the
  no-feature baseline.
- Class-weight sweep: raised SL recall to 58% at full balancing, but log
  loss degraded monotonically (0.94 to 1.28) and calibration broke down
  at high thresholds (gaps up to +0.93) — rejected given the project's
  calibration-first objective.
- Four workload features (game pitch count, times through the order,
  in-inning pitch count, prior same-game matchup count), tested
  individually: none beat the handedness baseline beyond noise.
- `stand x full-count` (24-cell) interaction: a small accuracy edge over
  the 6-cell version, but a less statistically certain log-loss gain and
  several very thin cells (n as low as 11) — not adopted.

## Unresolved limitations and open questions

- **No untouched final test set exists.** Everything above comes from
  repeated use of the same 33-game holdout — see the note below.
- `OTHER` remains a heterogeneous bucket (sinker + changeup + sweeper)
  that nothing tested predicts well; unclear whether this is fixable
  within the current target definition.
- The strikes=1/right-handed FS gap, and the MLP's FF-calibration and
  0.80-threshold issues, remain unexplained.
- Whether to adopt the MLP over the logistic interaction model is
  undecided — the trade-off between a uniform accuracy/log-loss gain and
  new localized calibration problems (plus losing linear
  interpretability) hasn't been resolved.
- Season range, location-target definition, and multi-pitcher
  generalization remain entirely unexplored, as originally scoped out at
  the start of the session.

## The 33-game holdout is a development/validation set, not a final test set

Every result in this session — including the ones labeled "confirmed" —
was produced by repeatedly evaluating on the same 33 held-out games while
iterating on features and models. That is development/validation use,
not a final generalization check. Before any claim here is treated as a
statement about true out-of-sample performance, it should be re-checked
against a fresh, never-touched test set.

## Where we are now

The working model is multinomial logistic regression on the richest
sequencing feature set, plus batter handedness and a `stand x strikes`
interaction (accuracy 0.5688, log loss 0.9061 on the development set).
A small MLP beats it on both metrics but introduces new calibration
quirks and hasn't been adopted. SL and OTHER remain unpredicted by
every unweighted model tried; that's an accepted, understood limitation
given the project's calibration-first objective, not an open bug.

## Future directions (not decisions — preserved as ideas only)

These have not been accepted, scoped, or committed to. They're recorded
so they aren't lost, not because they've been chosen as next steps:

1. Ensemble the MLPs (the 10 seeds already trained behave differently
   enough that averaging might help).
2. Add richer batter information beyond handedness (identity-based
   history, swing/take tendencies).
3. Better represent previous-pitch outcomes and sequences (deeper
   history, alternative encodings beyond one-hot class/result).
4. Train a general model across many pitchers and fine-tune to a
   specific pitcher.
5. Learn pitcher and batter embeddings rather than hand-designing
   categorical features.
6. Eventually model entire plate appearances as sequences, using a
   larger multi-pitcher dataset.

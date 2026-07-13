# Pitch Sitch Predictch

Pitch Sitch Predictch is an exploratory machine-learning project about predicting a pitcher’s next pitch from the situation immediately before it is thrown.

The central idea is to represent a pre-pitch situation using information such as:

- the count;
- batter and pitcher handedness;
- inning, outs, score, and baserunners;
- recent pitch history;
- batter and pitcher tendencies;
- other information available before the pitch.

A model could then estimate a probability distribution over the pitcher’s next decision, potentially including pitch type, location, or physical pitch characteristics.

## Initial case study

The first exploration will focus on Kevin Gausman.

Gausman is a useful starting point because he has a large recent sample and a concentrated arsenal built primarily around his four-seam fastball and splitter. This creates a relatively understandable first setting in which to investigate:

- how predictable pitch selection is;
- which contextual features contain useful information;
- how much data is required;
- whether the model can identify particularly predictable situations;
- what level of predictive confidence might become meaningful to a batter or analyst.

Gausman is the first case study, but the project should be designed conceptually around a general pitcher rather than around Gausman-specific assumptions.

## Broad research questions

The project may explore questions such as:

1. How predictable is a pitcher’s next pitch from pre-pitch context?
2. Which parts of the situation contribute the most predictive information?
3. How should a pitch situation be represented?
4. How much pitcher-specific data is required?
5. Is overall accuracy the right measure, or is high accuracy on a smaller set of high-confidence situations more meaningful?
6. How much of pitch selection is shared across pitchers?
7. Can a model trained across many pitchers adapt efficiently to one specific pitcher?
8. Can pitch location or pitch characteristics be modelled after pitch-type prediction is understood?

These are working questions, not a fixed project specification.

## Current phase

The project is currently paused after Session 1 (through 2026-07-13), which completed the goals originally listed here: a working data pipeline, an accepted pitch-type target and game-level train/test split, and a progression of baseline models from frequency lookup tables through logistic regression (with a batter-handedness x count interaction) to a small MLP.

See `notes/research-log.md` for the full session narrative, cumulative results table, and categorized findings, and `notes/decisions.md` for the accepted methodological decisions behind it.

The project remains exploratory. The 33-game holdout used throughout Session 1 is a development/validation set, not an untouched final test set. Several open questions and a set of longer-term ideas (not accepted decisions) are recorded at the end of the research log.

Important methodological choices should be investigated before they are locked into the implementation.

## Working principles

- Use only information that would have been available before the target pitch.
- Begin with simple, interpretable baselines.
- Keep exploratory conclusions separate from confirmed findings.
- Prefer small experiments that answer one question at a time.
- Record important decisions and why they were made.
- Allow the repository structure to develop with the project.
- Avoid adding complex modelling or agent infrastructure before it serves a demonstrated need.

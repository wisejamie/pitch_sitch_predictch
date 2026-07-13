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

For what's currently true — the active model, confirmed findings, and current limitations — read `notes/current-state.md`. It's replaced each session, so it stays short.

For history and reasoning, `notes/research-log.md` is a short index into `notes/sessions/`, where each session's full narrative and results live. `notes/decisions.md` holds the accepted methodological decisions behind all of it.

Important methodological choices should be investigated before they are locked into the implementation.

## Working principles

- Use only information that would have been available before the target pitch.
- Begin with simple, interpretable baselines.
- Keep exploratory conclusions separate from confirmed findings.
- Prefer small experiments that answer one question at a time.
- Record important decisions and why they were made.
- Allow the repository structure to develop with the project.
- Avoid adding complex modelling or agent infrastructure before it serves a demonstrated need.

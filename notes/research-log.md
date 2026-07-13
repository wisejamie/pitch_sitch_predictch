# Research Log

Detailed research history is stored by session under `notes/sessions/`. This file is intentionally just an index — for what's currently true, read `notes/current-state.md` instead; read a session file only when its specific history or reasoning is relevant.

| Session                                     | Focus                           | Main result                                                                                                                                                                                                 |
| ------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [000](sessions/000-project-framing.md)      | Initial project framing         | Defined candidate targets, features, evaluation methods, and longer-term ideas — hypotheses only, not decisions                                                                                             |
| [001](sessions/001-gausman-first-models.md) | First Gausman modelling session | Established baselines (lookup tables through logistic regression with a handedness x count interaction) and found that a small MLP improved on the strongest logistic model, on the 33-game development set |
| [002](sessions/002-mlp-ensemble.md) | MLP probability ensemble | Adopted the 10-seed MLP ensemble as the leading model (beats logistic on accuracy/log loss/Brier); confirmed every individual seed occasionally predicts SL where logistic never does; found FF calibration and OTHER ROC-AUC still favor logistic |

Accepted methodological decisions live in `notes/decisions.md`, not here.

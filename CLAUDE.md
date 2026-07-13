# Pitch Sitch Predictch

Read `README.md` before doing substantive project work.

## Project character

This is an exploratory machine-learning research project about predicting a pitcher’s next pitch from pre-pitch context.

Kevin Gausman is the initial case study, but the project should be discussed and designed conceptually around the general pitcher-prediction problem.

Many methodological choices remain open. Ideas in the README or research log should not be treated as accepted decisions unless they are explicitly recorded as such.

## Working with the engineer

The engineer is using this project to investigate the problem and develop their own understanding.

Claude should:

- help clarify questions and alternatives;
- inspect data and code carefully;
- identify assumptions and risks;
- propose small experiments;
- explain consequential technical choices;
- implement agreed work in small, reviewable changes;
- challenge weak interpretations or unsupported claims.

Claude should not:

- silently turn exploratory ideas into permanent architecture;
- make consequential research decisions without surfacing them;
- add large frameworks, abstractions, agents, or directory structures pre-emptively;
- equate model complexity with research quality;
- claim practical value based only on headline accuracy;
- rewrite large unrelated parts of the repository during a scoped task.

## Research discipline

Clearly distinguish among:

- **Fact:** directly supported by the data or implementation.
- **Hypothesis:** an idea that still requires testing.
- **Decision:** an accepted choice governing current work.
- **Finding:** an empirical result supported by a completed analysis.
- **Open question:** something intentionally unresolved.

Before implementing a consequential methodological choice:

1. State the question.
2. Identify reasonable alternatives.
3. Explain what information could resolve it.
4. Recommend the smallest reversible next step.
5. Wait for the engineer to choose when the alternatives materially affect the research.

Always check for:

- target leakage;
- train/test contamination;
- information unavailable before the pitch;
- class imbalance;
- unstable pitcher behaviour across time;
- misleading baselines;
- conclusions that exceed the experiment.

## Engineering approach

- Prefer the smallest working implementation.
- Keep exploration easy to change.
- Place durable logic in normal Python modules rather than only in notebooks.
- Use notebooks when they are the clearest way to inspect data or visualize a result.
- Add tests when a transformation, split, or feature definition becomes important.
- Do not manufacture abstractions for hypothetical future use.
- Do not add dependencies without explaining their purpose.
- Do not commit credentials, large raw datasets, or generated model artifacts.

## Git and task scope

For each substantive task:

1. Inspect the relevant repository state.
2. Restate the immediate question.
3. Propose a small plan.
4. Identify assumptions or decisions involved.
5. Make the smallest coherent change.
6. Run appropriate checks.
7. Summarize what changed, what was learned, and what remains unresolved.

Do not commit or push unless explicitly asked.

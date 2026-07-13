---
name: wrap-session
description: Wrap up a Pitch Sitch Predictch research session — write a session note, update the research-log index, refresh notes/current-state.md, and propose (not apply) decisions.md changes only when a real methodological decision was made. Use when the user asks to pause, wrap up, or summarize a session.
---

# Wrap session

This project keeps a three-layer documentation hierarchy: `notes/current-state.md` (short, replaced each session), `notes/decisions.md` (accepted decisions, append-only), and `notes/sessions/<NNN>-<slug>.md` (one file per session, detailed narrative), indexed by `notes/research-log.md`. This skill keeps that structure intact when a session ends.

## Steps

1. **Inspect before writing.** Look at repo state, which scripts were run this session, and their saved outputs. Re-run a script if you're unsure of an exact number rather than trusting a rounded value from the conversation.
2. **Create exactly one new session note** at `notes/sessions/<NNN>-<short-slug>.md`, continuing the numbering of existing session files. Follow the structure already established in `notes/sessions/001-gausman-first-models.md`: a narrative story, a cumulative results table (authoritative numbers only), the current feature set if it changed, then clearly separated sections for confirmed results, a pointer list to methodological decisions, interpretations/hypotheses, negative results, unresolved limitations, a development-set-status note, "where we are now", and future directions.
3. **Update `notes/research-log.md`** — add exactly one row to the index table for the new session. Do not let the index grow beyond one line per session; do not move any narrative content into this file.
4. **Rewrite `notes/current-state.md` in place.** It is meant to be replaced each session, not appended to — overwrite it so it reflects the current model, its metrics, the strongest confirmed findings, and current limitations, dropping anything superseded.
5. **Propose `notes/decisions.md` changes only if a real methodological decision was made this session** — a choice the engineer explicitly accepted, not just an experiment result or a tested-and-rejected option (those belong in the session note's negative-results section). Show the proposed entry and wait for confirmation before adding it.
6. **Keep these categories visually separate, in every file touched:** confirmed results, interpretations/hypotheses, negative results, accepted decisions, and open questions. Don't blend them into one undifferentiated narrative.
7. **Never invent or manually retype a metric** when an authoritative saved script output exists. If a number in conversation looks rounded or uncertain, re-run the script that produced it.
8. **Keep documentation readable, not exhaustive.** A session note should tell the story and the key numbers — it doesn't need to reproduce every diagnostic table generated during the session.

## What this skill does not do

Does not reorganize `pitch_sitch/`, move or merge scripts, add dependencies, or change any experimental results, models, or accepted methodology. It only documents what already happened.

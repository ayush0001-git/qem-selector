# final-reporter — end-result documentation pass (2026-07-22)

Owned outputs, all written this pass:

- `END_RESULT.md` (new) — the student's plain-language end-result document:
  what the experiment was, findings F1–F7 (each one bold sentence + verified
  numbers), paper mapping to RESEARCH_ANGLES.md (Angle 3 headline / Angle 2
  supporting / 9.3-QPU-min boundary-test spec), honest limitations, next 5
  actions.
- `PROJECT_STATUS.md` — refreshed:
  - header intro: sweep now complete + verified;
  - §3 rewritten as the RESEARCH-RUN snapshot (1620 rows / 540 groups, winner
    counts, error magnitudes, refusal-adjusted CDR 83.7%, headline model
    CV/LOFO/LOBO/LODO both labels, per-seed ablation deltas, n=3 hardware
    bridge, mandatory caveats); old small-run snapshot demoted to
    "§3-old SUPERSEDED" with a do-not-quote warning;
  - §3a parenthetical updated (sweep has run);
  - §4.2 annotated (n=3 hardware run exists, ~9.3 QPU-min left) and §4.3
    marked SUPERSEDED (dataset scale);
  - §5 next-steps rewritten: old step 1 (run sweep) marked DONE; new list =
    paper skeleton -> Angle 3 sim overlay -> QPU-min on hardware boundary
    test -> Angle 2 heatmap -> re-scan literature + mentor.

Sources honored: trainer headline (final-trainer), insight findings
(final-insight / docs/ANALYSIS.md), verifier audit (final-verifier — PASSED,
zero discrepancies; its one nuance — the Q2 full-menu-wins list names 23/30,
non-exhaustive — is respected: END_RESULT.md says "concentrated at", not an
exhaustive list). No numbers invented; all traced to metrics.json /
ANALYSIS.md / verifier tables.

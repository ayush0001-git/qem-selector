# Integrator V2 — 2026-07-23

Full report appended to PROJECT_STATE.md ("2026-07-23 — V2 integration COMPLETE").

- Suite: 737/737 green (hw_first_run.yaml already reverted to
  `hardware_confirmed: false`; the builders' lone red is gone, no test weakened).
- Fresh boundary_smoke e2e (24 units; integrator added FakeManilaV2@x0.5):
  base_shots schema OK, 7 techniques OK, dial-down raw error monotone
  (0.046 < 0.067 < 0.103), determinism vs prev run exact, both+significant
  fv2 calibrated training OK (9/24 ties), stats.json checklist passed,
  report §8/§9 rendered, recommend confident (exit 0) + abstain (exit 2),
  overlay 86.7 % agreement / 15 points / PNG.
- Regressions: tiny.yaml SHA256-identical (rem 11/cdr 8/zne 1); research
  aggregated re-train full-dict identical to stored metrics.json.
- boundary.yaml sized from measured bench: 7.44 s/unit x factors -> ~5.2 h
  expected (<= 8 h with 1.5x safety); grid untrimmed; math in config header.
- Owned edits: configs/boundary_smoke.yaml, configs/boundary.yaml (header),
  PROJECT_STATE.md, this note. No src/tests changes. Sim-only throughout.

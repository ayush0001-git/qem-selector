# Notes — noise-scaling-builder (2026-07-21)

## Delivered (ownership: src/qemsel/backends.py, tests/test_noise_scaling.py)

Noise-scaled backend variants named `"<FakeName>@x<scale>"` (e.g.
`FakeManilaV2@x1.5`, `FakeLagosV2@x2.0`) in `make_executor` and
`get_backend_info`. New public helper `parse_backend_name(name) ->
(base, scale)`; new private helpers `_collect_target_errors(target)` and
`_build_scaled_noise_model(backend, scale)`. 39 new tests in
`tests/test_noise_scaling.py`, all fast (11 s).

## Semantics

- **Grammar:** exactly one `@`, suffix `x<float>`, scale finite and > 0.
  Anything else -> ValueError (missing `x`, non-numeric, 0, negative,
  inf/nan, double `@`, empty base). `ibm_*` names accept NO suffix ever
  (noise scaling is simulation-only; `ibm_brisbane@x2.0` -> ValueError).
  Unknown base with valid suffix -> the usual "unknown backend" ValueError.
- **Scale 1.0 (plain name or `@x1.0`) routes through the verbatim pre-change
  `AerSimulator.from_backend` path — byte-identical.** Anchored by a
  hardcoded regression value captured by running the PRE-change code:
  `make_executor("FakeManilaV2", 256, 7)` on Bell `ZZ` == `0.8671875`
  (= 222/256, exact float; tested with `==`, not approx, for both the plain
  name and the `@x1.0` spelling).
- **Scale != 1.0:** SAME `from_backend` simulator (identical coupling map,
  basis gates, gate directions => identical transpilation across scales),
  but `set_options(noise_model=...)` swaps in a synthetic model built from
  `backend.target` calibration:
  - per-gate depolarizing, `p = min(scale * calibrated_error, 0.9)`,
    attached to the exact calibrated `(gate, qargs)` entries (so the routed
    non-device-edge coverage from the 2026-07-21 review fix is preserved —
    regression-tested on Lagos pair (3,4) at x2.0);
  - symmetric readout confusion per qubit,
    `p01 = p10 = min(scale * calibrated_readout_error, 0.45)`. (The Target
    stores ONE symmetric readout number per qubit — the same number the
    plain from_backend model uses, so symmetric is consistent, not a loss.)
- **Thermal relaxation deliberately NOT added** (documented in the module
  docstring): the calibrated gate error already CONTAINS the relaxation
  contribution, so depolarizing at `scale * total_error` PLUS a separately
  scaled T1/T2 channel would double-count and make the dial super-linear.
  This is a controlled noise-strength DIAL, not a claim of physical
  fidelity at scale != 1. (T1/T2 and durations ARE available in the target
  if someone later wants a physically decomposed model.)
- **`get_backend_info` on a scaled name returns the scaled-and-capped
  per-entry averages** — exactly what the noise model applies — so
  `feat_backend_avg_2q_error` / `feat_backend_avg_readout_error` flow
  through `features.extract_features` automatically (tested: x2.0 gives
  exactly 2x both backend features; circuit features unchanged). Cached per
  full name, copy-safe, same 6-key schema, `name` echoes the full string.
- **Caps apply ONLY on the scaled path** (0.9 gate / 0.45 readout;
  stability guards, not physics). Documented quirk: FakeLagosV2 q2 stores
  46.4% readout, so `@x1.0` reports 0.464 (uncapped, verbatim plain) while
  any scaled variant caps that qubit at 0.45.
- **Determinism:** scaled model is built from static calibration only; same
  `(name, shots, seed)` -> identical results, across scales (tested within
  one executor, across rebuilt executors).

## Monotonicity verification (test + build-time probe)

Fixed 10-CNOT circuit (|11> + 10 cx, ideal <ZZ> = +1), FakeManilaV2,
8000 shots: |raw error| x1.0 ~= 0.21, x2.0 ~= 0.36, x3.0 ~= 0.49. Tests
assert x2 > x1 + 0.05 and x3 > x2 + 0.05 (shot sd ~= 0.011 — generous).
Readout-only scaling verified separately (measure-only workload, x2 error
~2x the x1 error).

## Verification

- `tests/test_noise_scaling.py`: 39/39 green, 11 s.
- FULL suite: **409 passed, 0 failed** (3 benign warnings), 235 s.
  Note for the orchestrator: during my first two full-suite runs the suite
  was a moving target (parallel agents landing mitigation `raw_plus`,
  model min-class handling, experiment changes mid-run — transient
  failures in THEIR files that all passed on re-run); the final run above
  is fully green at 409 tests.
- Also fixed in passing (my file): the stale "~27% readout on q0/q1"
  FakeLagosV2 comment in backends.py (PROJECT_STATUS §6 item 9) now reads
  q0 16.9% / q1 13.6% / q2 46.4%.

## Seam notes for the parallel agents

- **experiment.py owner:** `_validate_config` currently rejects any backend
  not in `backends.BACKENDS` (and not `ibm_*`), so scaled names in a YAML
  config will be refused until you accept them. Suggested check:
  `base, scale = backends.parse_backend_name(b)` then validate `base` in
  `BACKENDS` (parse raises ValueError on malformed suffixes for you, and
  already blocks `ibm_*@x...`). `get_backend_info`/`make_executor`/
  `extract_features` all accept the full scaled string as-is — pass it
  through unchanged so `results.csv` records the scaled name in the
  `backend` column.
- `BACKENDS` itself still holds ONLY the 4 plain names (unchanged list —
  scaled variants are constructed names, not registry entries).
- REM calibrates through the executor, so it sees the scaled readout noise
  automatically; nothing to do on the mitigation side.

# Adversarial tester T-backends — round 1 (2026-07-23)

Verdict: PASS — 0 critical, 0 major, 0 minor. 167 executed checks + full
tiny.yaml regression rerun, all green. Verified by RUNNING code (scripts in
session scratchpad: `t_backends_adversarial.py`, `t_backends_monotone.py`).

## What was tested

1. Monotone error reduction (deep CNOT, generous shots)
   2-qubit circuit, 60 CX (identity), pauli ZZ, 32768 shots, seeds {0,1,2},
   on BOTH FakeManilaV2 and FakeLagosV2:
   - Manila mean |err|: 0.1465 @x0.25 < 0.2720 @x0.5 < 0.4171 @x1.0
   - Lagos  mean |err|: 0.2573 @x0.25 < 0.4545 @x0.5 < 0.5192 @x1.0
   - Per-seed monotone: 0/3 breaks on both devices (all 6 chains strictly
     ordered), not just in the mean.

2. get_backend_info scaling — all 4 BACKENDS x {0.25, 0.5}
   Every field of avg_1q_error / avg_2q_error / avg_readout_error /
   max_readout_error equals scale * plain to rel_tol 1e-9 (caps never bind
   below 1.0, incl. FakeLagosV2's 46.4% q2 readout: 0.5*0.464 = 0.232 < 0.45),
   strictly below plain, name echoes the full suffixed name, n_qubits and key
   set unchanged. `@x1.0` returns plain numbers verbatim with suffixed name
   echo. Scaled-info cache is not corruptible via the returned dict (mutation
   probe re-fetched clean).

3. Bad scales -> ValueError (checked on parse_backend_name AND
   get_backend_info AND make_executor, 3x16 = 48 checks; exception TYPE
   asserted, not just "raises"):
   x0, x0.0, x-0, x-1, xabc, bare `@x`, bare `@`, xnan, xinf, xInfinity,
   double suffix `@x1.5@x2.0`, `@y2.0`, `@2.0` (missing x), empty base
   `@x0.5`, and `ibm_brisbane@x0.5` / `@x1.0` (sim-only guard fires BEFORE
   any hardware dispatch — no ibm_ path touched). Unknown base with a valid
   scale (`Bogus@x0.5`) also ValueError on both consumers.

4. Plain names byte-identical
   - Reran configs/tiny.yaml myself to a fresh out dir: results.csv SHA256
     CBDF857DE0D5D0C7716939BB7B7F12F692A9D855D3D9B3338838923D2CDC3D33,
     6248 bytes — identical to stored results/tiny/results.csv
     (independent reproduction of the integrator's regression claim).
   - `<dev>@x1.0` executor value == plain executor value (identical floats)
     on Manila and Lagos.

5. Extras: executor determinism at x0.25/x0.5 (same name/shots/seed twice ->
   identical floats); executor does not mutate the input circuit (XZ pauli,
   basis-change path).

## Repro
venv = "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe";
run the two scratchpad scripts from the project root; regression:
`run_experiment.py --config configs\tiny.yaml --out <fresh dir>` then
Get-FileHash vs results\tiny\results.csv.

Sim-only throughout; no ibm_* backend touched (the only ibm_ names used were
in negative tests that raise in parse_backend_name before dispatch).

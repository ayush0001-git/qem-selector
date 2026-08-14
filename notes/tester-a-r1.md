# Tester A — Round 1 (unit + determinism) — 2026-07-21

Independent verification of the integrator's claims. I did not write this code.
Verdict up front: **no critical, no major failures. 1 minor finding.**
All checks below were run with `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe"`.
My probe scripts lived in the session scratchpad (det_test.py, edge_test.py,
hash_run.py); every failure-relevant repro is inlined below.

## 1. Full test suite

Command:

    cd "E:\quatum  computiiing\qem-selector"
    .venv\Scripts\python.exe -m pytest tests -q

- Run 1 (with `-rA --durations=15`): **251 passed, 3 warnings in 87.5 s**.
- Run 2 (plain `-q`, to capture the warnings summary and check flakiness):
  **251 passed, 3 warnings in 176.6 s**.
- Suite is stable across two consecutive full runs; nothing was deselected —
  the entire suite fits far under the 10-minute cap, so "slow tests" required
  no special handling.
- The 3 warnings are all the same benign mitiq message from
  `mitiq/zne/inference.py:88` ("The input circuit is very short...") on the
  three `test_noiseless_bell_expectations[* -zne]` cases. Matches the
  integrator's claim exactly.

## 2. Determinism (all PASS)

In-process, script `det_test.py` (results):

- `generate_suite(config)` called twice with a 40-circuit config (5 families
  x [2,3] qubits x [4,8] depths x seeds [0,1], incl. near_clifford params):
  all QASM strings (`qasm2.dumps`) and CircuitSpecs pairwise identical.
- Generators do not touch the global numpy RNG state (checked before/after).
- `ideal_expectation` called twice on 10 suite circuits: bit-identical floats.
- `mirror_circuit(3,6,seed)` for seeds 0-4: `ideal_expectation == +1.0`
  EXACTLY (the integrator's snap fix holds; known-answer contract verified).
- `make_executor("FakeManilaV2", 1000, 123)`: same executor called twice ->
  identical; a SECOND independently built executor with the same
  (backend, shots, seed) -> identical value. A different seed (124) gives a
  different value, so the seed is genuinely wired through (not ignored).
- Mixed pauli "XIZ" through the executor twice -> identical.
- `apply_technique` for zne / cdr / rem called twice with same inputs ->
  bit-identical results each.

Cross-process (stronger than asked — catches PYTHONHASHSEED / import-order
nondeterminism), script `hash_run.py` run in two fresh interpreters:

- sha256 over all 40 circuit_ids + QASM strings:
  `55fcd347ef8fe53c0dbb9c5abfc73ee81de49b78a2361f81b27c103ea0e1b0e3` both runs.
- `ideal_expectation` repr and executor value bit-identical across processes
  (`0.01052368290369303`, `0.006`).

## 3. Edge cases (script edge_test.py — all PASS except finding M1 below)

- **2-qubit minimum circuit through the FULL `apply_technique`** (real noisy
  FakeManilaV2 executor, 1000 shots) for raw/zne/cdr/rem: all return finite
  floats, no exceptions. Sanity bonus: every mitigated value moved toward the
  ideal (ideal -0.7172; raw -0.6160, zne -0.6780, cdr -0.7195, rem -0.6829).
- **Pauli with 'I' through the expectation helper**
  (`backends.expectation_from_counts`) with hand-built counts
  `{"01": 600, "10": 400}`: 'ZI' -> -0.2, 'IZ' -> +0.2, 'ZZ' -> -1.0 — the
  little-endian key handling is CORRECT (rightmost bit = q0, pauli[0] = q0).
  All-'I' returns 1.0; space-containing keys handled; empty counts and short
  keys raise ValueError as documented.
- 'I'-containing paulis also verified through `ideal_expectation` (Bell 'ZI'
  == 'IZ' == 0.0 exactly, 'II' == 1.0) and through the noisy executor with an
  endianness canary (x on q0 of 3 qubits: 'ZII' ~ -1, 'IZI' ~ +1). Consistent
  everywhere.
- 'ZI' and all-'I' paulis through all four techniques on the 2q circuit: all
  finite, no exceptions (REM's identity-observable early return works).
- `ideal_expectation` rejects wrong-length, invalid-char, and lowercase pauli
  strings with ValueError.
- All 5 families build at n_qubits=2 AND n_qubits=1 (contract says >= 1) with
  no measurements/clbits; ideal works on the 1q circuits (ghz_plus's
  cx-padder is correctly excluded for 1 qubit).
- **recommend.py on a missing model path** — see finding M1.
- Consistency check of the other CLIs with missing inputs: `train_model.py
  --data <missing>` -> clean argparse error; `make_report.py` with missing
  results -> clean `error: results file not found: ...`; `recommend.py --qasm
  <missing>` -> clean `error: QASM file not found: ...`. Only the --model
  path leaks a traceback.

## Findings

### M1 (MINOR) — recommend.py missing --model prints a raw traceback

Repro:

    cd "E:\quatum  computiiing\qem-selector"
    .venv\Scripts\python.exe scripts/recommend.py --model results/does_not_exist/model.joblib --backend FakeManilaV2 --demo ghz_plus --qubits 3 --depth 4 --seed 0

Output (exit code 1):

    Traceback (most recent call last):
      File "E:\quatum  computiiing\qem-selector\scripts\recommend.py", line 106, in <module>
        raise SystemExit(main())
      ...
      File "E:\quatum  computiiing\qem-selector\src\qemsel\recommend.py", line 35, in _load_bundle
        raise FileNotFoundError(
    FileNotFoundError: model file not found: results\does_not_exist\model.joblib — run scripts/train_model.py first to produce model.joblib

Why minor, not major: the exit code is nonzero and the message is clear and
actionable (it even tells you to run train_model.py first); the library-level
contract (`FileNotFoundError` from `qemsel.recommend.recommend`) is exactly as
documented and unit-tested. The defect is CLI polish/consistency only: the
sibling error paths in the same script (`--qasm` missing -> `SystemExit("error:
...")`) and the other two CLIs print clean one-line errors, so `main()` in
`scripts/recommend.py` (line ~96) should catch `FileNotFoundError`/`ValueError`
from `recommend(...)` and convert to a clean `SystemExit` message the same way.

## Not covered here (left to other testers / later rounds)

- experiment.py resume/crash-safety and features.py internals (unit tests
  cover them; I did not independently re-probe).
- Statistical quality of the tiny-run numbers (winner distribution etc.) —
  properties of a 20-row run, not unit-level failures.

## Bottom line

Integrator's claims verified: 251/251 twice, warnings as described, mirror
ideal exactly +1.0, determinism holds in-process AND cross-process, all
required edge cases pass. Zero critical, zero major; one minor CLI-polish
finding (M1).

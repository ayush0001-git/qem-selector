# Independent tester — real-hardware path (2026-07-21)

Role: adversarial verification of the new hardware path (src/qemsel/hardware.py,
scripts/test_hardware_connection.py, scripts/estimate_hardware_cost.py,
dispatch seams in backends.py/experiment.py, configs/hw_first_run.yaml).
I did not write any of it. NO real hardware call was made at any point;
all probes used temp credential files or refused before any network path.

## Verdict: PASS — zero critical/major findings. 3 minor.

## 1. Full suite + fake-path regression

- `pytest tests -q` (slow included): **300/300 passed** in 164 s (266 old + 34
  new in tests/test_hardware.py). Only the 3 known benign mitiq short-circuit
  UserWarnings. Repro:
  `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" -m pytest tests -q`
- **Network-isolation proof:** re-ran tests/test_hardware.py +
  tests/test_experiment.py + tests/test_backends_ideal.py with a pytest plugin
  that monkeypatches `socket.socket.connect` / `socket.create_connection` to
  raise: **114/114 passed with all sockets hard-blocked.** The mocks are real;
  zero network calls occur even with the user's REAL credentials sitting in
  configs/hardware.yaml during the run.
- **Fake-backend path untouched (determinism diff):** re-ran
  configs/tiny.yaml into a scratch dir and diffed against the pre-hardware
  baseline results/tiny/results.csv: 20/20 rows, every column identical at
  atol 1e-12 (values bit-identical), winners rem 11 / cdr 8 / zne 1 and
  cost-aware rem 13 / cdr 6 / raw 1 — exactly the PROJECT_STATE numbers;
  errors.log identical (same 8 intentional CDR refusals).

## 2. Adversarial credential/CLI probes (all with TEMP files, never the real one)

test_hardware_connection.py (`--credentials <tmp>`):
- missing file -> exit 2, step-by-step fix guidance, no traceback.
- blank/null token -> exit 2, "blank/placeholder" guidance, no traceback.
- malformed YAML embedding a fake secret -> exit 2,
  "ParserError; message withheld" — the fake secret does NOT appear anywhere
  in the output (verified by grep).
- non-mapping YAML (list) -> exit 2, clean "must be a YAML mapping" message.

Token-leak sweep: a checker script loaded the REAL token/CRN internally
(never printed) and scanned all 9 captured CLI outputs: **zero leaks**.
grep of src/ + scripts/ shows no print/log/repr of the credentials dict
anywhere; hardware.py withholds YAML parser text by design (tested in suite:
test_load_credentials_parse_error_never_quotes_content).

Refusal gates (safe: every one refuses BEFORE any network object is built):
- `run_experiment.py --config configs/hw_first_run.yaml` (hardware_confirmed:
  false, real creds present) -> **exit 2**, message states 28 jobs, ~85
  QPU-seconds (~1.4 min) of the free 10 min/month and how to confirm. No
  output dir created (validation precedes mkdir).
- oversized ibm config (432 jobs, ~4403 s) -> **exit 3** budget refusal with
  pointer to the estimator and --force-hardware.
- oversized + `--force-hardware` -> still **exit 2** (confirmation gate is
  not bypassable by the budget flag) with the full cost text.
- `hardware_confirmed: "yes"` (truthy string) is rejected — must be exactly
  True (covered by test_validate_config_ibm_without_confirmation_states_cost).

Cost estimator (`estimate_hardware_cost.py`):
- hw_first_run.yaml -> exit 0: 4 circuits x 1 backend x (raw 1 + zne 3 +
  rem 3) = 28 jobs, 3.02 s/job, ~85 s = 14% of budget, "fits" verdict plus a
  note that hardware_confirmed is still false. Matches the config header math.
- oversized -> exit 3 "DOES NOT FIT"; missing config / unknown technique ->
  exit 2 one-line errors. Assumptions echoed in every output.
- **Cost-model truthfulness verified against mitigation.py:** executor
  invocations are raw 1, zne len(ZNE_SCALE_FACTORS)=3, rem 1+2 calibration,
  cdr 1+10 training — exactly SHOT_MULTIPLIER, which the estimator uses.

## 3. Line-by-line: can the scripts submit a job / burn QPU time?

**No.**
- estimate_hardware_cost.py: imports only yaml + qemsel.hardware; calls
  estimate_config_qpu_seconds -> generate_suite (local circuit construction)
  + arithmetic. Never touches credentials, never constructs
  QiskitRuntimeService. Zero submission primitives (grep: no
  Sampler/Batch/Session/.run anywhere).
- test_hardware_connection.py: load_credentials (local read), get_service
  (QiskitRuntimeService auth only), list_real_backends (backends() +
  status(), read-only GETs), service.usage() (read-only, wrapped
  best-effort). No Sampler/Batch/Session/.run in the file. Library exception
  text is token-redacted before printing.
- hardware.py executor: the budget check `_charge_qpu_seconds` runs BEFORE
  `Batch`/`SamplerV2.run` and raises without charging; identity pauli
  short-circuits to 1.0 with no job (all verified by mocked tests
  test_budget_guard_hard_stops_before_submission,
  test_executor_identity_pauli_submits_nothing).

## Safety posture confirmed

- .gitignore line 18 covers configs/hardware.yaml (project is not yet a git
  repo — nothing can be committed today; the ignore is in place for when it
  becomes one).
- Real credentials file parses as a clean mapping (token 44 chars, CRN set,
  channel default ibm_quantum_platform, no qpu_seconds_cap -> default 120 s
  in-process cap applies). Inspected only via a script printing
  types/lengths/booleans — no value ever echoed.

## Minor findings (no action required to run)

1. **Batches/services never closed by the experiment loop** —
   experiment.py builds one executor per (circuit, backend) unit
   (line ~566) and never calls `executor.close()`; for ibm_* runs each unit
   opens its own QiskitRuntimeService + Batch that stays open until IBM's
   server-side inactivity timeout. Not a billing hazard (Batch mode bills
   per job, not per open batch) but it means 4 auth handshakes + 4 stale
   batches for hw_first_run; a shared service + close-on-finish would be
   tidier and slightly faster.
2. **Token redaction is exact-match only** (test_hardware_connection.py
   `_redact`): a library that echoed the token URL-encoded/truncated would
   evade it. Low risk (qiskit-ibm-runtime does not echo tokens); noted as a
   residual.
3. `import sys` in scripts/test_hardware_connection.py is unused (lint).

## Repro commands

    "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" -m pytest tests -q
    "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" scripts/estimate_hardware_cost.py --config configs/hw_first_run.yaml
    "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" scripts/run_experiment.py --config configs/hw_first_run.yaml --out results/hw_refusal_check   # exit 2, refuses with cost
    "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" scripts/test_hardware_connection.py --credentials <some-temp>.yaml   # exit-2 guidance paths

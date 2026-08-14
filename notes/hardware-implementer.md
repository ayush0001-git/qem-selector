# Hardware implementer — real IBM Quantum path (2026-07-21)

**Status: implemented + fully mock-verified. Full suite 300/300 green
(266 existing + 34 new). No network call was ever made; no job was or can be
submitted without the explicit gates below.**

## What was verified BEFORE writing code (introspection of installed libs)

Read the installed `qiskit_ibm_runtime` 0.48.0 + `qiskit` 2.5.0 sources under
`.venv\Lib\site-packages` (not docs from memory):

- **Channel names** (`qiskit_runtime_service.py`): valid values are
  `ibm_quantum_platform` (recommended, the new quantum.cloud.ibm.com),
  `ibm_cloud` (legacy alias, redirected to the same platform), `local`.
  Constructor: `QiskitRuntimeService(channel=..., token=..., instance=<CRN>)`.
- `service.backends()/backend(name)/least_busy()`; `backend.status()` returns
  `operational`, `status_msg`, `pending_jobs` (queue depth).
- `service.usage()` returns a dict with `usage_limit_seconds` /
  `usage_consumed_seconds` and computes `usage_remaining_seconds` — used
  (best-effort) by the connection-check script.
- **SamplerV2** validates ISA circuits against `backend.target` before
  submission (`base_primitive.py` -> `validate_isa_circuits`), so
  transpilation to the target is mandatory. `Batch(backend=...)` +
  `SamplerV2(mode=batch)`; `job.result()[0].join_data().get_counts()`.
- `qiskit.transpiler.generate_preset_pass_manager(optimization_level=0,
  backend=..., seed_transpiler=...)` exists in qiskit 2.5.0. Level 0 keeps
  ZNE-folded G Gdag G sequences (routing + basis translation still run).

## Files

- **`src/qemsel/hardware.py`** (new): `load_credentials` (None when file
  missing/token blank; YAML parse errors re-raised WITHOUT the parser message
  because it quotes file content), `get_service`, `list_real_backends`,
  `get_real_backend_info` (same key contract as `backends.get_backend_info`,
  summarized from the live `backend.target`, cached), `make_real_executor`
  (same executor contract as `backends.make_executor`: measure_all on a COPY,
  X/Y basis rotation, ISA transpile at optimization_level=0, submit via
  SamplerV2 in a shared lazily-opened Batch, counts -> expectation through
  the existing `backends.expectation_from_counts` so endianness stays solved
  in one place; `executor.close()` closes the batch),
  `estimate_job_qpu_seconds` / `estimate_config_qpu_seconds` (cost model),
  in-process usage ledger (`qpu_seconds_used`/`reset_qpu_usage`) with a
  **hard stop BEFORE submission** past a configurable cap (default 120 s;
  overridable per-executor or via `qpu_seconds_cap` in hardware.yaml).
  Runtime symbols are imported at module level ON PURPOSE as the monkeypatch
  site for tests.
- **`src/qemsel/backends.py`**: dispatch seam only — names starting `ibm_`
  route `make_executor`/`get_backend_info` to `qemsel.hardware` (local import,
  no cycle). Fake-backend logic untouched. `RealHardwareBackend` kept as a
  legacy non-constructible stub (existing test still passes) whose message now
  points at the real entry path.
- **`src/qemsel/experiment.py`**: `_validate_config` allows `ibm_*` backends
  ONLY when credentials load AND `hardware_confirmed: true` (exactly True);
  both refusal messages state the estimated cost (jobs + QPU-seconds/min) and
  how to confirm. Width pre-check skips `ibm_*` (would hit the network; the
  executor re-checks at call time).
- **`scripts/estimate_hardware_cost.py`** (new): local-only cost breakdown
  (per-technique jobs/unit, total jobs, QPU-seconds, fits-10-min verdict).
  Exit 0 fits / 2 config error / 3 does not fit.
- **`scripts/run_experiment.py`**: prints the estimate for `ibm_*` configs and
  refuses (exit 3) when it exceeds the free monthly budget unless
  `--force-hardware`; config ValueErrors (incl. the gates) now print as clean
  one-line errors (exit 2) instead of tracebacks.
- **`scripts/test_hardware_connection.py`** (new): reads credentials,
  connects, lists real backends with queue depth + remaining free-plan usage
  when the API exposes it. Submits NOTHING. Distinct guidance for missing
  file (exit 2), blank token (exit 2), bad API key vs bad CRN vs other
  (exit 1); any library message is token-redacted before printing.
- **`configs/hw_first_run.yaml`** (new): mirror_circuit + layered_random at
  n={2,3}, depth 4, seed 0 (4 circuits), ONE `ibm_brisbane` placeholder,
  1024 shots, techniques raw+zne+rem (CDR excluded — 11x cost, comment
  explains), `min_abs_ideal: 0.25` (free local screen), and
  `hardware_confirmed: false` by default. Estimator: 28 jobs, ~85 s
  (~1.4 min) — under the ~2-min design target and the 120 s in-process cap.
- **`tests/test_hardware.py`** (new, 34 tests): credentials handling (incl.
  a no-content-leak test for YAML parse errors), get_service channel/kwargs,
  backend listing, target summarization equality vs the FakeManilaV2
  reference, dispatch routing both seams (fake path proven unchanged),
  credential/confirmation gates in `_validate_config`, estimator arithmetic
  + the shipped hw_first_run.yaml budget invariant, executor
  counts->expectation with a fake sampler stack, X-basis rotation, identity
  short-circuit, shared-Batch reuse, pauli/width validation, and budget
  hard-stop (per-arg cap, hardware.yaml cap, process-wide accumulation).
- README "Switching to real IBM hardware" rewritten to the implemented flow;
  layout table updated. `configs/hardware.yaml`: only the stale comment block
  replaced (credential lines untouched).

## Cost model (documented assumptions, deliberately conservative)

1 executor call = 1 single-circuit job; estimate = 2.0 s/job overhead +
1 ms/shot (real devices run ~0.3–0.5 ms/shot at default rep delay, so this
carries 2–4x margin); `min_abs_ideal` screening ignored (upper bound). It is
a planning heuristic, not IBM's billing formula. Free plan budget modeled as
600 s/month.

## Safety notes

- `configs/hardware.yaml` is gitignored (verified). The user has already
  pasted real credentials there; nothing in code, tests, logs, or these notes
  reproduces them, and the connection script redacts the token from any
  library error text it prints.
- Triple gate before any job: credentials present -> `hardware_confirmed:
  true` in the config (exactly True; truthy strings rejected) -> in-process
  estimated-QPU cap (default 120 s) checked BEFORE each submission.
- Verified live-refusal behavior of the CLI with the real credentials file
  present (local file read only, no network): unconfirmed config -> exit 2
  with cost-stating message; oversized config -> exit 3 unless
  `--force-hardware`.

## For the user (first hardware run, ~5 commands)

1. `python scripts\test_hardware_connection.py` — checks key + CRN, lists
   devices you can access (nothing submitted).
2. Put an accessible device name into `configs\hw_first_run.yaml`.
3. `python scripts\estimate_hardware_cost.py --config configs\hw_first_run.yaml`
4. Set `hardware_confirmed: true` in that config (your cost consent).
5. `python scripts\run_experiment.py --config configs\hw_first_run.yaml --out results\hw_first_run`

## Known limitations / follow-ups

- Real shot noise is not seedable; `seed` only fixes transpilation. Hardware
  rows are therefore not bit-reproducible (document in the paper).
- Each executor call is one job; REM re-calibrates per unit (3 jobs) — a
  shared readout calibration across units would save ~30% on REM-heavy runs.
- `service.usage()` is only read by the connection script; the in-process cap
  does not query the server-side remaining budget (keeping the library path
  network-free until submission time).
- REM's symmetric-readout assumption (PROJECT_STATUS §4.8) now matters on
  real, asymmetric hardware — the affine-offset fix sketched in the code
  review becomes relevant for interpreting hw_first_run results.

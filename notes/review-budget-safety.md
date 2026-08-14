# Budget-Safety Review — real-hardware path (10 QPU-min/month Open Plan)

Reviewer: budget-safety subagent, 2026-07-21. Verdict: **PASS — zero critical/major findings.**
All verification was done by reading code + running fully mocked end-to-end checks
(scratchpad `budget_review_check.py`) and local-only CLI runs. Zero network calls, zero jobs.

## What was verified (mock-executed, not speculated)

Ran the REAL `run_experiment` + REAL mitiq mitigation code with `qemsel.hardware`'s
`QiskitRuntimeService`/`SamplerV2`/`Batch`/pass-manager monkeypatched to counting fakes:

1. **Confirmation gate**: `hardware_confirmed: false` (and the string `"true"`) →
   `ValueError` before ANY job or even service construction; cost stated in the error;
   no `run_meta.json`/out-dir side effects. Live CLI: shipped `hw_first_run.yaml` →
   exit 2 with cost line; oversized config (432 jobs, ~2633 s) → exit 3 even with
   `hardware_confirmed: true`; no out dirs created.
2. **Job count matches the estimator EXACTLY**: hw_first_run shape (4 circuits x 1 backend,
   raw+zne+rem, `min_abs_ideal: 0`) submitted **28 jobs** = estimator's 28
   (raw 1 + zne 3 + rem 3 per unit; every job single-circuit). With the shipped
   `min_abs_ideal: 0.25` it actually submitted **21** (one layered_random unit screened
   free locally) — estimator is a true upper bound.
   In-process ledger after the run: 84.672 s = 28 x 3.024 s, matching the estimator to 1e-6.
3. **No retries**: a job whose `result()` raises is submitted exactly once; the technique
   records NaN + an errors.log line; the run continues. Re-running the same out_dir
   (resume) submits **zero** new jobs. No retry loop exists anywhere in the submission path
   (grep: only `hardware.py` constructs SamplerV2/Batch/Service in `src`; only
   `test_hardware_connection.py` in `scripts`, and it submits nothing).
4. **Batch lifetime**: genuinely shared — ONE `Batch` per executor (per unit), opened
   lazily on the first call and reused by all 7 calls of that unit (verified: 4 units →
   4 Batch instantiations, not 28).
5. **Budget cap**: with a 10 s cap from the credentials file, exactly 3 jobs (3.024 s each)
   were submitted and the 4th refused BEFORE submission (`HardwareBudgetExceededError`);
   ledger never exceeded the cap; refusals are not charged.
6. **Token hygiene**: sentinel fake token appeared in none of `run_meta.json`,
   `results.csv`, `errors.log`, stdout. `run_meta.json` embeds only the experiment config
   (credentials live in a separate file, never merged). YAML parse errors withhold parser
   text (tested); `test_hardware_connection.py` redacts the token from library errors.
   `configs/hardware.yaml` is in `.gitignore`; note the project is currently NOT a git
   repo at all, so there is no commit exposure today and the entry protects a future `git init`.
7. Full suite: **300 passed** (3:07 min), zero network.

## Honest cost answer (task question 3)

One executor call = one single-circuit SamplerV2 job — confirmed empirically.
Per unit with raw+zne+rem: 7 jobs. Shipped config (4 circuits): **28 jobs**,
est. ~85 QPU-s (~14% of the month, upper bound; realistic ~30–60 s at IBM's
~0.3–0.5 ms/shot). Six circuits would be **42 jobs ≈ 127 est. QPU-s — which would
exceed the default 120 s in-process cap** (see finding 2 below). Wall-clock is a
different story: jobs are submitted sequentially (each call blocks on `job.result()`),
so expect open-plan queue wait (minutes–hours) for the first job, then ~28 result
round-trips inside the batch. The estimator estimates QPU seconds, not wall time,
and says so.

## Findings (all minor — none block hardware use)

1. **Batches are never closed by `run_experiment`** (`experiment.py` builds one executor
   per unit at line 566 and never calls `executor.close()`; verified 4/4 batches left
   open). Not a budget leak on the new platform — batch usage is the sum of job quantum
   time, not wall time, and idle batches auto-close on the interactive timeout — but it
   leaves one open batch per unit and defeats the purpose of the `close()` API the
   executor deliberately exposes. Fix: `try/finally` around the technique loop calling
   `close()` when the executor has it.
2. **`hw_first_run.yaml`'s own growth suggestion collides with the default cap**: the
   header suggests "Add seeds [0, 1] (8 circuits, ~2.8 min)". That's 56 jobs ≈ 169 est.
   QPU-s > the 120 s default in-process cap — the run would hard-stop mid-flight after
   ~39 jobs (~118 s already spent) and fill the remaining rows with NaN. Budget-safe
   (the cap working as designed) but a footgun: the comment should say "also raise
   `qpu_seconds_cap` in configs/hardware.yaml to ~180".
3. **Direct-API bypass of the consent flag (by design, documented)**: calling
   `backends.make_executor("ibm_...")` programmatically (outside a config-driven run)
   skips the `hardware_confirmed` gate — it still requires credentials and remains
   bounded by the 120 s in-process cap (verified process-wide across executors).
   The only user-facing documented path (config + `run_experiment`) enforces the flag.
   Acceptable; noting for the record.
4. **Per-unit executor rebuild** re-creates the service + preset pass manager per unit
   (4 auth handshakes for the shipped run). Performance-only; no budget impact.
5. **Cap-exceeded mid-run degrades rather than aborts**: once the cap trips, every
   later technique/unit fails fast into NaN rows and the run "completes". Nothing more
   is submitted (verified), but budget already spent yields a partially-NaN dataset.
   Consider aborting the sweep on the first `HardwareBudgetExceededError` to keep the
   spent budget's data coherent.

## Cross-checks of implementer claims

- Estimator vs reality: exact match (28/28) — claim TRUE.
- Shared Batch: TRUE (per executor/unit; lazily opened).
- Hard-stop BEFORE submission: TRUE (refusal not charged, nothing submitted).
- `hardware_confirmed` must be exactly `True`: TRUE (string rejected).
- 300/300 tests: TRUE.
- CLI refusals exit 2 / exit 3 with cost: TRUE (re-exercised live, local-only).
- hw_first_run ~85 s / fits free plan: TRUE (estimator prints 85 s, 14% of budget).

# Repro-Verifier Findings — 2026-07-21

Verifier ran the README literally in a fresh PowerShell context, checked run-to-run
determinism of the tiny experiment, and audited requirements.txt pins.
All outputs written under `data\repro-check\` (runA, runB, pytest_output.txt) —
no shared dirs (`results\tiny`, `data\small`) were touched.

## Verdict

| Check | Result |
|---|---|
| README works command-by-command | **YES** (all commands exit 0) |
| Determinism (2 identical tiny runs) | **YES — byte-identical results.csv** |
| requirements.txt pins match installed | **YES — all 14 pins exact** |

## 1. README walkthrough (literal, in order)

- `python -m venv .venv` — **skipped** (venv already exists). NOTE: README does
  not say this step can/should be skipped when `.venv` exists; rerunning it is a
  near-no-op, but the README is silent on it (minor doc nit).
- `pip install -r requirements.txt` — exit 0, every requirement already satisfied,
  no resolver conflicts.
- `pip install -e .` — exit 0, rebuilds/reinstalls qemsel 0.1.0 editable. Confirmed
  `pyproject.toml` claim: no runtime deps re-resolved.
- `scripts\verify_env.py` — exit 0, "ALL CHECKS PASSED" (versions, 41 fake
  backends, noisy Bell 0.9395, mitiq ZNE end-to-end, module introspection).
- `python -m pytest -q` — **266 passed**, 3 benign mitiq UserWarnings
  ("input circuit is very short"), 77.6 s, exit 0. Full log:
  `data\repro-check\pytest_output.txt`.
- Quickstart step 1 (`run_experiment.py --config configs\tiny.yaml`) — exit 0.
  Deviation: `--out` redirected to `data\repro-check\runA` / `runB` instead of
  `results\tiny` (per parallel-agent isolation rules; `--out` is a free parameter,
  command otherwise verbatim). 20/20 units, winners rem 11 / cdr 8 / zne 1,
  cost-aware rem 13 / cdr 6 / raw 1 — **exactly matches the post-fix numbers in
  PROJECT_STATE**, i.e. determinism also holds across sessions/agents.
- Quickstart step 2 (`train_model.py --results ... --out ...`) — exit 0. The
  README's `--results` flag works (alias of `--data`; both shown in `--help`).
  Honest cv_folds=0 fallback fired as documented (20 samples, min class = 1).
- Full workflow step 3 (`make_report.py --data ... --metrics ... --out ...`) —
  exit 0; report.md + all 4 PNGs (30–47 KB) produced.
- Full workflow step 4 (`recommend.py --model ... --backend FakeManilaV2 --demo
  ghz_plus`) — exit 0; JSON with technique=rem (p=0.97) + features printed.
- Full workflow step 1 with `configs\experiment.yaml` (80 units, ~11 min) was
  **not re-run** — it is the identical command/flag shape as the tiny run with a
  different config/paths; flags verified against argparse source.
- Python API snippet — runs verbatim (model path pointed at runA): prints `rem`
  and the probabilities dict, exit 0.
- `--help` on all 4 CLI scripts — exit 0 each.
- README output-file claims verified: results.csv, run_meta.json, errors.log
  (8 honest CDR refusals: 4 degenerate near_clifford + 4 fully-Clifford ghz_plus),
  no skipped_low_signal.log (tiny.yaml sets no `min_abs_ideal`) — all as documented.

## 2. Determinism (tiny config, two fresh out dirs)

- `data\repro-check\runA\results.csv` vs `runB\results.csv`:
  **sha256 identical** (`cbdf857d...2cdc3d33`). Cell-level comparison (script:
  scratchpad `repro_verifier_compare.py`): all 32 columns exactly equal,
  including every `*_value`/`*_abs_error` estimate column, `ideal`, both winner
  columns, and NaN placement (cdr NaN on the same 8 rows).
- `errors.log` identical between runs.
- `run_meta.json` differs ONLY in `timestamp` and `out_dir` — expected
  provenance fields, not results.
- Seeding chain that makes this work (verified in source): `seed_simulator` +
  `seed_transpiler` in `backends.make_executor`, seeded `fold_gates_at_random`
  for ZNE, `random_state=seed` for CDR training-circuit generation.
- Runtime note: runA 29 s, runB 50 s (background load) — README's "~4 min" for
  tiny is stale/conservative post-CDR-refusal-fix; harmless direction (over-estimate).

## 3. requirements.txt pin audit

`pip freeze` vs `requirements.txt`: **all 14 pinned lines match exactly**
(cirq-core 1.6.1, joblib 1.5.3, matplotlib 3.11.1, mitiq 1.0.0, numpy 2.2.6,
pandas 2.3.3, ply 3.11, pytest 9.1.1, PyYAML 6.0.3, qiskit 2.5.0,
qiskit-aer 0.17.2, qiskit-ibm-runtime 0.48.0, scikit-learn 1.9.0, scipy 1.17.1).
Spot-check requirement (4 packages) exceeded — all 14 verified.

## 4. Minor issues (none blocking)

1. README tiny-run duration "~4 min" is stale: actual 29–50 s on this machine
   (the CDR fail-loud guard skips the most expensive units). Over-estimate, so harmless.
2. README venv step has no "skip if `.venv` exists" note.
3. `errors.log` is UTF-8; its em-dashes render as mojibake ("â€”") in a default
   Windows PowerShell 5.1 console (`Get-Content` without UTF-8 codepage). File
   content itself is correct; cosmetic only.
4. pytest shows 266 tests, PROJECT_STATE says "263 fast + 3 slow-marked" = 266
   when run unfiltered via the README's plain `pytest -q` — consistent, but the
   README never mentions the slow marker (they only add ~seconds here anyway).

# docs-updater — hardware docs pass 2 (2026-07-21)

Task: bring README.md, PROJECT_STATUS.md, docs/LEARNING_GUIDE.md in line with
the now-real hardware path. Found that a previous pass (referenced in
PROJECT_STATUS §6.11 as "the hardware-docs pass") had already rewritten the
README "Switching to real IBM hardware" section and PROJECT_STATUS §4.2/§5.2.
This pass verified those against the code and filled the remaining gap.

## What was verified (against the actual scripts/source, not memory)

- `scripts/test_hardware_connection.py`: exists; optional `--credentials`
  flag only; free/read-only; exit codes 0/1/2; never prints the token
  (redacts it from library exception text too). README step 5 matches.
- `scripts/estimate_hardware_cost.py`: exists; required `--config` flag;
  pure local arithmetic, no credentials/network; exit 0 fits / 2 error /
  3 does-not-fit. README step 7 command is exact.
- `scripts/run_experiment.py`: flags `--config`, `--out`, `--force-hardware`
  (bypasses ONLY the budget-fit refusal; consent flag + credentials still
  enforced in `experiment._validate_config`). README step 8 and
  PROJECT_STATUS §4.2 match, including "prints the estimated cost on
  refusal" (the ValueError message embeds jobs + QPU-minutes).
- `src/qemsel/hardware.py` constants: FREE_PLAN_MONTHLY_SECONDS=600,
  DEFAULT_QPU_SECONDS_CAP=120, cost model 2 s/job + 1 ms/shot. All numbers
  quoted in the docs (28 jobs, ~85 s, ~1.4 min, ~14% of budget for
  hw_first_run) recompute correctly: 28 x (2 + 1024*0.001) = 84.7 s.
- `.gitignore` line 18 ignores `configs/hardware.yaml` — verified present.
  NOTE: the file now contains the user's real pasted credentials (values
  not echoed anywhere, including this note). §6.13's "add a
  hardware.yaml.example before any git init" flag stands and matters more
  now.

## What was changed

1. **README.md** — appended two sentences after the hw_first_run paragraph:
   queue expectations (fair-share queue, minutes-to-hours wall-clock,
   waiting is free, only QPU execution bills) and non-bit-reproducibility
   (shot noise unseedable; transpiler seed only). Cross-links
   LEARNING_GUIDE §8. Rest of the hardware section left as-is (accurate).
2. **PROJECT_STATUS.md** — no edits needed; §4.2 gate/confirm flow and §5
   step 2 were already correct and code-accurate. Its forward reference
   "LEARNING_GUIDE §8" was dangling — fixed by (3).
3. **docs/LEARNING_GUIDE.md** — new "## 8. Simulator vs real hardware
   (`hardware.py`)" section (old §8 "Where to learn more" renumbered to
   §9): queues (queue time free, QPU time billed, Batch does not skip the
   queue), ISA transpilation (native basis + routing; optimization_level=0
   mandatory so ZNE folds survive), why jobs not shots dominate cost
   (2 s/job overhead is ~2/3 of a 1024-shot job; technique multipliers
   raw 1 / zne 3 / rem 3 / cdr 11 — hence CDR excluded from hw_first_run),
   and the scientific deltas: asymmetric readout (REM first-order bias,
   ties to PROJECT_STATUS §4.8), noise drift framed as train/deploy
   distribution shift (the paper's sim-to-real angle, roadmap step 4),
   unseedable shot noise, crosstalk/non-Markovian effects.

## Not done / out of scope

- No hardware calls attempted (no live verification of connection script
  output format — it is mock-tested per notes/hardware-tester.md).
- README tiny-run "~4 min" staleness (PROJECT_STATUS §6.12) untouched —
  outside the hardware sections this task covers.

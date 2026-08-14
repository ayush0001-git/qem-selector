# QEM-Selector — Project Status

Written 2026-07-21, after the small-run verification pass; updated 2026-07-22
after the **research sweep completed and was independently verified** (see §3 —
new headline numbers; the old small-run snapshot is kept below it as
superseded history). Audience: you (final-year
AI/ML student, new to quantum). Everything below is taken from verified runs and the
agent logs in `notes/` — no aspirational numbers.

---

## 1. What was built

**Benchmark circuits (`src/qemsel/circuits.py`).** Five families of small quantum
circuits (2–5 qubits): `layered_random` (random rotations + CNOT layers),
`near_clifford` (mostly "easy" Clifford gates with a few T/rz gates mixed in),
`ghz_plus` (maximally entangled GHZ states padded with identity gates),
`hw_efficient_ansatz` (the layered rotation+CNOT pattern used in variational
algorithms), and `mirror_circuit` (a circuit followed by its own inverse, so the
correct answer is exactly +1 — a built-in known-answer test). A suite generator
turns a YAML config into a reproducible list of circuits; everything is seeded, so
two runs produce byte-identical circuits.

**Noisy simulation (`src/qemsel/backends.py`, `src/qemsel/ideal.py`).** Circuits run
on Qiskit Aer simulators loaded with noise models from IBM "fake backends" — snapshots
of the calibration data (gate error rates, readout error rates) of real IBM machines.
Four are wired up: FakeManilaV2 (low noise), FakeJakartaV2 (mid), FakeLagosV2
(very high readout error — the "REM should win here" device), FakeSherbrooke (127-qubit
Eagle). The executor transpiles against the real device topology, so two-qubit gates
pick up realistic routing overhead and noise. Any fake backend also accepts a
**noise-scale suffix** — `FakeLagosV2@x1.5` rebuilds its noise model with all
calibrated gate (depolarizing) and readout errors multiplied by 1.5 (caps 0.9/0.45),
same topology and transpilation — turning noise level into a continuous axis
(honest caveats in §4.10). `ideal.py` computes the exact,
noise-free answer with a statevector simulation — that is the ground truth every
error is measured against.

**Error mitigation (`src/qemsel/mitigation.py`).** A uniform wrapper around five
strategies: `raw` (do nothing), `raw_plus` (do nothing at 11x shots — the empirical
equal-budget control, matching CDR's budget), `zne` (Zero-Noise Extrapolation — run
the circuit at deliberately amplified noise and extrapolate back to zero), `cdr`
(Clifford Data Regression — learn a noisy-to-ideal correction map from classically
simulable near-Clifford training circuits), and `rem` (Readout-Error Mitigation —
undo measurement bit-flips using calibration circuits). ZNE/CDR go through the
`mitiq` library; REM is a self-calibrating implementation (mitiq's version needs a
different executor type). Each technique's true shot cost is tracked
(`SHOT_MULTIPLIER`: raw 1x, raw_plus 11x, zne 3x, rem 3x, cdr 11x). Importantly,
CDR *refuses* (raises, logged, recorded as NaN) on circuits where it would
degenerate into free classical simulation — an earlier version silently "won" on
those, which was an artifact, not mitigation — and REM now refuses when a
near-singular readout inversion would amplify shot noise into garbage
(`REM_MIN_DAMPING = 0.02`).

**Feature extraction (`src/qemsel/features.py`).** Ten cheap, static numbers per
(circuit, backend) pair: qubit count, depth, 1q/2q/CNOT gate counts, non-Clifford
gate count, Clifford fraction, depth per qubit, and the backend's average 2q-gate
and readout error rates. These are what the recommender sees — no quantum execution
needed at prediction time.

**Experiment runner (`src/qemsel/experiment.py`).** Sweeps every (circuit, backend)
pair, runs all techniques, and appends one row per pair to `results.csv` with
each technique's error versus ideal, plus two labels: `best_technique` (lowest
error) and `best_technique_cost_aware` (lowest error after a sqrt shot-cost
penalty). It is crash-safe: kill it any time, re-run the same command, and finished
units are skipped (verified by an actual kill test). It also screens out circuits
whose ideal value is too close to 0 (`min_abs_ideal`) — those rows would be
shot-noise lotteries, not signal; since the research upgrade the same threshold
also acts at the *source* (`generate_suite` deterministically re-seeds low-signal
circuits before any noisy work), which keeps the family mix balanced. Every run
additionally writes `aggregated.csv`: seed-AVERAGED errors, features and winner
labels per (family, n_qubits, depth, backend) group, with per-technique
seed-coverage counts and a coverage rule so a technique that failed on most seeds
cannot win a group — this is the lower-label-noise file the headline model trains
on.

**Model + recommender (`src/qemsel/model.py`, `src/qemsel/recommend.py`).** Trains
RandomForest and GradientBoosting classifiers (features -> winning technique),
picked by cross-validated macro-F1 against an always-shown majority-class baseline.
CV is *grouped* by (family, n_qubits, depth) so seed-duplicates of the same circuit
can never sit in both train and test (an earlier ungrouped version inflated accuracy
by ~+0.20 under a null test). Singleton classes no longer collapse CV: a class with
< 2 members is dropped from the CV evaluation (warned + recorded in
`dropped_classes`) while the refit bundle still learns it. Three held-out
generalization metrics are reported: **LOFO** (leave-one-family-out — new circuit
family), **LOBO** (leave-one-backend-out — noise-level *interpolation*, since scale
siblings of the held-out backend stay in training) and **LODO**
(leave-one-device-out, all `@x<scale>` siblings held out together — the honest
"new noise environment" number). `train_model.py --label both` trains both winner
labels (`model.joblib` + `model_cost_aware.joblib`) in one call. The recommender
loads the saved model and answers "which technique for this circuit on this
backend" from a QASM file, a demo family, or Python.

**Real-hardware path (`src/qemsel/hardware.py`).** Backend names starting `ibm_`
(e.g. `ibm_brisbane`) route through `qiskit-ibm-runtime` to real devices on the
new IBM Quantum Platform (quantum.cloud.ibm.com), with the same
`executor(circuit, pauli) -> float` contract as the simulated path (SamplerV2 in
a shared Batch, ISA transpile at optimization_level=0 so ZNE folds survive).
Because the free Open Plan allows only 10 QPU-minutes/month, no job can be
submitted without passing three gates (see §4.2). Fully mock-verified — never
yet run against a live device.

**Report + docs + tests.** `report.py` renders `report.md` with 7 sections — error
tables, cost-normalized view, win rates, a noise-scale sweep (with realized vs
nominal error rates), model metrics incl. LOFO/LOBO/LODO, reproducibility — and up
to 5 plots (`winner_vs_noise.png` appears when the data spans >= 2 noise scales).
`README.md` is command-accurate (independently replayed end-to-end),
`docs/LEARNING_GUIDE.md` explains the quantum concepts per module for an ML
student, and `tests/` holds 424 passing tests including physics regression tests
(e.g. "gate noise must actually fire on this qubit pair", "scaled noise must be
monotonically worse") and 34 mocked hardware-path tests that pass with all network
sockets hard-blocked.

---

## 2. How to run it

All commands from PowerShell. The project path contains **double spaces** — keep the
quotes exactly as shown.

```powershell
cd "E:\quatum  computiiing\qem-selector"

# One-time check that the environment is healthy
& ".\.venv\Scripts\python.exe" scripts\verify_env.py

# Test suite (~3-6 min with slow tests, expect 424 passed)
& ".\.venv\Scripts\python.exe" -m pytest -q

# 1. Benchmark sweep -> results.csv + aggregated.csv  (tiny: ~30-60 s; small: hour-scale; research: ~5 h)
& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\tiny.yaml --out results\tiny

# 2. Train the selector model -> model.joblib + metrics.json
#    (for research-style runs: train on aggregated.csv and add --label both)
& ".\.venv\Scripts\python.exe" scripts\train_model.py --results results\tiny\results.csv --out results\tiny

# 3. Report -> report.md + 4 PNGs
& ".\.venv\Scripts\python.exe" scripts\make_report.py --data results\tiny\results.csv --metrics results\tiny\metrics.json --out results\tiny

# 4. Ask for a recommendation
& ".\.venv\Scripts\python.exe" scripts\recommend.py --model results\tiny\model.joblib --backend FakeManilaV2 --demo ghz_plus
```

Swap `configs\tiny.yaml` / `results\tiny` for `configs\small.yaml` / `results\small`
or — the paper-grade run — `configs\research.yaml` / `results\research` to scale up
(dry-run the research shape first with `configs\research_smoke.yaml`, 45 units,
~2 min; `configs\full.yaml` is the legacy pre-research config). The sweep is
resumable: re-run the identical command after any interruption and it continues
where it stopped.
Fresh setup on a new machine: `python -m venv .venv`, then
`pip install -r requirements.txt` and `pip install -e .` with the venv's python
(exact pins; skip venv creation if `.venv` already exists).

Real-hardware runs (`ibm_*` backends) have their own gated flow — README section
"Switching to real IBM hardware": connection check (free) -> cost estimate
(local) -> `hardware_confirmed: true` -> run.

---

## 3. Current results snapshot (RESEARCH RUN, `results\research\`, 2026-07-22)

From `configs\research.yaml`: 180 circuits (5 families x {2..5} qubits x
{4,8,16} depths, 36 per family) x 9 noise environments ({FakeManilaV2,
FakeJakartaV2, FakeLagosV2} x scales {x1.0, x1.5, x2.0}) x 3 seeds, 4096 base
shots, all 5 techniques = **1620 rows** (`results.csv`) aggregating to **540
seed-averaged groups** (`aggregated.csv`, with feat_* columns). Integrity:
label == argmin(|error|) on 1620/1620 rows; `errors.log` (571 refusals: 415
CDR + 156 REM) exactly matches the NaN pattern. Every number below was
**independently reproduced from scratch to 4 decimals** by a separate verifier
(`notes/final-verifier.md`); analysis tables in `docs/ANALYSIS.md`; plain-
language summary in `END_RESULT.md`.

- **Winners (`best_technique`, per-seed):** cdr 1008 / rem 485 / zne 78 /
  raw_plus 37 / raw 12. Cost-aware: cdr 789 / rem 509 / raw 272 / zne 50.
  CDR's share grows with noise (59.3 -> 61.7 -> 65.7% at x1.0/1.5/2.0, taken
  from REM 33.7 -> 25.4%); among the 1205 rows CDR accepts it wins **83.7%**
  (its 415 structural refusals go mostly to rem, 352).
- **Mean |error| (pooled):** raw 0.4205 / raw_plus 0.4202 / zne 0.3728 /
  rem 0.1963 / cdr 0.0893 (median 0.0249). Reduction factors on own valid
  rows: cdr 4.3x, rem 2.0x, zne 1.1x. Raw grows 0.366 -> 0.475 with the noise
  dial; CDR only 0.084 -> 0.096.
- **Equal-budget control:** raw_plus (11x shots) beats raw on only 49.7% of
  rows (paired diff -0.0003 +/- 0.0138) — raw's error is bias, not shot
  noise; mitigation's wins survive the control. Do-nothing wins 0/1080 rows
  on Manila+Jakarta; all 49 such wins are on Lagos, mostly refusal-menu
  artifacts.
- **ZNE region:** 78 per-seed wins, only 30 beat the full menu; 28/30 at
  depth 8/16 at moderate noise. Worse-than-raw 28.5% at depth 4 vs ~15% at
  depth 8/16; on readout-dominated Lagos mean improvement only +0.006
  (worse-than-raw 41.1%). Direction matches the finite-shot help-harm theory;
  fixed 4096 shots, so this is the sim-side boundary preview only.
- **Headline MODEL (seed-averaged `aggregated.csv`, 540 rows, `--label both`;
  bundles `model.joblib` / `model_cost_aware.joblib`):**
  - `best_technique` (GB): grouped 5-fold CV **0.796 +/- 0.053** (macro-F1
    0.417) vs majority baseline 0.594; **LOFO 0.787** (F1 0.440); LOBO
    (scale interpolation) 0.893 (F1 0.607); **LODO (new device) 0.865**
    (F1 0.357; folds Jakarta 0.967 / Lagos 0.694 / Manila 0.933).
  - Cost-aware (GB): CV **0.728 +/- 0.133** (F1 0.583) vs baseline 0.437;
    LOFO 0.702 (F1 0.513); LOBO 0.783 (F1 0.573); LODO 0.704 (F1 0.428;
    Lagos fold 0.422 is the project's weakest number).
  - dropped_classes = [] in all runs. Quote LODO (not LOBO) for "new noise
    environment" claims.
- **Per-seed label ABLATION (results.csv, 1620 rows):** best CV 0.772 (RF) /
  cost 0.633 (GB) — seed-averaging removes the 20.9%/21.6% seed-flip label
  noise (only 58.0%/52.8% of groups unanimous; flips genuine, median gap
  0.063) and improves every held-out metric (LOFO +0.075/+0.104, LODO
  +0.049/+0.016). 0/540 aggregated winners rest on partial seed coverage.
- **Hardware bridge (n=3, ibm_marrakesh Heron, 2026-07-22):** sims are
  ~9-15x noisier in raw error than real Heron on identical circuits (hw raw
  0.016-0.031 vs sim 0.168-0.479). Winner agreement 2/3 (both mirrors ->
  rem); the miss (layered_random: hw raw wins, REM hurts) is exactly the
  low-noise regime absent from the sim grid. ZNE worse than raw 3/3.
  Motivating preliminary evidence only.
- Figures: `results\research\figs\{winner_share_vs_scale, error_vs_scale,
  zne_win_region}.png` + report plots; report at `results\research\report.md`.
- Mandatory caveats when quoting: Lagos readout cap 0.45 (realized dial
  ~x1.28/x1.44), noise-character change at the x1.0 -> x1.5 step, conditioned
  circuit ensembles (|ideal| >= 0.25), argmin-|error| labels (Decision
  Kernels caveat). See END_RESULT.md §4.

### 3-old. SUPERSEDED — small-run snapshot (`results\small\`, 2026-07-21)

**Everything in this subsection is superseded by the research-run numbers
above — do not quote these in the paper.** Kept as history.

From `configs\small.yaml`: 5 families x {2,3} qubits x {4,8} depths x 3 seeds x
2 backends (Manila = low noise, Lagos = heavy readout error), 4000 base shots.

- **Dataset:** 120 units attempted; 46 (38%) screened out as low-signal
  (|ideal| < 0.25); **74 rows** kept, all unique, label always equals the recomputed
  argmin. Two identical runs of the pipeline produce **byte-identical** results.csv.
- **Winners (`best_technique`):** rem 38 / cdr 35 / zne 1 / raw 0.
  Cost-aware: rem 37 / cdr 32 / raw 4 / zne 1. `raw` never wins on pure accuracy —
  mitigation is genuinely helping on every row.
- **Mean |error| (pooled | Manila | Lagos):** raw 0.423 | 0.209 | 0.637;
  zne 0.391 | 0.158 | 0.623; cdr 0.040 | 0.012 | 0.068 (non-Clifford rows only, and
  at 11x the shots); rem 0.102 | 0.053 | 0.151. This matches the physics: Lagos is
  readout-dominated, so REM shines there and ZNE barely helps.
- **CDR NaN rate: 40.5% (30/74)** — every one an *intentional* refusal (22/24
  ghz_plus rows are fully Clifford; 8/8 near_clifford rows have degenerate training
  sets), each logged in `errors.log`. Zero crashes. This is honest behavior, but it
  means CDR contributes no label signal on 2 of the 5 families.
- **Model — read carefully:** the pipeline's `metrics.json` says accuracy 0.905
  (macro-F1 0.608) vs baseline 0.514, **but this is training-set accuracy**: `zne`
  won only once, a 1-member class makes grouped CV undefined, so the pipeline fell
  back to cv_folds=0 and flags the number as optimistic. An independent check
  (drop the zne row, n=73, 5-fold grouped CV, RandomForest) gives
  **accuracy 0.823 +/- 0.088 vs grouped-majority baseline 0.521** (macro-F1 0.821),
  and **leave-one-family-out 0.808**. With ~74 rows one fold is ~15 rows and fold
  scores ranged 0.67–0.88, so the honest sentence is: *the model is clearly above
  baseline, but 0.82 is a noisy estimate, not a precise one.* Do not quote 0.905.
- **Sanity demos:** ghz_plus on Lagos -> rem; layered_random on Manila -> cdr.
  Both are what the physics says they should be.
- Report: `results\small\report.md` + 4 plots.
- **Post-upgrade check:** re-training on this same 74-row CSV with the upgraded
  `model.py` now reports the honest number directly — CV 0.822 (5-fold grouped,
  zne dropped as a singleton and recorded) vs baseline 0.521, LOFO 0.824,
  LOBO 0.797 — matching the earlier independent hand-check (0.823 +/- 0.088).
  The misleading 0.905 training-set headline can no longer occur.

---

## 3a. Research-run upgrades (landed 2026-07-21; the sweep has since RUN — see §3)

Everything the small run exposed has been fixed in code, verified by tests and a
45-unit end-to-end smoke run (`configs\research_smoke.yaml`, independently
re-computed with plain pandas — see `notes/tester-research.md`,
`notes/fixer.md`). The full sweep has since been run and verified — results in
§3 above; this section stays as the record of what changed and why.

1. **Noise-scaled backends** (`backends.py`): `<Fake*>@x<scale>` names scale all
   calibrated gate/readout errors (caps 0.9/0.45) on identical topology.
   Monotonicity measured (Lagos |raw err| 0.60 @x1.0 < 0.78 @x1.5 < 0.91 @x2.0);
   the plain-name path is proven byte-identical to pre-change behavior.
2. **`raw_plus` equal-budget baseline** (`mitigation.py`): raw at 11x shots
   (= CDR's budget). Smoke: raw_plus 0.5604 vs raw 0.5615 pooled mean |error| —
   the mitigation wins survive an equal-budget control because raw's error is
   bias, not variance. Resolves §6.6.
3. **Seed-averaged labels wired end to end** (`experiment.py` + `model.py`):
   `aggregated.csv` carries seed-mean errors AND features, per-technique seed
   coverage, winners from means with a coverage rule. Trains directly. Per-seed
   winners disagreed with seed-averaged winners on ~29% of smoke rows — that was
   the label noise. Resolves §6.4.
4. **Family-skew fixed at source** (`circuits.py`): `min_abs_ideal` rejection
   sampling inside `generate_suite` — the research grid keeps 180/180 circuits,
   exactly 36 per family (the small run's 24-vs-8 post-screen skew is gone).
   Caveat: this conditions random families on an atypically high-|ideal| subset;
   report §1 discloses it. Resolves §6.3.
5. **Model min-class handling + honest holdouts** (`model.py`): singleton classes
   dropped from CV (recorded, still in refit bundle) — resolves §6.1; LOFO always
   computed; LOBO (interpolation) and LODO (new device) added; `--label both`
   trains both label variants. Report §6 renders all of it side by side.
6. **REM damping floor 0.02** (`mitigation.py`): near-singular readout inversions
   refuse loudly (observed live on Lagos@x1.5). Resolves §6.8.
7. **Report upgrades** (`report.py`): 7 sections; noise-scale sweep tables +
   `winner_vs_noise.png`; realized-vs-nominal error-rate table with the cap and
   noise-character caveats; circuit-selection conditioning disclosure; both-labels
   comparison table.
8. **Configs**: `configs\research.yaml` — 180 circuits x 9 noise environments
   (Manila/Lagos/Jakarta x scales 1.0/1.5/2.0) = 1620 units, 4096 shots, 5
   techniques, ~5 h estimate (FakeSherbrooke dropped: 6-8x cost per unit);
   `configs\research_smoke.yaml` — same shape, 45 units, ~2 min. Legacy paths
   regression-proven: fresh `tiny.yaml` results.csv byte-identical to reference.

---

## 4. Known limitations

1. **All data so far is SIMULATED noise.** The "backends" are Qiskit Aer simulators
   loaded with noise models built from calibration snapshots of real IBM devices
   (FakeManilaV2 etc.). They capture realistic gate/readout error rates and device
   topology, but not drift, crosstalk, or non-Markovian effects of live hardware.
   Every claim in the current results is a claim about simulated noise.
2. **Real hardware is nearly unspent — n=3 circuits only.** *(Update
   2026-07-22: a first n=3 run on ibm_marrakesh exists — see §3 "hardware
   bridge"; ~9.3 free QPU-minutes remain. The gating description below still
   applies.)* The
   hardware path (`src/qemsel/hardware.py`, dispatched from `backends.py` for any
   `ibm_*` backend name) is written and mock-verified (34 tests, proven to pass
   with all network sockets blocked), but it has never touched a live device, so
   every number in §3 remains a claim about simulated noise. Credentials go in
   **`configs\hardware.yaml`** (API key + instance CRN from the new
   quantum.cloud.ibm.com platform; gitignored — never commit or share it). The
   free Open Plan allows **10 QPU-minutes/month**, so submission is triple-gated:
   (a) credentials must load, (b) the experiment config must set
   `hardware_confirmed: true` — until then `run_experiment.py` refuses and prints
   the estimated cost (jobs + QPU-minutes) it would have spent, and (c) an
   in-process cap (default 120 s, `qpu_seconds_cap` in hardware.yaml) hard-stops
   *before* any submission that would exceed it. Configs whose estimate exceeds
   the monthly budget are additionally refused unless `--force-hardware` (which
   still cannot bypass the consent flag). The intended flow is
   `test_hardware_connection.py` (free, submits nothing) ->
   `estimate_hardware_cost.py` (local arithmetic) -> flip the flag -> run.
   README "Switching to real IBM hardware" has the exact steps.
3. **SUPERSEDED — dataset scale.** The research sweep has run: 1620 verified
   rows / 540 seed-averaged groups are now the dataset (§3). The old small-run
   numbers (74 rows, ~+/-0.09 fold noise) are history only.
4. **CDR is structurally absent from parts of the suite** — by honest design (its
   fail-loud guards refuse fully-Clifford and degenerate-training circuits). On the
   research grid the pre-guards pass 1206/1620 units (74%) spanning ALL 5 families
   (layered_random / hw_efficient_ansatz / mirror 36/36 circuits each, ghz_plus
   11/36, near_clifford 15/36), so CDR's label signal is far broader than in the
   small run (where it was 2 of 5 families) — but refused rows still pick their
   winner among the other techniques only.
5. **Only 2 backends in the small run**, so the two backend features are effectively
   a backend ID there. The research config's 9 environments (3 devices x 3 noise
   scales) make them a continuous axis — after that run, quote LODO for "new
   environment" claims, not LOBO (§3a.5).
6. **The cost-aware *label* still uses the analytic sqrt shot-penalty proxy.** The
   empirical equal-budget control now exists as the `raw_plus` technique column
   (§3a.2) — it answers "does mitigation beat equal-budget shot-averaging" — but
   `best_technique_cost_aware` itself is still proxy-penalized, and raw_plus is
   structurally near-unwinnable under that penalty (a comparison column, not a
   reachable class; the report says so).
7. **The low-signal screen skewed the small run's family mix** (24 ghz_plus/mirror
   rows each vs 8 layered_random/near_clifford). Fixed at the source for future
   runs (§3a.4): the research grid keeps 36/36 per family. The small-run numbers in
   §3 still carry the skew.
8. **REM's inversion is exact only for symmetric readout errors** (true for the fake
   backends' stored models; real hardware is asymmetric — first-order bias there).
9. Features are angle-blind: different seeds of the same (family, n, depth) share
   identical feature vectors. Grouped CV handles the leakage, but it also means the
   model cannot distinguish circuits that differ only in rotation angles — which is
   also why the seed-averaged label ("best technique for this *kind* of circuit")
   matches what the features can express.
10. **The noise-scale dial is a controlled approximation, not physics.** Scaled
    variants run a synthetic depolarizing+readout model (plain x1.0 runs the richer
    `from_backend` composite channels — the first scaling step changes noise
    *character*), and caps compress the dial on Lagos (stored q2 readout 46.4% >
    0.45 cap: realized average readout scaling ~x1.28/x1.44 at nominal x1.5/x2.0).
    Report §5 prints realized rates; conclusions should cite those.

---

## 5. Next 5 concrete steps toward the paper

*(Updated 2026-07-22 after the research sweep + verification. The old step 1 —
"run the research sweep" — is **DONE** and verified; results in §3. Full
rationale for the new list in `END_RESULT.md` §3/§5 and
`docs\RESEARCH_ANGLES.md`.)*

1. **Draft the paper skeleton** per `docs\RESEARCH_ANGLES.md`: the
   execution-free selector (F2/F3/F5/F6 in END_RESULT.md) is the contribution;
   Angle 2 (CDR regressor as selectable technique) a sim-only supporting
   section; Angle 3 (learned ZNE-refusal vs analytic help-harm boundary) the
   validation headline. Pull numbers only from `results\research\metrics*.json`
   and `docs\ANALYSIS.md`; obey the DO-NOT-CLAIM list in `docs\LITERATURE.md`
   §3. Quote LODO for "new noise environment", LOFO for "new family", always
   with baselines.
2. **Build the Angle 3 sim overlay (the centerpiece figure):** add a
   shot-budget axis (e.g. 256/1024/4096/16384) to the zne-vs-raw comparison,
   align ZNE to Scavino's fixed-Richardson variant (arXiv:2605.08251), compute
   the analytic dMSE=0 boundary, and overlay the selector's learned ZNE-refusal
   region in the (noise x shots) plane. The current data (fixed 4096 shots) is
   the boundary *preview* — F4's depth/device/scale gradients already point the
   theory's way.
3. **Spend the ~9.3 remaining QPU-minutes on the Angle 3 hardware boundary
   test** (NOT on Angle 2 hardware — see RESEARCH_ANGLES recommendation):
   known-answer mirror/near-Clifford circuits at 2-3 shot budgets on Heron via
   the gated flow (`test_hardware_connection.py` ->
   `estimate_hardware_cost.py` -> `hardware_confirmed: true` -> run), measuring
   the empirical ZNE help-harm crossing to overlay on the sim curve. This also
   patches the selector's known blind spot: real Heron sits in the low-noise
   regime absent from the sim grid (F7 — the 1 miss in 3 was exactly there).
4. **Produce the Angle 2 crossover heatmap (sim only, zero QPU):** CDR
   linear-vs-nonlinear regressor error over (training-set size x non-Clifford
   fraction) across the 5 families, anchored to Korolev's Ridge-usually-wins
   result (arXiv:2606.02697). Mechanics: `CDR_FIT_FUNCTION` /
   `CDR_NUM_FIT_PARAMETERS` in `src/qemsel/mitigation.py` for parametric fits;
   bypass `execute_with_cdr` inside `_apply_cdr` for sklearn regressors.
5. **Re-run the novelty scan at submission time + get a mentor.** The initial
   arXiv literature check is DONE (2026-07-22, `docs\LITERATURE.md`: 24-paper
   table, NOVELTY STATEMENT + DO-NOT-CLAIM list, gap checklist, reading list;
   headline verdict: the learned execution-free selector across QEM technique
   families with LOFO/LODO honesty + sim-to-real transfer test is unclaimed).
   **Re-run the scan right before submission — the three closest papers are
   Apr–Jul 2026** and the field is moving monthly. In parallel, apply to QOSF
   mentorship (qosf.org — open-source, mitiq-adjacent) or find a quantum-info
   supervisor; this repo, with verified negative controls and honest holdouts,
   is the application artifact. Get an experienced eye on the methodology
   *before* the full write-up.

---

## 6. Unresolved items from the logs (specific)

Carried forward from `notes/verifier.md`, `notes/repro-verifier.md`,
`notes/enhancement-applier.md`, the review notes and the research-pass notes
(`notes/fixer.md`, `notes/tester-research.md`). Statuses refreshed after the
research-upgrade pass (§3a); none of the open ones block the research sweep:

1. **zne singleton class -> pipeline CV disabled** — **RESOLVED** (§3a.5):
   `model.py` drops <2-member classes from CV (warned + recorded in
   `dropped_classes`) while the refit bundle keeps them predictable. Verified on
   the real small-run CSV: honest CV 0.822 replaces the old 0.905 training-set
   headline.
2. **CDR refusals carry no explicit feature** (verifier issue 1) — **OPEN,
   deliberate**: a `cdr_refused` indicator changes the frozen FEATURE_NAMES
   interface and every bundle downstream. The model learns "CDR unavailable" only
   implicitly via clifford_fraction. Tracked as a feature-ablation candidate for
   the paper.
3. **Family-mix skew from screening** — **RESOLVED at source** (§3a.4):
   `generate_suite` rejection-samples sub-seeds against `min_abs_ideal`; the
   research grid keeps 180/180 circuits, 36 per family. (The small-run data in §3
   still carries the old skew.)
4. **Seed-averaged labels never implemented** — **RESOLVED** (§3a.3):
   `aggregated.csv` with seed-mean errors + features, coverage counts, coverage
   rule; trains directly through `model.train_and_eval`. Headline pipeline now
   trains on it.
5. **Significance-aware 'tie' labels — consciously not implemented** (enhancement
   decision 1). Still open; revisit only if seed-averaging proves insufficient on
   the research data.
6. **Empirical equal-budget raw baseline** — **RESOLVED** (§3a.2): `raw_plus`
   (raw at 11x shots) runs as a real technique column. Note §4.6: the cost-aware
   *label* still uses the sqrt proxy.
7. **REM affine-offset bias under asymmetric readout** (code review F4, LOW —
   not in the applied-fix list): the hardware path has now landed, so this is no
   longer hypothetical — it matters for interpreting `hw_first_run` REM results;
   the exact affine inversion for single-qubit support is sketched in the review.
8. **REM_MIN_DAMPING too permissive** — **RESOLVED** (§3a.6): floor raised
   1e-6 -> 0.02; near-singular inversions refuse loudly (refusals observed live on
   Lagos variants in the smoke run).
9. **Stale "~27% readout on q0/q1" description of FakeLagosV2** — **RESOLVED**
   everywhere (actual stored values: q0 16.9%, q1 13.6%, **q2 46.4%**, q3 1.7%,
   q4 2.9%; the remaining occurrence in PROJECT_STATE's early entry is historical
   record, corrected by a later entry there).
10. **`recommend.py` with a missing `--model` prints a raw traceback** (tester-A
    M1 / usability finding 4, LOW): message is clear but inconsistent with the
    other CLIs' clean one-line errors.
11. **LEARNING_GUIDE wording nit** (usability finding 5) — **FIXED** in the
    hardware-docs pass: ZNE now reads "standard deviation ~4.4x (variance ~19x)".
12. **README nits** (repro-verifier) — **mostly FIXED** in the research docs pass:
    tiny-run estimate now "~1 min" (actual 29–50 s), venv step carries a "skip if
    `.venv` exists" note. Remaining (cosmetic): `errors.log` em-dashes render as
    mojibake in a default PowerShell 5.1 console (file content is correct UTF-8).
13. **`configs\hardware.yaml` is gitignored but shipped** (builder-docs flag):
    now MORE important — the file holds real credentials. If this becomes a git
    repo, verify the ignore rule fires *before* the first commit and add a
    `hardware.yaml.example` (placeholders only) so the shape isn't lost.
14. **Model-selection winner's curse** (stats review finding 5, LOW): best-of-two
    models chosen on the same CV that produces the headline metric. Acceptable at
    this scale, but the paper must say "we report the better of two models by CV
    macro-F1" explicitly.

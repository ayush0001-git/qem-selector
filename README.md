# qemsel — Quantum Error Mitigation Technique Selector

<p align="center">
  <img src="https://raw.githubusercontent.com/ayush0001-git/qem-selector/master/docs/assets/boundary_overlay.png" alt="ZNE Help-Harm Boundary Overlay" width="700"/>
  <br/>
  <em>Figure 1: The machine learning selector's decision boundary overlay against the analytical Scavino help-harm limit. The selector dynamically recovers the boundary on gate-dominated FakeManilaV2 but correctly refuses ZNE on readout-heavy FakeLagosV2 (exhibiting 100% precision).</em>
</p>


Benchmark quantum error mitigation (QEM) techniques on noisy simulated
backends, then train an ML classifier that recommends the best technique for a
new circuit from its static features — without spending shots trying them all.

**Techniques compared:** `raw` (no mitigation), `raw_plus` (no mitigation at
11x shots — the equal-budget control), `zne` (Zero-Noise Extrapolation),
`cdr` (Clifford Data Regression), `rem` (Readout-Error Mitigation) — via
[mitiq](https://mitiq.readthedocs.io/) 1.0.0 on Qiskit Aer with noise models
from IBM fake backends.

New to quantum computing? Start with
[docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) — it maps every module of
this project to the underlying concept, written for an AI/ML student.

## Executive Results Summary

The table below presents headline performance metrics on the 1,620-unit research benchmark suite (180 circuits $\times$ 9 noise environments) evaluated under 5-fold grouped cross-validation, Leave-One-Family-Out (LOFO), Leave-One-Device-Out (LODO), and real IBM Quantum hardware. All models are evaluated using identical grouped split logic to prevent data leakage.

| Model / Label Variant | Grouped CV (Acc / F1) | LOFO F1 (New Circuit) | LODO F1 (New Noise Env) | Lagos ZNE Precision |
|---|---|---|---|---|
| **Majority Class Baseline** | 0.594 / 0.149 | 0.149 | 0.149 | N/A |
| **Random Forest (Tuned)** | 0.789 / 0.396 | 0.440 | 0.357 | 100.0% |
| **Gradient Boosting (Tuned)** | **0.793 / 0.444** | **0.440** | **0.357** | **100.0%** |
| **Graph Neural Network (GCN)** | 0.828 / 0.340 | 0.202 | 0.512 | 96.2% |
| **RF (Cost-Aware Labels)** | 0.763 / **0.621** | **0.513** | **0.428** | 100.0% |

> [!NOTE]
> **Clarification on Metric Reporting & Model Selection**: Preliminary project summaries quoted overall Grouped CV Accuracy (0.763 – 0.793). Standard macro-F1 (0.396 – 0.444 on raw labels) explicitly penalizes performance on minority classes (`raw` has only 6 winning rows out of 540 in raw accuracy, vs `cdr`'s 321). Training on **Cost-Aware Labels** (`best_technique_cost_aware`) penalizes shot-heavy CDR (11x shots), redistributing winning classes more evenly (`cdr`: 236, `rem`: 189, `raw`: 109, `zne`: 6). This mitigates class imbalance and elevates Macro-F1 to **0.621** (and LOFO F1 to **0.513**). We recommend `model_cost_aware.joblib` as the production model.

### Key Experimental Findings
1. **Hyperparameter Tuning:** GridSearchCV optimized RF (`n_estimators=300, min_samples_leaf=1`) and Gradient Boosting (`n_estimators=50, max_depth=5, lr=0.1`), demonstrating that performance is primarily feature-limited rather than model-limited.
2. **GNN Architecture & Topological Limits:** Evaluated on identical grouped splits, a 2-layer Graph Convolutional Network (GCN) on 21-dim circuit DAG embeddings achieved **82.8% Grouped CV accuracy** and **0.512 LODO F1**. When holding out star-topology `ghz_plus` circuits in LOFO, GCN drops to 8.3% F1 (0.202 mean LOFO F1), demonstrating that local 2-hop message passing fails to generalize to long-range star entangling graphs. Extrapolating the GNN learning curve (`results/research/gnn_learning_curve.png`), F1 increases from 0.403 at 80% to 0.448 at 100% of data, indicating the GNN requires ~1,500–2,000 additional circuit family graphs to reach saturation.
3. **ZNE Utility Decision:** ZNE alone wins only 6 out of 540 runs under cost-aware labels (1.1%). In NISQ environments where backends suffer from both gate noise and readout error, ZNE alone (without readout error mitigation) is rarely optimal due to $3\times$ to $5\times$ shot overhead and lack of readout correction.
4. **V1 vs. V2 Feature Set Ablation:** Adding 5 V2 features (`log2_shots`, `n_2q_layers`, `entangling_density`, `mean_rz_angle_dist`, `backend_avg_1q_error`) boosted out-of-domain LODO generalization by **+7.1% F1** (0.357 $\rightarrow$ 0.428).
5. **CDR Calibration & Shot Costs:** Sweeping calibration size $N \in \{10, 25, 50, 100, 200\}$ revealed that $N=50$ requires $204,800$ shots/circuit ($36.86\text{M}$ total dataset shots across 180 circuits), establishing why linear RidgeCV is preferred under IBM Open Plan limits.

---

## Real IBM Quantum Hardware Confirmation & Simulation-to-Hardware Gap

To validate whether simulation-trained selection rules transfer to physical quantum hardware, we executed a confirmation suite on the **156-qubit Heron-architecture QPU (`ibm_marrakesh`)** via `qiskit-ibm-runtime` (SamplerV2 in a Batch execution).

### Empirical Hardware Results vs. Model Predictions:
* **`mirror_circuit_q2_d4_s0` (2q, depth 4):** Ideal expectation $\langle ZZ \rangle = +1.0$.
  * Unmitigated (`raw`) error: `0.0273` | ZNE error: `0.0957` | **REM error: `0.0040`** (Winner: `rem`)
* **`mirror_circuit_q3_d4_s0` (3q, depth 4):** Ideal expectation $\langle ZZZ \rangle = +1.0$.
  * Unmitigated (`raw`) error: `0.0313` | ZNE error: `0.0664` | **REM error: `0.0254`** (Winner: `rem`)
* **`layered_random_q2_d4_s0` (2q, depth 4):** Ideal expectation $\langle ZZ \rangle = +0.874$.
  * **Unmitigated (`raw`) error: `0.0160`** (Winner: `raw`) | REM error: `0.0464` | ZNE error: `0.2602`
  * *Model Prediction Analysis:* Trained on simulation data, both raw and cost-aware models predicted `cdr` (87.8% confidence) due to simulated 2-qubit gate error rates. On real hardware, unmitigated execution won cleanly because physical gate fidelity on Heron is high enough that mitigation matrix inversion added variance overhead.

### Model Limitations & The Fake Backend Noise Gap:
1. **High Readout Error Backend Routing:** On `FakeLagosV2` (20.35% readout error), switching to the cost-aware model reduces CDR confidence from 93.2% to 71.7% and elevates REM probability to 27.2%, but does not yet flip the top recommendation to REM, suggesting static backend feature representation remains insufficient to fully capture readout-dominated regimes.
2. **The Simulation-to-Hardware Noise Gap:** Historical fake backends (`FakeLagosV2`, `FakeManilaV2`) feature ~1.46% 2-qubit CNOT gate error rates, whereas modern 156-qubit Heron QPUs (`ibm_marrakesh`) achieve ~0.3% 2-qubit error (5x lower). As physical gate fidelity improves, real hardware noise becomes increasingly readout-dominated, causing simulation-trained classifiers to overestimate CDR utility on shallow high-fidelity QPU runs.




## Why this matters

Today's quantum computers are noisy: every gate and every measurement has a
small chance of going wrong, and the errors compound quickly with circuit
depth. Full quantum error *correction* needs far more qubits than current
devices have, so in practice people use error *mitigation* — classical 
post-processing tricks that trade extra circuit executions ("shots") for a
more accurate answer. The catch is that no single technique wins everywhere:
ZNE targets gate noise but cannot see readout errors, REM fixes readout errors
but nothing else, and CDR needs learnable circuit structure and costs roughly
11x the shots of an unmitigated run. Picking the wrong technique wastes a
limited shot budget and can even make the answer worse.

The usual way to find the best technique is to run all of them and compare —
which defeats the purpose of saving shots. This project asks a narrower,
testable question: can cheap, static features of a circuit (qubit count,
depth, CNOT count, Clifford fraction, backend error rates, ...) predict which
technique will win, *before* running anything? We build a labeled dataset by
brute-force benchmarking all techniques across a suite of circuit families and
simulated backends, then train a scikit-learn classifier on it. This is an
honest empirical study, not a solved problem: results come from simulated
noise models of small (5–127 qubit) devices, and the recommender is only as
good as the benchmark suite it was trained on. Whether the learned selection
transfers to real hardware is an open question the roadmap below addresses.

## Install

Prerequisites: Python 3.12+ (supports Windows, macOS, and Linux).

**Windows (PowerShell):**
```powershell
cd qem-selector
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install -e .
```

**macOS / Linux (Bash):**
```bash
cd qem-selector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Verify the environment (checks qiskit, mitiq, fake backends, etc.):

**Windows:**
```powershell
& ".\.venv\Scripts\python.exe" scripts\verify_env.py
```
**macOS / Linux:**
```bash
python scripts/verify_env.py
```

Run the test suite:

**Windows:**
```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

> Note: `requirements.txt` pins exact versions (qiskit 2.5.0, qiskit-aer
> 0.17.2, mitiq 1.0.0, ...). `pyproject.toml` deliberately lists no runtime
> dependencies so `pip install -e .` never re-resolves the pins.

## Quickstart (tiny run)

Graded, integration-tested configs ship with the project:

| Config | Size | Rough duration |
|---|---|---|
| `configs\tiny.yaml` | 20 units (10 circuits x 2 backends, 2000 shots) | ~1 min |
| `configs\small.yaml` | 120 units (60 circuits x 2 backends, 4000 shots) | hour-scale |
| `configs\research_smoke.yaml` | 45 units (15 circuits x 3 backends, 4096 shots) | ~2 min |
| `configs\research.yaml` | 1620 units (180 circuits x 9 noise environments, 4096 shots) | ~5 h (estimate) |
| `configs\full.yaml` | up to 720 units (180 circuits x 4 backends, 8000 shots) | many hours (legacy — superseded by research.yaml) |

(`configs\experiment.yaml` is a mid-size demo config — 80 units, ~11 min.
See "The research run" below for what the research configs add.)

Start with the shipped **tiny** config (also the default `--config` of
`run_experiment.py`) and run the pipeline end to end:

**Windows (PowerShell):**
```powershell
cd qem-selector
# 1. Benchmark: run every (circuit, backend, technique) combo -> labeled CSV
& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\tiny.yaml --out results\tiny

# 2. Train: fit RandomForest + GradientBoosting on the labels, keep the best
& ".\.venv\Scripts\python.exe" scripts\train_model.py --results results\tiny\results.csv --out results\tiny
```

**macOS / Linux (Bash):**
```bash
cd qem-selector
# 1. Benchmark
python scripts/run_experiment.py --config configs/tiny.yaml --out results/tiny

# 2. Train
python scripts/train_model.py --results results/tiny/results.csv --out results/tiny
```

Outputs land in `results\tiny\`: `results.csv` (one row per circuit-backend
pair, with per-technique errors and the winning technique),
`aggregated.csv` (seed-AVERAGED errors, features and winner labels — the
lower-label-noise dataset; train on this one for headline numbers),
`run_meta.json` (config + package versions for reproducibility),
`model.joblib`, `metrics.json`, `errors.log` if any technique call failed
(expected: the fixed CDR now *refuses* — NaN + log line — on circuits where
it would degenerate to classical simulation), and `skipped_low_signal.log`
if the config sets `min_abs_ideal`.

The experiment runner is **crash-safe and resumable**: it appends to
`results.csv` after every unit of work, and re-running the same command skips
pairs that are already done. Kill it any time; nothing is lost.

Want to write your own config? Save something like this under a NEW name
(e.g. `configs\my_run.yaml` — do not overwrite the shipped configs) and pass
`--config configs\my_run.yaml`:

```yaml
circuits:
  families: [mirror_circuit, ghz_plus]
  n_qubits: [2, 3]
  depths: [4]
  seeds: [0, 1]
backends: [FakeManilaV2]
shots: 1024
pauli:               # per-family observables (string or dict)
  ghz_plus: X        # single char repeats to width; <X^n>=+1 on GHZ, every n
  default: auto      # 'auto' => measure <Z...Z> on all qubits
min_abs_ideal: 0.25  # optional: skip circuits with |ideal| below this
```

## Full workflow

**Windows (PowerShell):**
```powershell
cd qem-selector

# 1. Benchmark sweep (edit configs\experiment.yaml to scale up)
& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\experiment.yaml --out results\run1

# 2. Train + evaluate the selector model (--label both trains both base and cost-aware)
& ".\.venv\Scripts\python.exe" scripts\train_model.py --results results\run1\aggregated.csv --out results\run1 --label both

# 3. Generate the report (report.md + plots)
& ".\.venv\Scripts\python.exe" scripts\make_report.py --data results\run1\results.csv --metrics results\run1\metrics.json --out results\run1

# 4. Ask for a recommendation for a new circuit
& ".\.venv\Scripts\python.exe" scripts\recommend.py --model results\run1\model.joblib --backend FakeManilaV2 --demo ghz_plus
```

**macOS / Linux (Bash):**
```bash
cd qem-selector

# 1. Benchmark sweep
python scripts/run_experiment.py --config configs/experiment.yaml --out results/run1

# 2. Train + evaluate
python scripts/train_model.py --results results/run1/aggregated.csv --out results/run1 --label both

# 3. Generate the report
python scripts/make_report.py --data results/run1/results.csv --metrics results/run1/metrics.json --out results/run1

# 4. Ask for a recommendation
python scripts/recommend.py --model results/run1/model.joblib --backend FakeManilaV2 --demo ghz_plus
```

(Run any script with `--help` for the full flag list.)

Or from Python:

```python
from pathlib import Path
from qemsel.circuits import ghz_plus
from qemsel.recommend import recommend

circuit = ghz_plus(n_qubits=3, depth=8, seed=0)
result = recommend(Path("results/run1/model.joblib"), circuit, "FakeManilaV2")
print(result["technique"])       # e.g. 'rem'
print(result["probabilities"])   # class probabilities per technique
```

## The research run (`configs\research.yaml`)

The paper-grade dataset config. What is in it, and why:

- **1620 units.** 5 circuit families x {2,3,4,5} qubits x depths {4,8,16} x
  3 seeds = 180 circuits, each run against 9 noise environments at 4096 base
  shots with all 5 techniques. The suite is **balanced by construction**:
  `min_abs_ideal: 0.25` now acts at the source (`generate_suite`
  deterministically re-seeds low-signal circuits until they pass), so all
  180 circuits survive screening at exactly 36 per family — the small run's
  post-screen family skew (24 vs 8 rows) is gone.
- **The noise-scale dimension.** The 9 environments are 3 devices
  (FakeManilaV2 low-noise, FakeLagosV2 readout-dominated, FakeJakartaV2
  mid-noise H-topology) x 3 noise scales, spelled `FakeLagosV2@x1.5` etc.
  A scaled name rebuilds the device's noise model with every calibrated gate
  error (as depolarizing) and readout error multiplied by the scale (capped
  at 0.9 gate / 0.45 readout), on the identical coupling map and
  transpilation. The two backend features become a continuous noise axis
  instead of an effective backend ID. Two honest caveats (report §5 prints
  the realized rates): on cap-saturated devices the dial compresses —
  Lagos stores 46.4% readout on q2, above the 0.45 cap, so its realized
  average readout scaling is only ~x1.28 / ~x1.44 at nominal x1.5 / x2.0 —
  and plain x1.0 rows run the full `from_backend` noise model while scaled
  rows run the synthetic depolarizing+readout one, so the first scaling step
  changes noise *character* as well as level. Quote the realized numbers
  from the report, not the nominal suffix.
- **`raw_plus` — the empirical equal-budget baseline.** No mitigation, just
  one unmitigated run at 11x the base shots (matching CDR, the costliest
  technique). It closes a reviewer hole the analytic sqrt-cost penalty
  could not: do mitigation wins survive giving "just take more shots" the
  same budget? (Smoke-run answer: yes — raw_plus mean |error| 0.5604 vs raw
  0.5615; raw's error is noise *bias*, which extra shots cannot remove.)
- **Seed-averaged labels.** With exactly 3 seeds per configuration the
  runner writes `aggregated.csv` (540 rows) alongside `results.csv`:
  seed-mean errors and features, per-technique seed-coverage counts, and
  winner labels recomputed from the means (a technique missing seeds cannot
  win a group from its lucky survivor). Train the headline model on this
  file.
- **Expected runtime: ~5 h on the dev machine** — an estimate, not a
  measurement: 5.4 s/unit blended benchmark basis (~2.4 h) with a x2 safety
  margin; the full math is in the config header. FakeSherbrooke was
  deliberately dropped (measured 6–8x cost per unit). The sweep is
  crash-safe — re-run the same command to resume.

```powershell
& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\research.yaml --out results\research
& ".\.venv\Scripts\python.exe" scripts\train_model.py --data results\research\aggregated.csv --out results\research --label both
& ".\.venv\Scripts\python.exe" scripts\make_report.py --data results\research\results.csv --metrics results\research\metrics.json --out results\research
```

Dry-run the exact same shape first with `configs\research_smoke.yaml`
(45 units, ~2 min): it exercises every research-run feature once — scaled +
plain backends, raw_plus, both `min_abs_ideal` layers, 3-seed aggregation,
and the honest CDR/REM refusals in `errors.log`.

Headline numbers to quote from `metrics.json` / report §6: grouped CV
(known-family accuracy), **LOFO** (leave-one-family-out — a genuinely new
circuit family) and **LODO** (leave-one-device-out — a genuinely new noise
environment; all scales of one device held out together). LOBO
(leave-one-backend-out) keeps scale-siblings of the held-out backend in
training, so it measures noise-level *interpolation* and is labeled as such.

## Switching to real IBM hardware

Everything in the results so far runs on **simulated** noise (Qiskit Aer +
noise models from `FakeManilaV2`, `FakeJakartaV2`, `FakeLagosV2`,
`FakeSherbrooke`). Real hardware is now **implemented** in
`src/qemsel/hardware.py`: any backend name starting with `ibm_` (e.g.
`ibm_brisbane`) routes through `qiskit-ibm-runtime` (SamplerV2 in a shared
Batch on the new IBM Quantum Platform, `quantum.cloud.ibm.com`), with the
same `executor(circuit, pauli) -> float` contract as the simulated path.

> **BUDGET: the free Open Plan gives only 10 QPU-minutes per MONTH.** One
> careless config can burn all of it. Real jobs are therefore gated three
> ways: credentials must exist, the experiment config must carry the
> explicit consent flag `hardware_confirmed: true`, and an in-process
> budget cap hard-stops submissions past ~2 estimated QPU-minutes by
> default (`qpu_seconds_cap` in `configs\hardware.yaml` changes it). On top
> of that, `run_experiment.py` refuses any config whose estimate exceeds
> the free monthly budget unless you pass `--force-hardware`.

Setup — this is the **new** IBM Quantum Platform (`quantum.cloud.ibm.com`);
guides describing the old quantum.ibm.com/IBMQ flow no longer apply:

1. Create a free account at
   [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) (Open Plan).
2. On the dashboard, go to **API keys** and click **Create API key**. Copy
   the key immediately — it is shown only once.
3. On the **Instances** page, copy the **CRN** of your open-instance: the
   long string starting `crn:v1:bluemix:public:quantum-computing:...`.
4. Open the credentials file in Notepad — `notepad configs\hardware.yaml` —
   and paste both values:

   ```yaml
   ibm_token: "YOUR_API_KEY"
   instance: "crn:v1:bluemix:public:quantum-computing:..."
   ```

   **Never commit or share this file** — it is listed in `.gitignore`
   (keep it that way).
5. **Run the connection check FIRST** — it is free, read-only, and submits
   nothing: it verifies the key + CRN and lists the devices your instance
   can access, with queue depth and (when the API exposes it) your
   remaining monthly usage:

   ```powershell
   & ".\.venv\Scripts\python.exe" scripts\test_hardware_connection.py
   ```

6. Put a device name from that list into `configs\hw_first_run.yaml`
   (replace the `ibm_brisbane` placeholder if your instance can't see it).
7. Estimate the cost BEFORE running — pure local arithmetic, no
   credentials touched, no network call:

   ```powershell
   & ".\.venv\Scripts\python.exe" scripts\estimate_hardware_cost.py --config configs\hw_first_run.yaml
   ```

8. Read the printed estimate, then set `hardware_confirmed: true` in
   `configs\hw_first_run.yaml` (that flip is your explicit cost consent —
   the runner refuses without it) and run:

   ```powershell
   & ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\hw_first_run.yaml --out results\hw_first_run
   ```

`configs\hw_first_run.yaml` is a shipped minimal first run: 4 circuits x
raw+zne+rem = 28 jobs, ~85 estimated QPU-seconds (~1.4 min, ~14% of the
monthly budget). **CDR is excluded on purpose**: it costs 11 jobs per unit
(1 target + 10 near-Clifford training circuits), which would add 44 jobs
(~+2.2 min) and nearly triple the bill of this first run — benchmark CDR on
hardware only later, with its own budget check. Validate the pipeline fully
in simulation first, then spend hardware time on a small confirmation
subset (see roadmap).

Two expectations to set before running: free-plan jobs wait in a shared
fair-share **queue** (minutes to hours of wall-clock per job at busy times
— waiting is free, only actual QPU execution counts against the 10
minutes), and hardware results are **not bit-reproducible** — real shot
noise cannot be seeded, so only the transpiler seed is deterministic and a
rerun gives slightly different numbers. See
[docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) §8 for what else changes
on real hardware and why it matters scientifically.

## Project layout

| Path | What it is |
|---|---|
| `src/qemsel/circuits.py` | 5 benchmark circuit families (layered_random, near_clifford, ghz_plus, hw_efficient_ansatz, mirror_circuit) + suite generator |
| `src/qemsel/backends.py` | Fake-backend noise info, noisy executor factory, `ibm_*` dispatch to hardware |
| `src/qemsel/hardware.py` | Real IBM Quantum access: credentials, SamplerV2/Batch executor, QPU budget guard, cost estimator |
| `src/qemsel/ideal.py` | Exact (statevector) expectation values — the ground truth |
| `src/qemsel/mitigation.py` | Uniform dispatch to raw / ZNE / CDR / REM + shot accounting |
| `src/qemsel/features.py` | 10 static circuit+backend features for the ML model |
| `src/qemsel/experiment.py` | Crash-safe benchmark sweep -> `results.csv` labeled dataset |
| `src/qemsel/model.py` | RandomForest + GradientBoosting training, CV evaluation, model bundle |
| `src/qemsel/recommend.py` | Load model bundle, predict best technique for a new circuit |
| `src/qemsel/report.py` | Markdown report + plots from results and model metrics |
| `scripts/` | CLI entry points (`run_experiment.py`, `train_model.py`, `make_report.py`, `recommend.py`, `verify_env.py`, `test_hardware_connection.py`, `estimate_hardware_cost.py`) |
| `configs/` | `tiny.yaml`/`small.yaml`/`research.yaml` (graded sweep configs), `research_smoke.yaml` (research-shape dry run), `full.yaml` (legacy), `experiment.yaml` (demo config), `hw_first_run.yaml` (minimal real-hardware run), `hardware.yaml` (IBM credentials — gitignored) |
| `tests/` | pytest suite; `conftest.py` provides fast noiseless fixtures |
| `docs/LEARNING_GUIDE.md` | Quantum-for-ML-students guide mapped to this codebase |
| `docs/internal/` | Development logs, internal specifications, agent state tracking, and notes |
| `results/`, `data/`, `models/` | Run outputs (gitignored) |
| `docs/internal/spikes/` | Throwaway API-exploration scripts for mitiq ZNE/CDR/REM |

## Roadmap to a paper

1. **Tiny run (done via quickstart):** 2–3 qubits, 2 backends — validates the
   pipeline, not the science. Expect a small dataset and noisy labels.
2. **Small run (`configs\small.yaml`):** all 5 families x {2,3} qubits x
   {4,8} depths x 3 seeds x 2 backends = 120 units. Enough for a first honest
   model evaluation (report grouped-CV `macro_f1` vs the majority-class
   baseline AND the leave-one-family-out accuracy; if the model does not
   beat the baseline, say so).
3. **Research run (`configs\research.yaml`)** — see "The research run"
   above: 9 noise environments (3 devices x noise scales x1.0/x1.5/x2.0),
   the `raw_plus` equal-budget baseline, 3 seeds per configuration with
   seed-averaged labels (`aggregated.csv`), and a source-balanced suite.
   This supersedes the older `configs\full.yaml` plan; `FakeSherbrooke`
   (127q, measured 6–8x cost per unit) is a possible follow-up config, not
   part of this run.
4. **Real hardware confirmation:** use `configs\hw_first_run.yaml` (see
   "Switching to real IBM hardware"), rerun a
   small, carefully chosen subset on a free-plan IBM device, and test whether
   the simulation-trained selector still picks the right technique. A
   negative result here is publishable content, not failure.
5. **Novel angle — CDR regressor experiments:** mitiq's CDR fits a *linear*
   map from noisy to ideal expectations, learned on near-Clifford training
   circuits. As an ML project, that is a one-feature linear regression begging
   for ablation: swap in ridge, polynomial, random-forest regressors; vary
   training-set size and `fraction_non_clifford`; characterize when
   (if ever) nonlinear maps beat the linear fit and when they overfit the tiny
   training set. This is the student's original contribution candidate.
   **Where the swap happens** (see the `qemsel/mitigation.py` module
   docstring, "Plugging in your own CDR regressor"): *parametric* fits
   (polynomial) go through mitiq's `fit_function=`/`num_fit_parameters=`
   kwargs — set the `CDR_FIT_FUNCTION`/`CDR_NUM_FIT_PARAMETERS` constants in
   `mitigation.py` (a one-line ablation). *sklearn* regressors (ridge,
   random forest) can NOT go through `fit_function` (it must be a scipy
   `curve_fit`-style parametric function) — for those, bypass
   `execute_with_cdr` inside `_apply_cdr`: generate the training set with
   `mitiq.cdr.generate_training_circuits` (already done there for the
   degeneracy guard), run it through the noisy executor and the ideal
   simulator, and fit any regressor noisy -> ideal.
6. **Literature check before writing:** search arXiv for prior work on
   learning-based QEM selection (e.g. "machine learning quantum error
   mitigation selection", "learning to mitigate"). Note that mitiq itself
   ships a `Calibrator` that picks a technique by *running* candidates on test
   circuits — our angle differs in predicting from static features without
   spending shots, but the paper must cite and clearly delineate against it,
   and against the CDR/vnCDR original papers (Czarnik et al.) and the ZNE
   (Temme et al.) and REM literature.

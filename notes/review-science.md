# Scientific review (review-science agent) — 2026-07-21

Scope: research validity for a future paper. Methods: read all of src/, scripts/,
tests/, configs/, README, PROJECT_STATE, notes; verified every suspicion by
running code with the project venv (scripts in my session scratchpad, results
reproduced below). Only confirmed findings are listed.

**Bottom line: a Quantum-journal reviewer would reject the current methodology,
for three concrete reasons (findings 1-3). All are fixable before the small/full
runs. Nothing below invalidates the pipeline engineering, which is solid
(crash-safe resume, seeding, honest cv_folds=0 flag all check out).**

---

## Finding 1 (HIGH) — CDR degenerates to classical simulation on 2 of the 5 families; the "CDR wins 80%" headline is an artifact

File: `src/qemsel/mitigation.py` (`_apply_cdr`), interacts with
`src/qemsel/circuits.py` (`near_clifford`, `ghz_plus`).

Two mechanisms, both verified against installed mitiq 1.0.0 source
(`mitiq/cdr/cdr.py` line ~130) and by running the project generators:

1. **Fully-Clifford short-circuit.** `execute_with_cdr` starts with
   `if is_clifford(circuit): return simulator.evaluate(circuit)[0].real` — it
   returns OUR ideal statevector simulator's value directly. Zero error by
   construction, no quantum execution of the target at all. Verified: ghz_plus
   q2_d4 seeds 0/1 and ghz_plus q4_d8 seeds 0/1 compile to fully-Clifford
   circuits (rz-pair padding is only chosen sometimes).
2. **Degenerate constant fit.** When the circuit is not fully Clifford but every
   near-Clifford training circuit has the SAME ideal value (stabilizer-state
   expectation values of Z...Z concentrate on {-1, 0, +1}), the linear
   `curve_fit` collapses to the constant y = that value, ignoring the noisy
   measurement of the target entirely. CDR then returns the classically
   simulated value with ~1e-16 error. Verified in results/tiny/results.csv:
   all 4 near_clifford rows and all 4 ghz_plus rows have
   `cdr_abs_error <= 6.7e-16`.

**This does NOT fade at scale** (contrary to the hope in notes/tester-b-r1.md
§5c). I generated every near_clifford and ghz_plus circuit at small/full sizes
(n in {3,4,5}, d in {8,16}, seeds 0-2, 36 circuits) and computed the training
sets mitiq would build (same settings: 10 circuits, fraction 0.2, closest
replacement): **2/36 short-circuit and 34/36 have all-identical training
ideals** — 36/36 degenerate. That is 2 of 5 families = 40% of every planned
dataset row where "cdr" as winner means "classical simulation is exact", not
"CDR mitigates well". Consequences for the paper:

- Win-rate tables and the CDR mean-error column are dominated by an artifact.
- The classifier's learned rule (high clifford_fraction -> cdr) is the artifact,
  not a QEM-selection insight.

**Fix (fail loudly, per the architect's own philosophy):** in `_apply_cdr`,
after compiling, (a) raise `MitigationError("cdr", "circuit is fully Clifford —
CDR would return the classical simulator value")` when
`mitiq.cdr.clifford_utils.is_clifford(compiled)` is True, and (b) generate the
training circuits yourself (`mitiq.cdr.generate_training_circuits`, same seed
and settings), compute their ideal values, and raise when
`np.ptp(ideals) < tol` (degenerate regression). run_experiment already handles
MitigationError -> NaN + errors.log, so the dataset then honestly records "CDR
not applicable" instead of a fake win. Alternatively/additionally: redesign the
observable per family so Clifford proxies have varied ideals (see finding 3).
Minimum for any already-collected data: flag and exclude rows with
`cdr_abs_error < 1e-12` from all aggregate stats and from training.

## Finding 2 (HIGH) — 2-qubit gates on non-coupled pairs run with ZERO noise on FakeJakartaV2/FakeLagosV2; full.yaml (n_qubits 4-5) would silently under-noise

File: `src/qemsel/backends.py` (`make_executor`), bites `configs/full.yaml`.

`make_executor` transpiles against the bare `AerSimulator(noise_model=...)`,
which has **no coupling map**, with `optimization_level=0` (no routing). All
circuit families emit cx only on adjacent pairs (q, q+1) of the *logical* line.
`NoiseModel.from_backend` defines cx errors only for physically coupled pairs.
Lagos/Jakarta are 7-qubit H-topology devices whose edges among qubits 0-4 are
only (0,1),(1,2),(1,3) — **there is no (2,3) or (3,4) edge**, so those cx gates
have no noise entry and execute noiselessly. Verified empirically via the
project's own executor (|11> input so relaxation noise is visible; 50k shots):

```
Lagos  pair (0,1): <ZZ> drops 0.4816 -> 0.2142 after 100 cx   (noise fires)
Lagos  pair (2,3): <ZZ> 0.0776 -> 0.0776  delta exactly 0     (NO noise)
Lagos  pair (3,4): <ZZ> 0.9098 -> 0.9098  delta exactly 0     (NO noise)
Jakarta pair (0,1): delta -0.4713;  pairs (2,3),(3,4): delta exactly 0
```

Manila (5q linear chain) and Sherbrooke (cx is rewritten to ecr, and the
heavy-hex path 0-1-2-3-4 is connected — verified 100x-cx identity decays
1.0 -> 0.907) are fine. Current tiny/small runs (n <= 3) are unaffected; the
planned **full run is not**: at n=5 on Lagos/Jakarta, 2 of 4 GHZ-chain cx and
half of the brick-pattern cx are noiseless. The dataset would then say "deep
circuit on noisy backend" (features claim avg_2q_error ~1.5%) while the
simulation applied no 2q noise on those pairs — feature-label relationships
break, and ZNE/CDR results on those rows are physically meaningless.

**Fix:** route to the device: transpile against the fake backend itself (it
carries the coupling map) before handing the circuit to the executor's Aer run,
or relabel circuit qubits onto a connected path per backend (Lagos/Jakarta have
the 5-vertex path 2-1-3-5-4; 0-1-3-5-4 also works), or cap `n_qubits` at 3 for
these two backends in full.yaml. Add a regression test: for every backend and
every qubit pair the suite will use, a 100x cx identity chain from |11> must
change <ZZ> vs the 0-gate reference.

## Finding 3 (HIGH) — 42% of the planned full-run circuits have |ideal| < 0.1 under the fixed Z...Z observable; their winner labels are shot-noise lotteries

Files: `configs/full.yaml` / `configs/small.yaml`, `src/qemsel/circuits.py`,
`src/qemsel/experiment.py` (pauli: auto -> 'Z'*n for every family).

Computed exactly over the 180 circuits of full.yaml:

```
family                |ideal|<0.1   notes
layered_random         16/36
near_clifford          31/36        ideals concentrate on {0, +-1}; mostly 0
ghz_plus               18/36        ALL odd-n rows: <Z^n> = 0 exactly on GHZ
hw_efficient_ansatz    10/36
mirror_circuit          0/36        (known-answer +1, good)
total                  75/180 = 42%
```

Depolarizing-type noise biases measured expectations toward 0. When ideal ~ 0,
raw is already ~0-error, every technique's |error| is within shot noise
(sigma ~ 1/sqrt(shots) ~ 0.011-0.016 at 4-8k shots), and `best_technique` is
decided by which technique's noise fluctuation landed closest — i.e., ~40% of
the training labels would be noise. The project's own spike notes
(notes/spike-cdr.md gotcha 3) already warned: "pick a circuit whose ideal
expectation is well away from 0". The suite generator never enforces this.
This also feeds finding 1: odd-n ghz_plus rows get ideal=0 AND a degenerate
CDR that predicts ~0 -> CDR "wins" the lottery rows too.

**Fix (any of, ideally all):**
- Per-family observables with O(1) signal: for GHZ use <X^n> (=1 for every n,
  odd or even); mirror already uses <Z^n>=+1. Requires allowing per-family
  pauli in the config schema (executor + ideal.py already support X/Y).
- Reject/regenerate circuits with |ideal| < threshold (e.g. 0.25) at
  suite-generation time, and say so in the paper.
- Significance-aware labels: a technique is the winner only if it beats the
  runner-up by > k*sigma_shot (propagated per technique); otherwise emit a
  'tie' label or drop the row. PROJECT_STATE's "seed-averaging" next-step helps
  but does not fix rows where the ideal itself carries no signal.

## Finding 4 (MEDIUM) — CV protocol overstates generalization: near-duplicate feature vectors leak across folds

File: `src/qemsel/model.py` (`train_and_eval`).

Features are angle-blind: different seeds of the same (family, n, depth) produce
byte-identical feature vectors (verified in results/tiny/results.csv: e.g. both
layered_random seeds share the vector; all near_clifford rows share one), and
the same circuit appears once per backend differing only in the 2 backend
features. `StratifiedKFold(shuffle=True)` therefore routinely puts a test row's
exact feature vector in the training fold with (majority of) its label. The
reported accuracy then measures within-suite label consistency, not the paper's
actual claim ("recommend for a NEW circuit"). **Fix:** group-aware CV —
`GroupKFold` with groups = (family, n_qubits, depth) — and report
leave-one-family-out as the headline generalization number (it is the honest
proxy for "new circuit"); keep the random-CV number as a secondary metric.

## Finding 5 (MEDIUM) — cost-aware comparison exists but is not the one analyzed; two inconsistent cost models; no empirical equal-budget baseline

Files: `src/qemsel/experiment.py` (`_pick_winners`, sqrt model),
`src/qemsel/report.py` (`_section_cost_normalized`, linear model),
`src/qemsel/model.py` (trains on the accuracy label only).

- The fair per-row winner `best_technique_cost_aware`
  (`abs_error * sqrt(shots/base)`) is computed and stored, but **nothing
  downstream uses it**: report.py never references the column (no cost-aware
  win-rate table; §4 and both win figures are accuracy-only), and model.py
  trains only on `best_technique`. The recommender the paper ships therefore
  optimizes accuracy-at-any-shot-cost, while the README sells shot savings.
- Report §3 uses a DIFFERENT cost model (mean error x linear shots ratio: CDR
  penalized 11x) than the CSV column (sqrt: ~3.3x) — and §3's own footnote
  states the sqrt scaling is the right first-order rationale. A reviewer will
  ask which model the paper stands behind.
- Both are proxies applied to errors that the data shows are bias-dominated
  (raw mean error 0.24 >> shot noise 0.02), where extra shots buy ~nothing, so
  the sqrt penalty over-charges mitigation; the linear one more so. The honest
  comparison is empirical: also run `raw` at `max(SHOT_MULTIPLIER)*base_shots`
  (one extra executor call per unit, cheap) and let "just take more shots"
  compete as its own technique row.

**Fix:** (a) add cost-aware win-rate and per-family winner tables to report §3
using the existing column; (b) train and report a second model on the
cost-aware label (or a config switch `label: accuracy|cost_aware`); (c) unify
on the sqrt model or, better, add the empirical `raw_boosted` baseline and
retire the analytic penalty to a robustness appendix.

## Finding 6 (LOW) — winner class balance not persisted; recommender can never say "raw"

Files: `src/qemsel/model.py`, `src/qemsel/report.py`.

Class balance is printed to stdout only; `metrics.json` has no `class_balance`
key (it is derivable from confusion-matrix row sums, and report §4's win table
is a proxy, but the paper needs it explicit next to the model metrics).
Related: `raw` never wins the accuracy label (0/20 tiny; likely near-0 at
scale), so the trained classifier is structurally unable to recommend "no
mitigation" — an option the cost-aware label does sometimes choose (1/20 tiny).
**Fix:** add `class_balance` (dict label->count) to the `train_and_eval`
metrics dict and render it in report §5 with a sentence on missing classes;
the cost-aware-label model of finding 5 restores `raw` as a reachable class.

---

## Answers to the reviewer-brief questions

- **Is the winner metric fair given ZNE/CDR consume more shots?** The fair
  per-row metric is implemented (`best_technique_cost_aware`, sqrt penalty) but
  is NOT what the report's win rates, figures, or the trained model use; the
  report's §3 uses a different (linear) penalty. See finding 5.
- **Are circuit families diverse enough to generalize?** Five families is a
  reasonable start, but two of them (near_clifford, ghz_plus) are degenerate
  under the chosen observable (findings 1, 3), features cannot distinguish
  seeds (finding 4), and all families are brick/chain circuits on <= 5 qubits
  with <Z...Z> only. Generalization claims need leave-one-family-out numbers.
- **Ideal values correct for every family?** Yes — verified conventions
  (pauli reversal, endianness) and the mirror/GHZ known answers; the 1e-10
  integer snap is sound. The problem is not correctness but that 42% of ideals
  are ~0 and carry no benchmark signal (finding 3).
- **Is CDR's near-Clifford compilation appropriate / does it fail loudly?**
  Compilation to {rz,sx,x,cx} is correct, but CDR silently returns
  classical-simulation values on 40% of the suite (finding 1) and silently
  mis-corrects when the regression is degenerate or the training set is far
  from the target (observed 28x worse than raw on layered_random_q2_d4_s1 @
  Manila). It must fail loudly in the degenerate cases.
- **Winner class balance reported?** stdout only; not in metrics.json/report
  (finding 6).
- **Would a Quantum reviewer reject?** Yes, as-is: (i) headline technique
  ranking driven by the CDR classical-simulation artifact; (ii) planned full
  run silently under-noises 2q gates on half the backends; (iii) ~42% of
  labels are shot-noise lotteries; (iv) CV protocol leaks near-duplicates, so
  the ML claim is unsupported; (v) the cost-fairness analysis the abstract
  needs exists in the CSV but not in the analysis. All five are fixable with
  modest changes before the small/full runs.

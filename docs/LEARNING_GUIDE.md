# Learning Guide: Quantum Error Mitigation for an AI/ML Student

You know machine learning. You do not (yet) know quantum computing. This
guide gives you just enough quantum to understand every module in this
project, in plain English, with each concept mapped to the file that uses it.
Read it top to bottom once; it is short on purpose.

## 1. Qubits, gates, circuits (`circuits.py`)

A classical bit is 0 or 1. A **qubit** is described by two complex numbers
(amplitudes), one for 0 and one for 1; squaring their magnitudes gives the
probability of reading 0 or 1 when you measure. Until you measure, the qubit
genuinely occupies a weighted combination of both — that is superposition. With
n qubits you need 2^n amplitudes, one per bitstring, which is why simulating
even 30 qubits exactly strains a laptop, and why quantum hardware is
interesting at all.

A **gate** is a small matrix applied to those amplitudes — always reversible,
always length-preserving (unitary). One-qubit gates rotate a single qubit's
amplitudes (`h`, `rx`, `rz`, ...); two-qubit gates like CNOT (`cx`) make two
qubits correlated in a way no classical distribution can fully imitate —
**entanglement**. A **circuit** is just a sequence of gates on a register of
qubits: think of it as a tiny straight-line program, drawn left to right. Its
**depth** is the number of sequential layers — the circuit's "runtime", and
the main thing noise punishes.

`circuits.py` generates five families of benchmark circuits: random ones
(`layered_random`), mostly-Clifford ones (`near_clifford` — see CDR below),
GHZ states (`ghz_plus` — maximal entanglement, a standard stress test),
variational-ansatz style (`hw_efficient_ansatz` — the shape used in quantum
ML/chemistry), and `mirror_circuit` (a circuit followed by its own inverse, so
the right answer is known to be exactly +1 — a free known-answer test at any
size).

## 2. Expectation values — what we actually measure (`ideal.py`)

Running a quantum circuit once gives you one random bitstring, e.g. `011`.
Run it 4096 times ("shots") and you get a histogram over bitstrings. Almost
every quantum algorithm reports its answer as an **expectation value**: a
weighted average over that histogram, between -1 and +1.

The observable used here is a **Pauli string** like `"ZZZ"`: for each shot,
look at the measured bits and score +1 if an even number of them are 1, and -1
if odd (parity); average over shots. That average estimates ⟨Z...Z⟩. Why this
instead of the raw histogram? Because it is a single scalar with a known ideal
value, so "how wrong is the noisy device" becomes a simple number:
`abs_error = |measured - ideal|`. That number is this project's entire target
metric.

`ideal.py` computes the exact, noise-free expectation by simulating the full
state (statevector). It is the ground truth every technique is scored
against, and it is only possible because our circuits are small.

One project convention worth knowing: our pauli strings index qubit 0 at the
**left**, but qiskit's own labels put qubit 0 at the right, and its bitstrings
too (little-endian). The code converts internally; just don't be surprised
when you see `pauli[::-1]` or `bitstring[::-1]`.

## 3. What noise does (`backends.py`)

Real devices are imperfect in three main ways:

- **Gate errors** — every gate slightly misfires (~0.1% for 1-qubit gates,
  ~1% for 2-qubit gates). Errors compound with depth: deep circuit + many
  CNOTs = expectation values decaying toward 0.
- **Readout errors** — the final measurement itself lies: a qubit that is
  really 1 is sometimes read as 0 and vice versa (typically a few percent per
  qubit — but on one of our backends some qubits exceed 25%, and one is
  near-useless at ~46%).
- **Decoherence** — qubits gradually forget their state while idling.

`backends.py` builds simulators loaded with **noise models copied from real
IBM devices** ("fake backends": FakeManilaV2, FakeJakartaV2, FakeLagosV2,
FakeSherbrooke). Each has different error rates — FakeLagosV2 has terrible
readout on some qubits, which is exactly the kind of variation that makes
technique selection non-trivial. `make_executor` wraps a noisy simulator into
a function `(circuit, pauli) -> noisy expectation value`; everything
downstream just calls that function.

**Noise scaling (`@x<scale>` backend names).** A name like
`FakeLagosV2@x1.5` means "the same device, but every calibrated error rate
multiplied by 1.5": each gate error becomes a depolarizing channel at 1.5x
the calibrated rate, each readout flip probability is 1.5x too (capped at
0.9 for gates, 0.45 for readout), while the qubit layout and transpilation
stay identical. The point is to turn "which backend" from a categorical ID
into a **continuous noise-level axis** the model can learn a real
relationship from. The honest caveat: this is a controlled dial, not a
prediction of physics. The scaled model is a simplified
depolarizing+readout rebuild — the unscaled x1.0 model has richer channel
structure (e.g. thermal relaxation), so the first scaling step changes the
*character* of the noise, not just its strength — and on a qubit already at
a cap the dial compresses (Lagos stores 46.4% readout on q2, above the 45%
cap, so nominal x1.5 realizes only ~x1.28 average readout scaling there).
The report prints the *realized* per-backend error rates; quote those, not
the nominal suffix.

## 4. Error mitigation — three techniques, two baselines (`mitigation.py`)

Error *correction* (redundantly encoding qubits) needs hardware nobody has at
scale. Error *mitigation* instead runs extra or modified circuits and
post-processes the results classically. You pay shots to buy accuracy.

### ZNE — Zero-Noise Extrapolation (shot cost ~3x)

You cannot reduce the noise, but you can *increase* it in a controlled way:
replace a gate G with G G† G (do it, undo it, redo it) — logically identical,
physically ~3x the gate noise. Run the circuit at noise scales 1x, 2x, 3x, fit
a curve through the three expectation values, and extrapolate back to what the
value would be at zero noise. That is the whole idea: regression on noise
level, evaluated at x=0. It targets **gate noise** only — it can do nothing
about readout errors — and extrapolation amplifies shot noise (the fit
coefficients for 3 points are [3, -3, 1], so the standard deviation blows up
~4.4x — variance ~19x).

### CDR — Clifford Data Regression (shot cost ~11x) — the ML one

You will recognize this immediately: it is **supervised regression with
cleverly generated training data**.

The problem: for the circuit you care about, you have the noisy output but no
label (the ideal value is what you're trying to find). The trick: there is a
special class of circuits — **Clifford circuits** (built from gates like H, S,
CNOT) — that a classical computer can simulate efficiently even at large qubit
counts (Gottesman–Knill theorem). So CDR takes your circuit, replaces most of
its non-Clifford gates with nearby Clifford ones to make ~10 "training
circuits" that are structurally similar to yours, and for each one obtains
BOTH the noisy value (run it on the device) and the exact value (simulate it
classically — free labels!). Then it fits a linear regression
`ideal ≈ a * noisy + b`, and applies that learned map to your circuit's noisy
value. One feature, ten training points, linear model — that's stock mitiq
CDR. It works when the circuit is *near*-Clifford (so the training circuits
resemble the real one) and its ~11x shot cost (1 real + 10 training circuits)
is the price of the training set. Swapping that linear fit for other
regressors is this project's planned original experiment.

### REM — Readout-Error Mitigation (shot cost ~3x)

Readout error is just **label noise with a known confusion matrix**. You can
measure each qubit's flip probabilities — P(read 1 | truly 0) and
P(read 0 | truly 1) — directly from calibration data. Build the confusion
matrix A that maps true bitstring probabilities to observed ones
(`p_observed = A p_true`), then apply the (pseudo-)inverse:
`p_true ≈ A⁻¹ p_observed`. If you have ever corrected a noisy-label dataset
with a known noise matrix, this is literally that. It fixes **only** readout
error — gate noise passes straight through — but where readout dominates it
is spectacularly effective (in our spike on FakeLagosV2: error 0.60 → 0.04).
Caveat: a qubit with near-50% flip probability makes A nearly singular and the
inversion amplifies noise into garbage — same pathology as inverting an
ill-conditioned matrix anywhere else.

### raw

No mitigation. It is a real contender, not a strawman: with a fixed total
shot budget, raw uses all shots on the actual circuit while the others split
shots across extra circuits. On a quiet backend with a shallow circuit, raw
can win the cost-normalized comparison.

### raw_plus — the equal-budget control (shot cost 11x)

Also no mitigation — just one unmitigated run at 11x the shots (the same
budget as CDR, the costliest technique). Why does the benchmark need it?
Fairness: every mitigation technique spends *extra shots*, so "CDR beat raw"
could in principle just mean "an 11x budget beat a 1x budget". raw_plus is
the empirical control that closes that hole: if a technique cannot beat
plain averaging at the *same total budget*, it is not actually mitigating
anything. The expected — and measured — result is that raw_plus barely
improves on raw: the noisy expectation converges to a **biased** value, and
more shots only shrink the statistical error bar around the wrong number.
They do nothing to the bias itself, which is precisely the error mitigation
targets. Running the control and showing that beats asserting it.

## 5. The features (`features.py`)

For each (circuit, backend) pair we extract 10 cheap, static numbers — no
quantum execution needed:

| Feature | Why it should predict the winner |
|---|---|
| `n_qubits` | More qubits = more readout errors to accumulate (helps REM matter) |
| `depth` | Deeper = more gate noise (ZNE/CDR territory), weaker signal overall |
| `n_1q_gates`, `n_2q_gates` | Volume of noise sources; 2q gates are ~10x noisier |
| `n_cnot` | The dominant error contributor on real devices |
| `n_non_clifford` | Many non-Clifford gates = CDR's training circuits resemble the real circuit less |
| `clifford_fraction` | High = classically-simulable structure = CDR's sweet spot |
| `depth_per_qubit` | Shape: deep-and-narrow vs shallow-and-wide noise profiles |
| `backend_avg_2q_error` | Noisy gates make gate-noise mitigation (ZNE/CDR) valuable |
| `backend_avg_readout_error` | Noisy readout makes REM valuable |

This is a deliberately interpretable feature set — feature importances in the
final report should tell a physics story (e.g. "readout error rate drives REM
selection"), which matters more for the write-up than squeezing out accuracy.

## 6. Why the winner varies (the reason this project exists)

Each technique attacks one error channel and ignores the rest:

- Readout-dominated situation (shallow circuit, bad measurement qubits):
  **REM** wins; ZNE extrapolates gate noise that barely exists and converges
  to the readout-error floor, not to the true value.
- Gate-noise-dominated (deep circuit, many CNOTs, clean readout): **ZNE** or
  **CDR** wins; REM corrects a readout error that was never the problem.
- Near-Clifford structure: **CDR** gets high-quality training circuits and
  can beat ZNE decisively.
- Shallow circuit on a quiet backend: **raw** — mitigation overhead buys
  nothing.

Since circuit properties and backend error profiles both vary, the best
technique is a *function of features* — i.e., a classification problem.

## 7. The ML pipeline, in your native language

| Module | What it is in ML terms |
|---|---|
| `experiment.py` | Dataset builder: brute-force all techniques on every (circuit, backend), label each row with the argmin-error technique (`best_technique`). Crash-safe, resumable. |
| `model.py` | Train RandomForest + GradientBoosting classifiers, *grouped* stratified k-fold CV plus LOFO/LOBO/LODO holdouts (see below), compare against the majority-class baseline, save the best as a joblib bundle. |
| `recommend.py` | Inference: features in, technique + class probabilities out. |
| `report.py` | Evaluation artifacts: error distributions, win rates, confusion matrix, feature importances. |

Two honest caveats you should carry into the write-up. First, **labels are
noisy**: when two techniques finish within shot noise of each other, the
"winner" can flip between reruns — the label is partly a coin toss. The
pipeline's answer is **seed-averaged labels**: each configuration runs at 3
seeds, and `aggregated.csv` (written next to `results.csv`) recomputes each
technique's error as the mean over seeds and labels the winner of the
*means*. Averaging n values shrinks the noise on the mean by ~sqrt(n)
(~1.7x at 3 seeds), so far fewer labels sit inside the noise band — on the
smoke data, per-seed winners disagreed with the seed-averaged winner on
~29% of rows; that disagreement *is* the label noise being removed. One
subtlety: different seeds are genuinely different circuits (different
random angles), so the aggregated label means "best technique for this
kind of circuit (family, size, depth) on this backend" — which matches the
features, since they are angle-blind and identical across seeds anyway.
Train on `aggregated.csv`; keep `results.csv` for per-seed ablations.
Second, **beating the baseline is the bar**: if `rem` wins 70% of rows, a
model with 72% accuracy has learned almost nothing; report macro-F1 and the
baseline side by side.

**How generalization is scored.** Grouped CV answers the weakest question —
"a new configuration of a known family on a known backend". Three held-out
metrics ask progressively harder ones:

- **LOFO** (leave-one-family-out): every circuit of one family is held out;
  the model has never seen that *kind* of circuit. The honest "works on a
  new circuit family" number.
- **LOBO** (leave-one-backend-out): one backend *string* is held out — but
  noise-scaled siblings of the same device (e.g. `FakeManilaV2` and
  `@x2.0` while `@x1.5` is held out) stay in training, bracketing the
  held-out backend's features. That measures noise-level **interpolation**
  on a known device, not a new environment — do not oversell it.
- **LODO** (leave-one-device-out): *all* scales of one device held out
  together; the model has never seen that device at any noise level. The
  honest "works in a new noise environment" number.

Quote LOFO and LODO as headlines; LOBO is the interpolation number.

## 8. Simulator vs real hardware (`hardware.py`)

Everything in sections 1–7 runs on a simulator *pretending* to be an IBM
device, using a frozen snapshot of a real device's calibration data. The
project can also run on the real thing: any backend name starting with
`ibm_` (e.g. `ibm_brisbane`) routes through `hardware.py` to an actual
quantum computer on the IBM Quantum Platform, behind the same
`executor(circuit, pauli) -> float` contract as the simulator. Four things
change, and they are worth understanding before spending any of the free
plan's **10 QPU-minutes per month** (the README section "Switching to real
IBM hardware" has the gated flow: connection check -> cost estimate ->
`hardware_confirmed: true` -> run).

**Queues.** Your job does not run when you submit it — it waits in a shared
fair-share queue with everyone else's. Minutes to hours of wall-clock per
job are normal on the free plan. The good news: queue time is free; only
the seconds the QPU actually spends executing count against the budget. The
bad news: the shipped 28-job first run costs ~85 QPU-seconds but can take
an afternoon of wall-clock time. (Jobs are submitted inside a runtime
`Batch`, which keeps them grouped on the backend, but a batch does not skip
the queue.)

**ISA transpilation.** A real device executes only its native gate set (its
ISA — instruction set architecture), and two-qubit gates only between
qubits that are physically wired to each other. Before submission, qiskit's
transpiler rewrites the circuit: gates are translated into the native
basis, and SWAPs are inserted to route interacting qubits next to each
other. The circuit that actually runs is therefore deeper than the one you
wrote — that extra depth is real extra noise, and it depends on which
physical qubits the transpiler picks. One project-specific subtlety: we
transpile at `optimization_level=0` (the dumbest setting) on purpose —
ZNE's noise-scaling G G† G sequences are logically the identity, and any
optimizing transpiler would delete them, silently turning "3x noise" back
into "1x noise".

**Jobs, not shots, dominate the cost.** The cost model is ~2 s of fixed
per-job overhead plus ~1 ms per shot: at 1024 shots that is ~3 s per job,
two-thirds of it overhead. Halving the shots barely helps — but every
executor call is a whole new job, and the techniques multiply calls: per
(circuit, backend) unit, raw is 1 job, ZNE 3, REM 3, and CDR **11** (the
target circuit plus 10 training circuits). That is why CDR is excluded
from `configs\hw_first_run.yaml` — it alone would nearly triple the bill.
The lever that controls hardware cost is circuits x techniques, not the
shot count.

**The science changes.** The simulator's noise model is a clean, static
approximation; the real device is neither:

- *Readout errors are asymmetric.* P(read 1 | truly 0) != P(read 0 |
  truly 1) — relaxation during the measurement makes 1→0 flips more
  likely. The fake backends store symmetric flip probabilities, and REM's
  inversion is exact only in that symmetric case — on real hardware it
  carries a first-order bias (PROJECT_STATUS §4.8). Watch for this when
  reading `hw_first_run` REM results.
- *Noise drifts.* Real error rates wander over hours and jump at every
  recalibration; the fake backends are frozen snapshots. The model's two
  backend features describe the device as it *was* calibrated, not as it
  is right now — in ML terms, distribution shift between training and
  deployment. This is the paper's hardware angle (roadmap step 4): does a
  selector trained on static simulated noise still pick the right
  technique on live, drifted, asymmetric noise? A "no" is a publishable
  sim-to-real transfer finding, not a failure.
- *No seeding.* Real shot noise cannot be seeded, so hardware results are
  not bit-reproducible (only the transpiler seed is deterministic); two
  identical runs differ at the 1/sqrt(shots) level.
- *Unmodeled effects.* Crosstalk (a gate on one qubit pair disturbing its
  neighbors) and non-Markovian noise (errors correlated in time) exist on
  the real device and are absent from the simulation entirely.

## 9. Where to learn more

- **IBM Quantum Learning** — <https://learning.quantum.ibm.com> — free
  courses; "Basics of quantum information" covers qubits, gates, and
  measurement properly.
- **Mitiq documentation** — <https://mitiq.readthedocs.io> — the user guides
  for ZNE, CDR, and REM explain each technique with runnable code; we use
  mitiq 1.0.0 directly in `mitigation.py`.
- **Qiskit documentation** — <https://docs.quantum.ibm.com> — circuits,
  transpilation, Aer noise simulation.
- Key papers when you write up: Temme, Bravyi & Gambetta (2017) for ZNE;
  Czarnik et al. (2020) for CDR; and mitiq's own paper (LaRose et al.) for
  the toolkit.
- This repo's `notes/spike-zne.md`, `notes/spike-cdr.md`, `notes/spike-rem.md`
  — battle-tested notes on how each technique actually behaves here,
  including the failure modes (readout floors, extrapolation variance,
  ill-conditioned confusion matrices).

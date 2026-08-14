# Code review — correctness + robustness (review-code agent, 2026-07-21)

Scope: `src/qemsel/*`, `scripts/*`, `tests/*`, configs, README, INTERFACES.md.
Every finding below was CONFIRMED by running code with the project venv
(probe scripts in my session scratchpad; key numbers inlined). I deliberately
skipped ground already covered by tester-a/tester-b (endianness of
`expectation_from_counts`, determinism, kill/resume) — I re-verified the
executor's X/Y basis rotations and endianness independently with a noiseless
noise model and they are CORRECT (x(0): ZI=-1/IZ=+1; h+s: YI=+1; h+sdg: YI=-1;
random 3q circuits match `ideal_expectation` to shot noise for ZZZ/XIZ/IYX/ZIY).
REM inversion is exact under symmetric readout error (verified against a
synthetic tensored-confusion executor, error < 4e-16); ZNE and CDR return the
ideal value under a noiseless executor, and CDR's qubit indexing survives the
mitiq qiskit->cirq round trip for circuits without idle qubits.

## F1 (HIGH) — 2-qubit gate noise silently dropped for uncoupled pairs / wrong-direction ECR

**File:** `src/qemsel/backends.py` (`make_executor`), bites `configs/full.yaml`.

`make_executor` transpiles against `AerSimulator(noise_model=...)`, which has
NO coupling map and no gate-direction constraints. Aer applies a noise-model
error only when (gate name, qargs) exactly matches an entry. Consequences,
all empirically confirmed:

- **FakeJakartaV2 / FakeLagosV2** have H-shaped 7q couplings (edges 0-1, 1-2,
  1-3, 3-5, 4-5, 5-6). The brick/chain pairs `(2,3)` and `(3,4)` used by every
  circuit family at `n_qubits >= 4` are NOT device edges, so
  `NoiseModel.from_backend` has no cx entry for them, and the all-to-all
  AerSimulator executes them with ZERO gate noise. Probe (|11> input, which is
  noise-sensitive; |00> is a thermal-relaxation fixed point):
  `FakeLagosV2 pair (0,1): 20 cx add 0.0900 decay; pair (3,4): 0.0000 decay.`
- **FakeSherbrooke** defines ECR in ONE direction per edge (144 directed
  entries, 0 with reverse). The executor's transpile emits `ecr(q, q+1)` in the
  natural direction, which matches the noisy direction only by luck. For the
  5q GHZ chain, 3 of 4 ECRs get zero noise, including `(0,1)` — so even
  2-qubit circuits on Sherbrooke lose all 2q noise. Probe:
  `pair (0,1) (no nm entry): 20 cx add 0.0143 decay (1q wrappers only);
  pair (1,2) (nm entry): 0.1842 decay` — ~13x difference.

**Impact:** tiny.yaml/small.yaml are unaffected (Manila is a 5q line with both
cx directions; Lagos at n<=3 uses only (0,1),(1,2)). But **full.yaml
(n_qubits up to 5, all four backends) would produce systematically
under-noised rows**: at n=5 on Jakarta/Lagos half the 2q-gate locations are
noiseless; on Sherbrooke most are. Labels (`best_technique`) for the flagship
dataset would be derived from unphysical noise, biasing the study toward
"readout-only" regimes (REM over-favored, ZNE starved of the gate noise it
targets). This must be fixed BEFORE the full run.

**Fix (validated in probe):** build the simulator with
`AerSimulator.from_backend(backend)` and transpile against it (keep
`optimization_level=0`, `seed_transpiler=seed`). That preserves coupling map
and gate directions, so the transpiler direction-fixes ECR and inserts routing
SWAPs for non-adjacent pairs — all 4 ecr gates of the 5q chain then carry
noise, and the endianness canary still passes under routing/127q padding
(x(0): ZII=-0.974, IZI=+0.949) because `measure_all()` happens before
transpile, pinning counts bits to logical qubits. Costs: routed circuits carry
realistic SWAP overhead (that is device truth, a feature for the paper) and
Sherbrooke runs ~3 s/execution. Alternative (cheaper, less faithful):
post-process the noise model to add each 2q error to the reverse direction and
to the missing adjacent pairs. Whichever fix lands, add a regression test:
deep-cx on |11> on pair (3,4) of FakeLagosV2 must decay more than the x-x
baseline.

## F2 (MEDIUM) — executor silently accepts circuits wider than the backend

**File:** `src/qemsel/backends.py` (`make_executor`), `src/qemsel/experiment.py`
(`_validate_config`).

`make_executor("FakeManilaV2", ...)` happily runs a 6-qubit GHZ (returned
0.647): qubits beyond the device (noise-model qubits 0-4) simulate with NO
gate/readout noise at all. Nothing in `_validate_config` or the executor
compares `circuit.num_qubits` to the backend size, so a config typo
(`n_qubits: [6]` with FakeManilaV2) yields plausible-looking but partially
noiseless rows instead of an error. Fix: in the executor (or
`_validate_config`, using `get_backend_info(name)["n_qubits"]` vs
`max(config['circuits']['n_qubits'])`), raise ValueError when the circuit is
wider than the backend. One line, prevents silent wrong data.

## F3 (MEDIUM) — crash mid-CSV-append can permanently poison results.csv

**File:** `src/qemsel/experiment.py` (`_append_row`, `_load_existing`).

The crash-safety contract says "a crash loses at most one unit". Confirmed
counterexample: if the process dies mid-write and leaves a partial final line
WITHOUT a trailing newline (e.g. `b_q2_d4_s0,fam,FakeManilaV2,0.4`):

1. On resume, `_load_existing` parses the partial row (pandas fills missing
   trailing fields with NaN), its `(circuit_id, backend)` enters `done_pairs`,
   and the unit is skipped FOREVER — a permanently NaN-valued row that is
   indistinguishable from a legitimate "all techniques failed" row.
2. Worse: the next `_append_row` glues its row onto the partial line
   (`...,0.4c_q2_d4_s0,fam,...` — observed), producing a line with too many
   fields. Every subsequent `pd.read_csv` — resume, `train_model.py`,
   `make_report.py` — then dies with
   `ParserError: Expected 6 fields in line 3, saw 9`. The many-hours full run
   becomes unresumable without hand-editing the CSV.

Fix: before appending to an existing non-empty file, check its last byte and
write a `"\n"` first if missing (open `"ab"`, seek(-1, 2), read). Optionally,
on load, validate the final line's field count and truncate a malformed tail
(logging what was dropped) so case 1 also self-heals. Tester B's kill test
passed because the kill landed between appends — the mid-write window is
small but real (power loss + OS write-back makes torn tails likelier).

## F4 (LOW) — REM ignores the affine offset: first-order bias under asymmetric readout

**File:** `src/qemsel/mitigation.py` (`_apply_rem` docstring + math).

The docstring claims "first-order accurate in the asymmetry". Confirmed with a
synthetic asymmetric-confusion executor (p0=[.02,.03,.01], p1=[.10,.12,.08]):
the residual error is FIRST order in c_i = p1_i - p0_i (e.g. single-qubit
support: rem = true + c/a exactly; observed bias 0.077 while raw error was
0.001 — REM made it worse). The measured channel is affine
(`meas = a*true + c`), but the code divides by the damping only and discards
the offset, even though f0/f1 contain enough information to invert the affine
map exactly for single-qubit support: `alpha=(f0-f1)/2, beta=(f0+f1)/2,
mitigated=(raw-beta)/alpha`.

Why LOW today: I verified `NoiseModel.from_backend` builds SYMMETRIC readout
confusion matrices for these fake backends (Lagos q0: [[0.831,0.169],
[0.169,0.831]] etc. — only a scalar `error` is stored, so p01==p10), and the
implementation is exact under symmetry. It becomes real the moment
`RealHardwareBackend` lands (hardware readout is asymmetric). Suggest: fix the
docstring wording now ("zeroth-order: exact only for symmetric errors"), and
use the exact affine inversion for |support| == 1.

## F5 (LOW) — CDR fails (NaN row) on circuits containing idle qubits

**File:** `src/qemsel/mitigation.py` (`_apply_cdr`).

mitiq's qiskit->cirq->qiskit round trip drops idle wires from the training
circuits, so the executor/simulator lambdas then see fewer qubits than
`len(pauli)` and raise (`[cdr] ValueError: pauli length 3 != circuit
num_qubits 2` — confirmed via `apply_technique`, which correctly converts it
to MitigationError -> NaN). It can never produce silently wrong numbers (the
length validation always fires) and I verified none of the 5 shipped families
generate idle qubits at any (n, depth) — so the pipeline is safe — but any
library user calling `apply_technique("cdr", ...)` on their own circuit with a
spectator qubit gets an opaque failure. Fix: detect idle qubits in
`_apply_cdr` and raise MitigationError with a clear message, or pad each idle
qubit with an explicit gate that survives the round trip before calling mitiq.

## F6 (LOW) — report `_fmt` crashes on infinity

**File:** `src/qemsel/report.py` (`_fmt`).

`_fmt(float("inf"))` raises `OverflowError: cannot convert float infinity to
integer` at `val == int(val)`. NaN is handled but +/-inf is not; a single inf
in a mean (REM's unclipped `raw/damping` can be arbitrarily large when damping
is near REM_MIN_DAMPING) would kill report generation. One-line fix: guard
with `math.isfinite(val)` instead of only `isnan`.

## Observations (no action required)

- **PROJECT_STATE inaccuracy:** FakeLagosV2's stored readout errors are q0
  16.9%, q1 13.6%, **q2 46.4%**, q3 1.7%, q4 2.9% — not "~27% on q0/q1". At
  n>=3, q2 dominates: the tensored damping for Z⊗Z⊗Z is ~0.035, so REM's
  inversion amplifies shot noise ~29x. Measured: rem on a 3q GHZ (ideal 0)
  across 5 sim seeds gave {+0.07, -0.86, -0.92, +0.31, -0.13}. REM labels on
  Lagos rows with q2 in support are close to coin flips at 2000-8000 shots —
  one more reason for the planned seed-averaging of labels before the full
  run. (REM_MIN_DAMPING=1e-6 is far too small to catch this; a threshold near
  0.02-0.05 would flag it as a failure instead of recording noise.)
- Executor contract holds everywhere I traced it (mitigation wraps it 1-arg,
  REM calibration goes through it, experiment builds it per unit with
  spec.seed); no mutable default args, no global RNG use, no seed leaks found
  (grep + runtime checks). The broad `except Exception` in `run_experiment` is
  intentional per-technique isolation and correctly logs to errors.log.
- Measure/transpile ordering is correct in the current code AND remains
  correct under the F1 fix because measurements are appended before
  transpiling, so counts bits always index logical qubits.
- `tests/conftest.py::tiny_results_df` lacks the `best_technique_cost_aware`
  column; harmless today (model/report don't require it) but worth adding if
  anything starts consuming that column.

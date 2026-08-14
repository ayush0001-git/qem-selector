# Spike notes: CDR (mitiq.cdr.execute_with_cdr) — mitiq 1.0.0

**Status: WORKS.** `spikes/spike_cdr.py` runs end-to-end.
Result (2q circuit, 12 non-Clifford rz gates, FakeManilaV2 noise, 20k shots):
ideal <ZZ> = +0.7513, raw = +0.6447 (|err| 0.107), CDR = +0.7764 (|err| 0.025) — 4.25x.
Stability check over training seeds random_state=1,2,3: mitigated errors
0.0072 / 0.0078 / 0.0023 → improvement is real, 13–46x, not a fluke.

## Exact API (verified against installed source: `.venv/Lib/site-packages/mitiq/cdr/cdr.py`)

```python
from mitiq.cdr import execute_with_cdr

mitigated = execute_with_cdr(
    circuit,                    # qiskit QuantumCircuit OK (or cirq)
    executor,                   # noisy: callable(circuit) -> float
    observable=None,            # optional mitiq.Observable
    simulator=ideal_simulator,  # KEYWORD-ONLY and REQUIRED: callable(circuit) -> float
    num_training_circuits=10,   # default
    fraction_non_clifford=0.1,  # default: fraction of non-Clifford rz KEPT in training circuits
    scale_factors=(1,),         # default (1,) = plain CDR; add >1 factors = vnCDR
    # **kwargs: method_select ('uniform'|'gaussian'), method_replace
    #           ('uniform'|'gaussian'|'closest'), sigma_select, sigma_replace,
    #           random_state (int seed for training-circuit sampling)
)
```

- `simulator` is **keyword-only** (`*` in the signature) — positional passing raises TypeError.
- Both `executor` and `simulator` are plain callables auto-wrapped in `mitiq.Executor`.
  With `observable=None` BOTH must return an expectation value (float).
  (Alternative: pass a `mitiq.Observable` and return `MeasurementResult`; not needed.)
- Also available: `mitiq.cdr.mitigate_executor(...)` and `@cdr_decorator(...)` wrappers.

## How training circuits are generated

You do NOT supply them. `execute_with_cdr` calls
`mitiq.cdr.generate_training_circuits(circuit, num_training_circuits,
fraction_non_clifford, method_select='uniform', method_replace='closest',
random_state, ...)` internally. It finds non-Clifford ops via
`cirq.has_stabilizer_effect` and replaces ~(1 - fraction_non_clifford) of them
with Clifford rz angles (multiples of pi/2), producing `num_training_circuits`
near-Clifford variants. It runs the noisy executor on [circuit] + training
circuits, the ideal simulator on the training circuits only, then does a linear
`scipy.optimize.curve_fit` (default `linear_fit_function`, intercept included)
and applies the learned map to the noisy value of the circuit of interest.

`generate_training_circuits` is decorated `@atomic_one_to_many_converter`:
qiskit circuit in → qiskit training circuits out. So **your executors always
receive circuits in the same frontend you passed in**.

## Gotchas

1. **Circuit basis requirement:** all non-Clifford gates must be rz rotations
   (circuit compiled to roughly {rz, sx/h, cx}). h/cx are Clifford, fine.
   If the circuit contains e.g. general u/ry gates, gate replacement can't work.
   A fully-Clifford circuit short-circuits: returns the simulator value directly.
2. **Executor receives circuits WITHOUT measurements** — copy + `measure_all()`
   inside the executor, then `transpile(circ, noisy_aer_sim, optimization_level=0)`.
   opt level 0 keeps replaced/folded gates from being optimized away.
3. **Pick a circuit whose ideal expectation is well away from 0.** Noise biases
   <Z...Z> toward 0, so if ideal ≈ 0 the raw error is tiny and any "improvement"
   drowns in shot noise (first attempt: ideal -0.083, raw err 0.0023 < shot
   noise 0.007 @ 20k shots). Aim for |ideal| ≳ 0.5.
4. **Little-endian counts:** qiskit count keys have qubit 0 as the RIGHTMOST
   char. For Z⊗...⊗Z on ALL qubits, bitstring parity is endian-agnostic; for a
   subset, reverse the key (`key[::-1]`) so index i = qubit i.
5. **Ideal simulator = exact statevector** (`qiskit.quantum_info.Statevector`
   `.expectation_value(SparsePauliOp("ZZ"))`) → noise-free training labels, no
   shot noise. Call `remove_final_measurements(inplace=False)` defensively.
6. **Cost per CDR call:** noisy executor evaluates (1 + num_training_circuits)
   × len(scale_factors) circuits; simulator evaluates num_training_circuits.
   Serial callable is fine at this size (31 Aer runs in seconds).
7. `random_state=<int>` (via **kwargs) makes training-circuit sampling
   reproducible. Seed Aer per-run (`seed_simulator=`) separately if you want
   fully deterministic output; use a fresh seed per executor call.
8. `ply` must be installed for mitiq's qiskit→cirq conversion (already pinned;
   see env-setup notes).
9. `fraction_non_clifford` is the fraction KEPT non-Clifford (not replaced).
   0.1–0.2 worked well; num_training_circuits=30 gave a stable fit.

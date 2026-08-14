# Notes — ZNE API spike (2026-07-21)

**Status: WORKS.** `spikes/spike_zne.py` runs end-to-end and passes its asserts
(both Richardson and linear ZNE land closer to ideal than raw).

## Exact API (introspected from installed mitiq 1.0.0 — do not trust memory)

```python
mitiq.zne.execute_with_zne(
    circuit,                 # qiskit QuantumCircuit OR cirq Circuit
    executor,                # Callable[[circuit], float] or mitiq.Executor
    observable=None,         # optional mitiq Observable; None => executor returns the expectation itself
    *,
    factory=None,            # None => RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])
    scale_noise=fold_gates_at_random,   # default gate folding (mitiq.zne.scaling)
    num_to_average=1,
) -> float
```

- `RichardsonFactory(scale_factors, shot_list=None)`, `LinearFactory(scale_factors, shot_list=None)`
  — both in `mitiq.zne` (also `ExpFactory`, `PolyFactory`, `AdaExpFactory`, `PolyExpFactory`).
- The factory instance you pass in is **mutated** by `execute_with_zne`; afterwards you can read
  `factory.get_scale_factors()`, `factory.get_expectation_values()`, `factory.get_zero_noise_limit()`,
  `factory.get_zero_noise_limit_error()`, `factory.plot_fit()`. Great for the benchmark pipeline
  (log the raw scaled expectations, not just the final number).
- Related entry points that exist and may be useful later: `zne.mitigate_executor`,
  `zne.construct_circuits` (get the folded circuits without running), `zne.combine_results`,
  `zne.scaling.fold_global`, `zne.scaling.fold_all`, `zne.scaling.insert_id_layers`.

## Gotchas (things that bit or almost bit)

1. **Build the circuit WITHOUT measurements; add `measure_all()` on a copy inside the executor.**
   Folding tolerates *terminal* measurements but raises `UnfoldableCircuitError` on mid-circuit
   ones (`mitiq.zne.scaling.folding._check_foldable`). Measurement-free input + measure in the
   executor is the robust pattern.
2. **The executor receives a circuit of the same frontend as the input** (qiskit in → qiskit
   back), already folded. Do NOT re-derive anything from the original circuit inside the executor.
3. **`transpile(circ, aer_backend, optimization_level=0)`** — level 0 is mandatory or the folded
   `G G† G` sequences get optimized away and ZNE silently measures nothing. Pass only the
   AerSimulator built with the noise model (adding `basis_gates=` triggers a qiskit 2.5
   UserWarning). Verified folding survives: 32 → 66 → 96 CNOTs at scales 1/2/3 after transpile.
4. **Noise only attaches to the noise model's basis gates** (`['cx','id','rz','sx','x',...]` for
   FakeManilaV2). An `rx` in the logical circuit only becomes noisy after transpilation decomposes
   it to `rz`/`sx`. Transpiling *inside* the executor (i.e., after folding) handles this correctly.
5. **Counts are little-endian** (leftmost char of the bitstring = highest qubit index). For the
   all-Z parity observable `<Z...Z> = sum (-1)^popcount(b) p(b)` order is irrelevant, but for a
   general Pauli like `ZIZ` you must use `bitstring[::-1][qubit_index]`.
6. **ZNE cannot fix readout error — measure the readout floor before judging it.** On FakeManilaV2,
   q2 has 9.6% readout error (q0 3.5%, q1 2.2%); parity damping factor
   `(1-2p0)(1-2p1)(1-2p2) = 0.717`, so ideal 0.562 → floor ≈ 0.403 with *zero* gate noise
   (verified empirically: rx-only circuit gives 0.405). ZNE extrapolation converges to this floor,
   not to ideal. In one run: raw 0.366 → Richardson 0.407 ≈ floor (essentially all recoverable
   gate noise removed). **For the benchmark pipeline, an "improvement" metric for ZNE should be
   read against what is recoverable** — circuits on high-readout-error backends (FakeLagosV2!)
   will make ZNE look bad and REM look good, which is exactly the signal the classifier needs.
7. **Richardson ZNE is flaky at moderate shots — this actually bit us.** With 20k shots and
   unseeded folding, one run FAILED the "mitigated closer than raw" assert (Richardson 0.344 vs
   raw 0.357): Richardson with scale factors [1,2,3] has extrapolation coefficients [3,-3,1],
   amplifying single-point shot noise ~4.4x, and `fold_gates_at_random` re-randomizes each call
   (default `seed=None`), so the folded circuit itself changes run to run. Fix (now in the spike):
   `scale_noise=functools.partial(fold_gates_at_random, seed=SEED)` + a deterministic
   `seed_simulator` per executor call (incrementing counter — do NOT reuse one seed, that
   correlates scale points) + 100k shots. Richardson can also overshoot the readout floor
   (extrapolation variance/bias); Linear is more stable but more biased toward the noisy side.
   **Benchmark pipeline implication:** average multiple ZNE runs (or use `num_to_average`) or the
   ZNE-vs-others labels will be noisy.
8. `ply` must be installed (env agent already pinned it) — mitiq's qiskit→cirq conversion needs
   it; symptom is `ModuleNotFoundError: No module named 'ply'` deep inside `execute_with_zne`.

## Final output (spikes/spike_zne.py, FakeManilaV2, 100k shots, seeded — reproduces exactly)

```
ideal <ZZZ>                : +0.5622  (= cos(0.6)^3)
readout-limited floor      : +0.3984  (gate noise = 0, ZNE's best case)
raw noisy                  : +0.3582   |err| = 0.2040
ZNE Richardson [1,2,3]     : +0.4255   |err| = 0.1367
ZNE Linear     [1,2,3]     : +0.4027   |err| = 0.1595
ZNE default factory        : varies (intentionally unseeded folding, demo of default call path)
```

Circuit: 3 qubits, rx(0.6) on each + 16 self-cancelling CNOT pairs (32 CNOTs) — identity-composing
two-qubit layers give a gate-noise-dominated circuit with an exactly known ideal value.
Verified folding scales gate count correctly through the whole qiskit->cirq->fold->qiskit->
transpile(opt=0) chain: 32 -> 66 -> 96 CNOTs at scale factors 1/2/3.

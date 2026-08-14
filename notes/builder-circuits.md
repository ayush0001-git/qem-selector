# Notes — builder-circuits (2026-07-21)

## Status: DONE, tests green

- `src/qemsel/circuits.py` fully implemented (all stub signatures verbatim,
  docstrings preserved). `tests/test_circuits.py`: 87 tests, all pass in ~0.3 s
  standalone. Full `tests/` dir also passes (165 tests) with my module in place.

## Implementation choices integrators should know

1. **No barriers anywhere.** Deliberate: mitiq's qiskit->cirq conversion and
   features' gate counting are simpler without them; executors use
   `optimization_level=0` so U/U-dagger cancellation is not a risk for
   mirror_circuit anyway.
2. **layered_random:** per layer, one random rx/ry/rz (angle U[0,2pi)) per
   qubit, then a deterministic alternating brick CX pattern (even layers pair
   from q0, odd from q1). For n_qubits=2, odd layers have NO CX — expected.
3. **near_clifford:** 1q slots drawn from {h,s,sdg,x,z}; with prob
   `non_clifford_fraction` replaced by T (50%) or rz at an angle rejected-sampled
   to be > 1e-6 away from any multiple of pi/2 (features' Clifford tol is 1e-9,
   so these always count as non-Clifford). Brick CX layers as in layered_random.
   Fraction outside [0,1] raises ValueError.
4. **ghz_plus:** GHZ prep, then while `qc.depth() < depth` appends a random
   identity pair (rz(a);rz(-a), X;X, H;H, or CX;CX on a random adjacent pair).
   State stays EXACTLY GHZ; final depth may exceed `depth` slightly. If `depth`
   <= prep depth, no padding (circuit is deeper than asked — by design).
5. **hw_efficient_ansatz:** per block, ry layer then rz layer (all angles bound
   floats, `num_parameters == 0`), then linear CX chain; one extra ry+rz layer
   at the end. Gate counts: ry = rz = n*(depth+1), cx = (n-1)*depth.
6. **mirror_circuit:** exactly `U.compose(U.inverse())` with
   U = layered_random(n, max(1, depth//2), seed); circuit name set to the
   circuit_id pattern. Verified `<Z*n>` = +1 within 1e-12 for n=2..5, depths
   1..16.
7. **generate_suite:** validates required keys present AND non-empty
   (ValueError), unknown families (ValueError). Nesting order: family >
   n_qubits > depths > seeds. `params` copied into each CircuitSpec (fresh dict
   per spec, no sharing).
8. All generators raise ValueError for n_qubits < 1 or depth < 1. n_qubits=1
   is allowed (no CX pairs then), though the project uses 2-5.
9. Determinism: every rng draw comes from a local
   `np.random.default_rng(seed)`; verified identical op lists across calls.
   Circuit `.name` is set (e.g. `layered_random_q3_d6_s0`) — cosmetic only.

## No interface deviations. Nothing to flag.

# Adversarial tester T-features — round 1 (2026-07-23)

Verdict: PASS (0 critical, 0 major, 3 minor/informational). All verification by
RUNNING code. Harness: scratchpad adv_features_t1.py (session temp dir); repro
commands below.

## Mandated checks — all PASS

1. Hand-computed entangling_density + 2q-layer count (my own circuits, not the
   builders'):
   - 4q mixed circuit (h, 3x cx, t, rz(0.7), swap, rx(pi/2)): depth=4,
     n_2q_layers=3 ({cx01,cx23} | cx12 | swap), entangling_density=4/16=0.25,
     n_cnot=3 (swap correctly excluded), n_non_clifford=2, mean_rz_angle_dist
     =(0.7/(pi/4))/2 — all exact.
   - Asymmetric 3q (2 serial cx + 5 serial h on a spectator qubit): depth=5,
     n_2q_layers=2, entangling_density=2/15 — n_2q_layers correctly independent
     of the 1q-driven critical path.
   - ccx probe: n_2q_gates=0 but n_2q_layers=1 (2+-qubit filter, per spec),
     entangling_density=0, clifford_fraction=0.
2. log2_shots exact for base_shots in {1, 2, 256, 2000, 4096, 2^62, 0.5,
   1500.0} (0.0, 1.0, 8.0, log2(2000), 12.0, 62.0, -1.0, log2(1500)).
3. V1 byte-identical to captured reference: regenerated the tiny suite from
   configs\tiny.yaml and compared extract_features(circuit, backend) (exact
   two-arg v1 path, REAL backends) against results\tiny\results.csv — all
   20 rows x 10 feat_* values bitwise equal to the raw CSV tokens. Builders'
   own captured-reference tests also green (68 passed). V2's ten-feature
   prefix is bitwise == standalone V1 on every smoke circuit.
4. Stable ordering: list(feats) == FEATURE_NAMES / FEATURE_NAMES_V2 over 400
   calls across 4 circuits; JSON serialization byte-identical across two
   processes with PYTHONHASHSEED=0 vs 987654321.

## Bonus checks — PASS

- V2 bitwise-identical to the captured results\boundary_smoke\results.csv run:
  24 rows x 15 feat_* values (incl. dialed FakeManilaV2@x0.25/@x0.5 backend
  features, feat_log2_shots for both budgets 256/4096) — regenerated circuits
  via generate_suite(min_abs_ideal=0.25), all bitwise equal to raw CSV tokens.
- Error contract: base_shots None/0/-1 -> ValueError mentioning base_shots
  (fires before backend lookup); version 3 / "2" -> ValueError; measured
  circuit -> ValueError on both versions; unknown backend -> ValueError; v1
  ignores base_shots even when invalid; all values plain float; input circuit
  not mutated.

## Minor / informational findings (no action required this round)

1. MINOR — barriers inflate n_2q_layers: barrier(0,1) passes the
   `len(instr.qubits) >= 2` filter, so h(0); barrier(0,1); cx(0,1) gives
   n_2q_layers=2.0 (true 2q-GATE layers = 1) while depth()=2.0 (default
   filter excludes directives). Docstring says "layers containing a 2+-qubit
   gate" but also pins the exact depth() call, and circuits.py guarantees no
   barriers are emitted — latent hazard for external callers only.
2. MINOR — hostile base_shots values pass the `<= 0` gate: base_shots=nan ->
   log2_shots=nan (silent feature-matrix poison), inf -> inf, True -> 0.0
   (bool is int). Unreachable through run_experiment's config validation.
3. INFO — version=2.0 (float) is accepted (dict-key hashing: 2.0 == 2);
   harmless.

## Repro

    & "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" -m pytest "E:\quatum  computiiing\qem-selector\tests\test_features.py" "E:\quatum  computiiing\qem-selector\tests\test_features_v2.py" -q

Harness gotcha (cost me one false alarm): comparing against the result CSVs
with pandas defaults reports last-ULP "drift" — pandas read_csv's default
float parser is not round-trip. Use float_precision="round_trip" or compare
repr(computed) to the raw CSV token. The stored CSVs themselves are exact.

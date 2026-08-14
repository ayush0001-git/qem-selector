# B5 — features.py V2

Implemented `extract_features(..., version=2, base_shots=...)`. V1 path
untouched behaviourally (byte-identical; regression pinned against values
captured from pre-V2 code). Only new files touched: `src/qemsel/features.py`,
`tests/test_features_v2.py`.

Five additive features (order = FEATURE_NAMES_V2, 15 keys):
- `log2_shots` = `math.log2(float(base_shots))`; base_shots > 0 REQUIRED for v2
  (ValueError, checked before backend lookup). Accepts int or float.
- `n_2q_layers` = `circuit.depth(filter_function=lambda i: len(i.qubits)>=2)`.
- `entangling_density` = `n_2q_gates/(n_qubits*depth)`, 0.0 if denom 0.
- `mean_rz_angle_dist` = mean over rx/ry/rz/p of
  `abs(remainder(angle, pi/2))/(pi/4)`; unbound param -> 1.0; 0.0 if none.
- `backend_avg_1q_error` from get_backend_info.

Shared V1 loop reused; angle accumulation is v2-only (no v1 effect). All values
plain floats, key order asserted. 42 new tests; full suite green.

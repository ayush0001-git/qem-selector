# B2 — backends.py sub-unity noise dial (0 < scale < 1)

**Finding:** `backends.py` ALREADY implements the sub-unity path — no code change
needed. `parse_backend_name` accepts any finite scale > 0; `get_backend_info` and
`_build_scaled_noise_model` use `min(scale*e, cap)`, which for scale < 1 is just
`scale*e` (caps never bind). Verified end-to-end at x0.25/x0.5. So I left
`backends.py` byte-identical (strongest possible regression guarantee) and delivered
the pinning tests only.

**Captured (current code, exact):** plain Bell <ZZ> (256,seed7) = 0.8671875;
sub-unity Bell = 0.9375 (x0.5) / 0.96875 (x0.25). Ratios EXACTLY the scale on
Manila AND cap-saturated Lagos (plain q2 readout 46.4% > 0.45 cap, yet x0.5 → 23.19%,
strictly under cap — the key sub-unity contract). Monotone down-dial confirmed
(Lagos readout 0.145/0.280/0.516; deep-cx 0.172/0.318/0.573) and continuous across
unity (Manila x0.5<x1.0<x1.5).

**Deliverable:** `tests/test_noise_scaling_v2.py` (36 tests): parsing, exact-linear
info scaling + caps-never-bind, extract_features flow-through, both-direction
monotonicity, non-device-edge routing, determinism, plain/>=1 byte-identical
regression. New file only; existing 424 untouched.

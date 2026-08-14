# B6-stats

Implemented `src/qemsel/stats.py` (all 8 stubs) + `tests/test_stats.py` (51 tests).
No qemsel imports; deterministic (`default_rng(seed)`); JSON-serializable outputs.

Key decisions:
- `koester_checklist` argmin check mirrors experiment's aggregated **coverage rule**:
  a technique is eligible to win only if `<tech>_n_seeds == group max`. Without this,
  the aggregated schema showed 99 false mismatches. Per-seed schema has no such rule.
- `winner_margin_below_k_sigma`: computed via `sigma_shot(value, shots)` on the per-seed
  schema; returns `n_flagged=None, fraction=None` on aggregated (no shots/scatter columns,
  and no qemsel import to reconstruct shots from base_shots — refused to fabricate a 0).
- argmin tie-break = stable sort in column-appearance order = experiment's config order.

Checklist run on frozen sweep (anchored in tests):
- `results/research/results.csv` (per_seed, 1620): overshoot zne 5 / cdr 159 / rem 53;
  error_beyond rem 5; nan_rate cdr 415/1620, rem 156/1620; argmin mismatch 0;
  winner_margin flagged 229 (14.1% ties); passed True.
- `results/research/aggregated.csv` (540): mismatch 0, partial_coverage 0, passed True.

Owned only stats.py + test_stats.py (compute_stats.py explicitly out of this task's scope).
424 existing tests untouched.

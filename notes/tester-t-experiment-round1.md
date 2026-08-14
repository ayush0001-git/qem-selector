# Adversarial tester — T-experiment (shots axis), round 1

Date: 2026-07-23. Verified by RUNNING code (no src changes). All artifacts in
`C:\Users\ayush\AppData\Local\Temp\claude\E--quatum--computiiing\13852342-c2be-4d47-b461-7c0561ff460a\scratchpad`
(configs t_count.yaml / t_count_ext.yaml / t_scalar512.yaml / t_listone.yaml /
t_lowsig.yaml, driver t_errlog.py, checkers check_*.py, logs).

## PASSED

1. **Cross product / unit count**: `shots: [128, 512, 2048]` x 4 circuits x 2
   backends -> exactly 24 rows, each (circuit_id, backend, base_shots) key
   exactly once, row order = circuits(outer) x backends x budgets(innermost,
   config order). Log line "24 units (4 circuits x 2 backends x 3 shot-budgets)".
2. **Kill+restart resume**: hard-killed (Stop-Process -Force) after 8 data rows
   (pair ghz_s1@Manila had 2 of 3 budgets done). Resume skipped exactly those 8
   per-budget keys, computed 16, final results.csv **SHA256-identical** to an
   uninterrupted single-pass run (61CF52A1..., 6346 bytes).
3. **Torn tail**: truncated results.csv mid final row -> self-heal message,
   1 unit recomputed, byte-identical final.
4. **Budget-list extension resume**: [128,512,2048] -> [128,512,2048,64] on the
   same out_dir: 32 units, 24 skipped, only the 8 new 64-shot units computed,
   first 24 rows byte-untouched.
5. **Scalar backward compat**: fresh configs/tiny.yaml run SHA256-identical to
   results/tiny/results.csv (CBDF857D..., 6248 bytes), no base_shots column.
6. **List-of-one**: `shots: [512]` IS list mode (base_shots column present) per
   contract; all other values identical to scalar 512 (round_trip compare).
7. **base_shots column**: between pauli and ideal, int64, per-row correct;
   `<tech>_shots == shots_consumed(tech, row's own budget)` on all 24x3 cells;
   ideal shots-independent per pair; raw values differ across budgets
   (independent per-unit executors).
8. **aggregated.csv V2 keys**: family,n_qubits,depth,backend,base_shots,n_seeds;
   12 groups, n_seeds=2, means match per-seed rows.
9. **Schema-mismatch guard both directions** (list config into scalar out_dir
   and vice versa): clean exit 2 "use a fresh out_dir or a matching config",
   existing results.csv untouched.
10. **_normalize_shots validation**: bool / 0 / negative / [] / dupes / floats /
    [True,...] all ValueError; tuple accepted; order preserved.
11. **Low-signal screen in list mode**: min_abs_ideal 0.9 skipped layered_random
    for ALL budgets (1 V1-format log line per pair), counter still reached 8/8,
    rows = surviving pairs x budgets.
12. **List-mode errors.log**: injected zne failure at base_shots=512 ->
    `ghz_plus_q2_d4_s0,FakeManilaV2,s512,zne: RuntimeError(...)`, NaN triple,
    other budget unaffected, winner excludes failed tech.

## FINDING (minor, not shots-axis-specific)

**aggregated.csv is not byte-deterministic across resumed vs single-pass runs.**
After resume, existing rows come from `pd.read_csv` with the DEFAULT float
parser, which is not exactly round-trip (confirmed: 80 cell-level diffs vs
`float_precision="round_trip"` on the same results.csv, pandas 2.3.3). The
recomputed aggregated means then drift in the last 1-2 ULPs (6 of 12 lines
differed between my resumed and single-pass runs; winner labels unaffected).
results.csv itself is byte-identical (resumed rows are never rewritten).
Pre-existing V1 mechanism (scalar resume has it too). Candidate fix:
`float_precision="round_trip"` in `experiment._load_existing`. Repro:
run scratchpad t_count.yaml twice (once interrupted+resumed, once clean) and
diff aggregated.csv.

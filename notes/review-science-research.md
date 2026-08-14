# Science/Stats review — research-run readiness (2026-07-21)

Reviewer scope: research validity of the noise-scaling + raw_plus + seed-averaging +
min_abs_ideal upgrades, verified against code and the fresh
`results/research_smoke/` run (45 rows). All numbers below were recomputed by
running code (scripts in the session scratchpad), not taken from agent logs.

**VERDICT: do NOT treat the pipeline's headline numbers as paper-ready yet
(3 major findings), but NOTHING blocks launching the ~5 h research sweep —
every major item is a post-processing or disclosure fix computable from
results.csv after the run. Launch the run; fix the items below before quoting
any headline number.**

## Q1 — min_abs_ideal rejection sampling: real, family- and size-dependent bias

Measured on the exact research.yaml grid (180 circuits, 0 best-so-far warnings):

| family | mean attempts | bumped slots | worst cell |
|---|---|---|---|
| ghz_plus / mirror | 1.00 (exempt) | 0/36 | — |
| hw_efficient_ansatz | 3.03 | 24/36 | n=5 d=16: mean 7.0, max 17 |
| layered_random | 4.44 | 28/36 | n=5 d=4: mean 10.3, max 22 |
| near_clifford | 6.44 | 31/36 | n=4 d=8: mean 17.0, max 27 |

Unconditioned |<Z^n>| distributions (200 fresh seeds): layered_random n=5 d=16
median 0.127, P(pass 0.25) = 0.18, accepted mean 0.381 — the suite keeps the
top ~fifth of the |ideal| distribution, i.e. **atypically non-scrambling
circuits**. near_clifford is qualitatively worse: unconditioned MEDIAN is
exactly 0.000 (P(pass) 0.07–0.35); the accepted subset is the minority whose
Clifford backbone keeps Z^n (near-)stabilizer — at n=2 d=4 every accepted
circuit has |ideal| = 1.0 exactly. Verdict: acceptable ONLY with explicit
disclosure that the families are *conditioned* ensembles ("random circuits
conditioned on |<Z^n>| >= 0.25"), that the conditioning strengthens with n and
depth, and that results do not transfer to typical scrambling circuits (whose
observables vanish — defensible: nobody mitigates a zero-signal observable).
This disclosure currently exists NOWHERE user-facing (not in report.md).
Side effect to note: post-conditioning |ideal| magnitude differs systematically
across families/cells, and abs_error comparisons inherit that.

## Q2 — seed-averaged winner matters, but the model never sees it (MAJOR)

Smoke data: per-seed winner != seed-averaged winner on **13/45 rows (28.9%)**
(cost-aware 12/45 = 26.7%); only **6/15 groups** have unanimous per-seed
winners (8/15 cost-aware). So seed-averaging removes substantial label noise —
the motivation was right. BUT:

* `aggregated.csv` has **no feat_* columns** (`experiment._aggregated_columns`),
  and `model.train_and_eval(aggregated_df, ...)` raises
  `ValueError: df lacks required columns: ['feat_n_qubits', ...]` (verified).
* `train_model.py` was (and will be) pointed at `results.csv`, so the CV 0.727,
  LOFO 0.711, LOBO 0.689 smoke headline numbers — and the research run's
  planned headline — are **per-seed-label numbers carrying the ~29% flip
  noise** the aggregation was built to remove.
* PROJECT_STATE "§6.4 seed-averaged labels -> RESOLVED" is over-claimed:
  resolved as a data artifact, unresolved as a modeling input.

Fix (post-run, small): merge the aggregated winner back onto the per-seed
feature rows by (family, n_qubits, depth, backend) and train on that (features
legitimately vary per seed for near_clifford/ghz_plus, so merge-back beats
adding features to aggregated.csv). Also found: 3/15 smoke aggregate winners
(all cdr, ghz_plus groups) come from a **1-of-3-seed mean** competing against
3-seed means (CDR refused the other seeds). Record per-technique
n_valid_seeds in aggregated.csv or require full coverage to win; at minimum
disclose — recommending cdr for a config where CDR refuses 2/3 of seeds is
semantically shaky (interacts with open §6.2 cdr_refused feature).

## Q3 — raw_plus is implemented fairly (PASS)

Verified: same `make_executor` path (identical transpile/noise construction),
one execution at genuinely 11x shots (`raw_plus_shots == 11*raw_shots` on
45/45; values differ from raw on 45/45), and it enters BOTH winner labels with
multiplier 11 consistently (per-row `_pick_winners` and aggregated path both
use sqrt(shots/base)). Paired raw_plus−raw error: −0.0011 ± 0.0132 —
bias-dominated, exactly the physics the control is meant to show. Note:
raw_plus is structurally near-unwinnable in the *cost-aware* label (raw's
error with a sqrt(11) penalty; 0 wins in smoke while raw won 8) — correct
behavior, but the report should say its value is as a comparison column, not a
reachable class.

## Q4 — CDR label signal in research.yaml: better than claimed (PASS)

Ran the actual `_apply_cdr` pre-guards (Clifford check + training-ideal spread;
backend-independent) on all 180 research circuits:
layered_random 36/36 OK, hw_efficient 36/36, mirror 36/36,
**ghz_plus 11/36 OK** (17 Clifford-refuse, 8 spread-refuse),
**near_clifford 15/36 OK** (1 + 20). Total **1206/1620 units (74%)
non-refused** — comfortably enough signal, spanning all 5 families (the
config header's "972 units" undersells it). Runtime failures beyond the
guards are possible but were zero in smoke's viable families.

## Q5 — LOFO clean; LOBO headline needs a leave-one-device-out companion (MAJOR)

LOFO/LOBO code is leakage-free in the literal sense (held-out group rows never
in train; CV grouping by (family,n_qubits,depth) spans backends — conservative,
good). But in the 9-environment design, **6 of 9 LOBO folds hold out a
scale-sibling** (e.g. Manila@x1.5) while the SAME device at x1.0/x2.0 — with
the identical circuit set and bracketing backend-feature values — stays in
training. Those folds measure noise-level *interpolation on a known device*,
not "generalizes to a NEW noise environment". A smoke-scale probe (Lagos@x1.5
held out with vs without its plain sibling: 0.667 vs 0.733) shows no inflation
at 45 rows, but that is noise-limited; the structural point stands at 1620.
Required for the paper: report **leave-one-device-out** (hold out all 3 scales
of one device; 3 folds) next to LOBO, and label LOBO as the interpolation
number. Post-hoc computable, no code in the sweep affected. (Winner's-curse on
best-of-2 model selection stays documented at §6.14.)

## Q6 — noise scaling: monotone in aggregate, but two mandatory caveats

(a) **Lagos cap compression / local non-monotonicity (MAJOR, disclosure).**
The 0.45 readout cap bites Lagos hard: realized avg readout scales **x1.277 /
x1.440** (not 1.5/2.0); q2 readout **decreases** 0.4638 -> 0.45 from x1.0 to
x1.5; q6 is flat 0.45 at both x1.5/x2.0; max_readout_error is non-monotone
(0.4638 -> 0.45). Empirical confirmation in smoke errors.log: the same
circuit's REM damping was 0.0159 on plain Lagos but 0.0190 (easier!) on
Lagos@x1.5. So for the one readout-dominated device, the readout dial is
compressed and locally inverted — REM-vs-scale trends on Lagos are partly cap
artifacts, and research.yaml's "readout errors scaled by <scale>" /
"scale and topology unconfounded" and the integrator's "features exactly 1.5x
plain" are false for Lagos (2q errors DO scale exactly on all devices;
Manila/Jakarta readout scales exactly too). The cap itself is necessary
(uncapped q2 at x1.5 = 0.696 -> negative damping, unphysical). Fix = disclose
+ prefer plotting winners against *realized* avg error rather than nominal
scale; don't claim a continuous readout axis for Lagos.

(b) **x1.0 is a different noise-model family (minor, disclosure).** Plain names
run the `from_backend` composite channels (thermal-relaxation mix, 3-circuit
quantum errors — verified); scaled variants are pure depolarizing + symmetric
readout. Readout is symmetric in BOTH (verified on Lagos — good, REM
consistent), but gate-noise *character* changes at the 1.0 -> 1.5 step, so
that step conflates strength with channel type (ZNE is the sensitive one).
Aggregate monotonicity holds empirically (raw worse on 13/15 paired Lagos
circuits) and the builder documented the dial honestly in backends.py, but
report §5 currently presents scale as one continuous axis with no caveat.

## Record corrections (minor)

* Integrator's smoke NaN attribution is swapped: actual is **near_clifford 9/9**
  CDR-refused and **ghz_plus 6/9** — and the ghz_plus refusals are
  degenerate-training-spread refusals, NOT "fully-Clifford" (rz(a)/rz(-a)
  padding makes them non-Clifford; errors.log confirms).
* Smoke report §5 pools scales over unequal device sets (x1 = Manila+Lagos,
  x1.5 = Lagos only) — confounded there; fine for the symmetric research
  config, but the section should print per-scale device composition.

## Bottom line

Launch `configs/research.yaml` as planned. Before any paper claim: (1) retrain
on seed-averaged labels via merge-back; (2) add leave-one-device-out beside
LOBO; (3) write the three disclosures (conditioned circuit ensemble, Lagos cap
compression, x1.0 model-family switch) into report/paper text.

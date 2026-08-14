# QEM-Selector — Research-Sweep Insight Analysis

Written 2026-07-22 by the insight-analyst pass, after the full research sweep
completed. Pure pandas/matplotlib post-processing — no model training here (the
trainer pass owns that). Every number below was computed directly from the CSVs;
nothing is taken from agent logs.

**Data:**
- `results\research\results.csv` — 1620 per-seed rows (180 circuits x 9 noise
  environments; 3 devices {FakeManilaV2, FakeJakartaV2, FakeLagosV2} x 3 nominal
  scales {1.0, 1.5, 2.0}; 5 techniques raw / raw_plus / zne / cdr / rem; 4096
  base shots; 3 seeds per config).
- `results\research\aggregated.csv` — 540 seed-averaged groups.
- `results\research\errors.log` — 571 refusal lines (all intentional; see section 4).
- `results\hw_first_run\results.csv` — 3 ibm_marrakesh (Heron) rows, 1024 shots,
  techniques raw / zne / rem only (used in section 6).

**Integrity checks passed:** recomputing `argmin` over the five `*_abs_error`
columns reproduces `best_technique` on 1620/1620 rows (0 mismatches); the
NaN counts in results.csv (cdr 415, rem 156) exactly match the errors.log
refusal counts by technique.

Winner tallies for reference —
per-seed `best_technique`: cdr 1008 / rem 485 / zne 78 / raw_plus 37 / raw 12;
per-seed `best_technique_cost_aware`: cdr 789 / rem 509 / raw 272 / zne 50;
seed-averaged `best_technique`: cdr 321 / rem 172 / zne 25 / raw_plus 16 / raw 6;
seed-averaged cost-aware: cdr 236 / rem 189 / raw 109 / zne 6.

Figures produced by this pass live in `results\research\figs\`:
`winner_share_vs_scale.png`, `error_vs_scale.png`, `zne_win_region.png`.

---

## 1. Winner vs noise scale — does CDR's share grow with noise?

**Yes.** Per-seed `best_technique` win share (% of 540 rows per scale, pooled
over the 3 devices):

| scale | cdr | rem | zne | raw_plus | raw |
|---|---|---|---|---|---|
| 1.0 | 59.3 | 33.7 | 5.2 | 1.9 | 0.0 |
| 1.5 | 61.7 | 30.7 | 4.6 | 2.0 | 0.9 |
| 2.0 | 65.7 | 25.4 | 4.6 | 3.0 | 1.3 |

CDR gains +6.4 pp from x1.0 to x2.0, taken almost entirely from REM (-8.3 pp).
Among rows where CDR actually competes (its refusals removed, section 4) the
trend is steeper: **79.6% -> 83.0% -> 88.3%**.

Per device x scale (counts, 180 rows per cell):

| device | scale | cdr | rem | zne | raw_plus | raw |
|---|---|---|---|---|---|---|
| FakeJakartaV2 | 1.0 | 115 | 50 | 15 | 0 | 0 |
| FakeJakartaV2 | 1.5 | 122 | 47 | 11 | 0 | 0 |
| FakeJakartaV2 | 2.0 | 131 | 44 | 5 | 0 | 0 |
| FakeLagosV2 | 1.0 | 89 | 74 | 7 | 10 | 0 |
| FakeLagosV2 | 1.5 | 88 | 63 | 13 | 11 | 5 |
| FakeLagosV2 | 2.0 | 100 | 37 | 20 | 16 | 7 |
| FakeManilaV2 | 1.0 | 116 | 58 | 6 | 0 | 0 |
| FakeManilaV2 | 1.5 | 123 | 56 | 1 | 0 | 0 |
| FakeManilaV2 | 2.0 | 124 | 56 | 0 | 0 | 0 |

**Does raw/raw_plus vanish with more noise? No — the opposite, and it is a
Lagos-only menu artifact.** Combined raw+raw_plus share grows 1.9% -> 3.0% ->
4.3% with scale, but all 49 of those wins sit on FakeLagosV2; on Manila and
Jakarta raw/raw_plus win **0 of 1080 rows**. See section 3 for why (they win by
default when CDR and/or REM refuse on the ultra-noisy readout device).

Cost-aware label, pooled per scale (%):

| scale | cdr | rem | raw | zne |
|---|---|---|---|---|
| 1.0 | 47.6 | 34.4 | 13.3 | 4.6 |
| 1.5 | 48.7 | 31.5 | 16.7 | 3.1 |
| 2.0 | 49.8 | 28.3 | 20.4 | 1.5 |

Under the sqrt shot-cost penalty, raw's share grows with noise — driven by
Lagos, where cost-aware raw goes 31.7% -> 40.6% -> 49.4% (Manila 1.7 -> 5.0,
Jakarta flat 6.7). On Lagos at high noise, mitigation's accuracy edge often no
longer pays for its shot multiplier. Cost-aware zne nearly dies (4.6% -> 1.5%);
raw_plus is structurally near-unwinnable under this label (0 wins; known
property, PROJECT_STATUS section 4.6).

**Caveat that must accompany any Lagos-vs-scale claim** (review-science Q6a):
the 0.45 readout cap compresses Lagos's dial. Realized backend features:

| device | scale | avg 2q error | avg readout error |
|---|---|---|---|
| FakeJakartaV2 | 1.0 / 1.5 / 2.0 | 0.0092 / 0.0138 / 0.0185 | 0.0307 / 0.0460 / 0.0614 |
| FakeManilaV2 | 1.0 / 1.5 / 2.0 | 0.0100 / 0.0149 / 0.0199 | 0.0373 / 0.0560 / 0.0746 |
| FakeLagosV2 | 1.0 / 1.5 / 2.0 | 0.0146 / 0.0220 / 0.0293 | 0.2035 / 0.2599 / 0.2932 |

2q errors scale exactly everywhere; Manila/Jakarta readout scales exactly;
Lagos readout realizes only x1.28 / x1.44 at nominal x1.5 / x2.0.

> Figure: `figs\winner_share_vs_scale.png` (stacked shares per device x scale).

---

## 2. The ZNE-win region — 78 per-seed wins characterized

Breakdown of the 78 `best_technique == zne` rows:

- **By family:** near_clifford 28, layered_random 17, ghz_plus 17,
  hw_efficient_ansatz 15, mirror_circuit 1.
- **By depth:** d4 18, d8 36, d16 24 — 77% of wins at depth >= 8.
- **By device x scale:** Jakarta 15/11/5 (falling with scale),
  Manila 6/1/0 (falling to zero), Lagos 7/13/20 (rising — see artifact below).
- **Menu context:** on 33/78 wins CDR had refused, on 33/78 REM had refused
  (18 both). Only **30/78 wins beat the full 5-technique menu**.

**The full-menu wins are the physically meaningful ZNE region**, and they
concentrate at moderate noise x deeper circuits: Jakarta x1.0 (11) + x1.5 (6),
Manila x1.0 (6), Lagos x1.0/x1.5 (5), and 28/30 at depth 8 or 16 (only 2 at
depth 4). By family: hw_efficient_ansatz 10, layered_random 9, near_clifford 7,
ghz_plus 3, mirror 1.

**The apparent Lagos surge (7 -> 13 -> 20 with scale) is a refusal artifact:
all 20 Lagos@x2.0 ZNE wins occurred on rows where REM had refused** (10 of them
also CDR-refused). ZNE is winning a depleted menu there, not beating REM.

Feature contrast (zne-win rows vs the other 1542 rows):

| feature | zne-win mean / median | rest mean / median |
|---|---|---|
| feat_clifford_fraction | 0.617 / 0.789 | 0.496 / 0.274 |
| feat_backend_avg_readout_error | 0.157 / 0.204 | 0.116 / 0.061 |
| feat_depth | 20.0 / 16 | 20.0 / 16 |
| ideal (signed) | 0.248 / 0.387 | 0.429 / 0.703 |

The high clifford_fraction and high readout error on win rows are largely the
*menu* signature (high Clifford fraction -> CDR refuses; high readout + high
scale -> REM refuses on Lagos), which is exactly why the selector can learn
"pick ZNE" from static features. Win margins are thin: median 0.011 (mean
0.016) over the runner-up, and ZNE's own error on its win rows is large
(median 0.225) — ZNE wins where everything is bad and it is least bad.

**Qualitative check against theory (RESEARCH_ANGLES angle 3 — ZNE should help
at moderate noise x enough depth, hurt at low noise / shallow depth).
Direction confirmed on the sim side:**

- ZNE beats raw on 79.9% of all 1620 rows (mean improvement +0.048), but the
  *depth* gradient goes the predicted way: mean improvement 0.029 / 0.050 /
  0.065 at depth 4 / 8 / 16, and ZNE is *worse than raw* on 28.5% of depth-4
  rows vs 14.4% / 15.2% at depth 8 / 16. Shallow circuits are ZNE's bad region.
- Noise-character gradient: on the readout-dominated Lagos, folding-ZNE (which
  amplifies gate noise, not readout) barely helps — mean improvement +0.006 and
  worse-than-raw on **41.1%** of rows, vs 8.0% (Jakarta) / 9.1% (Manila).
- Winner-level: ZNE's genuine (full-menu) wins die out toward high noise
  (Jakarta 15 -> 5, Manila 6 -> 0 as scale grows) — with the shot budget fixed,
  scaling noise pushes past ZNE's sweet spot, consistent with the finite-shot
  help-harm picture (Scavino 2605.08251). The x1.0 fake backends already sit at
  "moderate" noise; the truly-low-noise side of the boundary appears only on
  real Heron hardware, where ZNE lost on 3/3 circuits (section 6).

This is the sim-side *preview* of the boundary story only — the full overlay
needs a shot-budget axis, which this sweep (fixed 4096 shots) does not vary.

Seed-averaged labels tell the same story with less noise: 25/540 aggregated
groups pick zne, 20 of them on Jakarta@x1.0 or Lagos (13 on Lagos at
x1.5/x2.0 — again REM-refusal territory), and 12/25 in near_clifford.

> Figure: `figs\zne_win_region.png` (depth x backend 2q error, ZNE wins
> highlighted).

---

## 3. raw_plus — does 11x-shots-raw ever beat mitigation?

**Almost never on merit.** raw_plus wins 37/1620 per-seed rows (2.3%), raw
12/1620 (0.7%). Every one of those 49 wins is on FakeLagosV2:

- Of the 37 raw_plus wins: REM had refused on 18, CDR on 22, both on 13.
- Of the 12 raw wins (all Lagos @x1.5/@x2.0): REM had refused on 11, CDR on 8.
- Mean raw error on raw_plus-win rows is 0.758 — these are rows where *every*
  technique is terrible and the mitigators either refused or degraded.
- raw_plus's median win margin over the best available alternative: 0.0115.

Head-to-head across all rows where the opponent is valid: raw_plus beats CDR on
**5.6%** (67/1205), REM on **6.4%** (93/1464), ZNE on 21.6% (350/1620).

**Bias vs variance:** the paired difference raw_plus - raw over all 1620 rows is
mean **-0.0003 +/- 0.0138** (median +0.0001), and raw_plus is better than raw on
only **49.7%** of rows — a coin flip. Multiplying shots by 11 buys essentially
nothing, so raw's error is **bias** (systematic noise-induced offset), not
variance (shot noise). That is precisely why real mitigation — which attacks the
bias — beats an equal-budget shot-averaging control: the 11x-shots control
cannot touch what makes raw wrong. This is the empirical justification for
counting CDR's wins as genuine even at its 11x shot cost.

---

## 4. Technique error magnitudes and the refusal accounting

Mean / median |error| per technique per scale (per-seed rows):

| technique | x1.0 mean/med | x1.5 mean/med | x2.0 mean/med | n_valid per scale |
|---|---|---|---|---|
| raw | 0.366 / 0.290 | 0.421 / 0.330 | 0.475 / 0.367 | 540 / 540 / 540 |
| raw_plus | 0.365 / 0.287 | 0.420 / 0.334 | 0.476 / 0.367 | 540 / 540 / 540 |
| zne | 0.318 / 0.238 | 0.373 / 0.295 | 0.427 / 0.335 | 540 / 540 / 540 |
| cdr | 0.084 / 0.023 | 0.088 / 0.022 | 0.096 / 0.030 | 402 / 401 / 402 |
| rem | 0.199 / 0.107 | 0.198 / 0.113 | 0.191 / 0.135 | 532 / 494 / 438 |

Pooled means: raw 0.4205, raw_plus 0.4202, zne 0.3728, cdr 0.0893 (median
0.0249), rem 0.1963. On their own valid rows the error-reduction factors vs raw
are **cdr 4.3x** (0.383 -> 0.089), **rem 2.0x**, **zne 1.1x**. Note the scale
robustness: raw's mean error grows +30% from x1.0 to x2.0 while CDR's grows
only +14% (0.084 -> 0.096) — CDR's learned correction keeps absorbing most of
the added noise, which is *why* its win share grows with scale (section 1).
REM's apparent flat/declining mean at higher scales is a survivor effect: its
hardest Lagos rows progressively refuse and leave the valid set (below).

Per-family mean |error| (pooled scales):

| family | raw | raw_plus | zne | cdr | rem |
|---|---|---|---|---|---|
| ghz_plus | 0.551 | 0.552 | 0.490 | 0.050 | 0.200 |
| hw_efficient_ansatz | 0.244 | 0.241 | 0.212 | 0.096 | 0.172 |
| layered_random | 0.250 | 0.249 | 0.222 | 0.105 | 0.159 |
| mirror_circuit | 0.595 | 0.595 | 0.522 | 0.078 | 0.262 |
| near_clifford | 0.464 | 0.465 | 0.417 | 0.092 | 0.188 |

**Refusal accounting (errors.log, 571 lines, all intentional):**

- **CDR: 415 refusals** = 46 circuits per environment x 9 environments (+1
  extra runtime refusal on Jakarta@x1.5). Perfectly structural: by family
  ghz_plus 225, near_clifford 190, zero elsewhere; by reason 253
  degenerate-training-spread ("all N training circuits have the same ideal
  value") + 162 fully-Clifford-after-compilation. Backend-independent by
  design (the guards run on the circuit), matching the pre-run prediction of
  1206/1620 valid (realized: 1205).
- **REM: 156 refusals, all on FakeLagosV2** — 8 / 46 / 102 at x1.0 / x1.5 /
  x2.0, spread across all 5 families (27-38 each). The calibrated readout
  damping approaches zero as Lagos's already-extreme readout error is scaled
  up, so the inversion refuses (`REM_MIN_DAMPING = 0.02`). This is why REM's
  win share collapses on Lagos at x2.0 (74 -> 37 wins, section 1) and why
  ZNE / raw / raw_plus "wins" surge there.

**Is CDR's 62.2% win share inflated by refusals shrinking its denominator?
No — the opposite.** Refusals can only remove rows CDR might have won; they
never add wins. Computed both ways:

- Overall: CDR wins 1008/1620 = **62.2%**.
- Among the 1205 rows where CDR competes: 1008/1205 = **83.7%**
  (79.6% / 83.0% / 88.3% by scale).
- On the 415 CDR-refused rows the winner is rem 352 / zne 33 / raw_plus 22 /
  raw 8 — i.e. the refused rows mostly pad *REM's* count.
- Where CDR competes and loses (197 rows): rem 133 / zne 45 / raw_plus 15 /
  raw 4.

Same check for REM: valid on 1464 rows (90.4%); win share 29.9% overall, 33.1%
among valid.

The honest framing for the paper: "CDR wins 62% of all configurations and 84%
of the configurations it accepts; it refuses 26% of configurations by design
(Clifford-degenerate circuits), where REM is the usual winner."

> Figure: `figs\error_vs_scale.png` (mean |error| vs scale per technique).

---

## 5. Seed-flip rate — label noise, measured (reviewer major #1)

For each (family, n_qubits, depth, backend) group, compare each per-seed winner
(results.csv) with the seed-averaged group winner (aggregated.csv):

| label | per-seed rows disagreeing | groups with unanimous seeds |
|---|---|---|
| best_technique | **338/1620 = 20.9%** | 313/540 = 58.0% |
| best_technique_cost_aware | **350/1620 = 21.6%** | 285/540 = 52.8% |

So roughly **one in five per-seed labels disagrees with the seed-averaged label
of its own group** — this is the label noise the merge-back training scheme
removes, and it is in line with the 28.9% measured on the 45-row smoke run (the
research-scale estimate is the reliable one).

- Flip rate is scale-independent (21.5 / 20.7 / 20.4% at x1.0/x1.5/x2.0) —
  label noise does not shrink at high noise.
- Strongly family-dependent: ghz_plus 32.4%, near_clifford 26.9%,
  layered_random 21.9%, hw_efficient_ansatz 17.9%, **mirror_circuit 5.2%**.
  (Cost-aware: 28.7 / 25.3 / 20.1 / 25.9 / 8.0.)
- Dominant confusion is cdr <-> rem: of the 338 flips, per-seed cdr vs group
  rem 117, per-seed rem vs group cdr 72; all zne-involved pairs together 76.
- The flips are genuine seed variation, not rounding ties: on flipped rows the
  group winner's error on that seed is worse than the per-seed winner's by a
  median of 0.063 (mean 0.147).
- **The smoke-run concern about partial-coverage winners did not materialize:
  0 of 540 aggregated winners come from a technique with fewer valid seeds
  than the group's seed count** (the coverage rule + research-scale refusal
  structure made every group winner full-coverage).

Implication: per-seed-label training metrics inherit ~21% label noise;
seed-averaged labels (merge-back onto per-seed feature rows) are the right
headline target, exactly as the science review demanded.

---

## 6. Mirror-family hardware bridge — sim vs ibm_marrakesh (n=3, preliminary)

The 3 hardware rows (Heron, 1024 shots, menu raw/zne/rem) vs the same
`circuit_id`s on the three plain (x1.0) sim devices, with the sim winner
recomputed over the same 3-technique menu for a fair comparison:

| circuit | hw raw err | sim raw err (mean of 3 devices) | ratio | hw winner | sim winner (Manila/Lagos/Jakarta) | agree? |
|---|---|---|---|---|---|---|
| mirror_circuit_q2_d4_s0 | 0.0273 | 0.2376 | x8.7 | rem | rem / rem / rem | yes 3/3 |
| mirror_circuit_q3_d4_s0 | 0.0312 | 0.4793 | x15.3 | rem | rem / rem / zne | yes 2/3 |
| layered_random_q2_d4_s0 | 0.0160 | 0.1684 | x10.5 | raw | rem / rem / rem | **no 0/3** |

Findings:

- **Raw-error gap: the fake-backend sims are 8.7x-15.3x noisier than the real
  Heron device** on identical circuits (hw raw 0.016-0.031 vs sim 0.168-0.479).
  Expected — the fake backends are Falcon-generation calibration snapshots,
  Heron r2 is a far cleaner processor (LITERATURE section 4, McKay EPLG row).
  Any sim-to-real transfer claim must acknowledge the selector was trained in a
  noise regime an order of magnitude harsher than the target device.
- **Winner agreement 2/3 on the mirror family; the miss is exactly the
  out-of-distribution low-noise regime.** On hardware, layered_random's raw
  error (0.016) is already so small that REM's calibration noise makes it
  *worse* (0.046); in the sim grid raw is never that good, so sim always says
  rem. The disagreement is not a random error — it is the regime gap itself,
  and it is the sim-side reason to expect the "raw/do-nothing region" to
  matter on clean hardware (ties into the help-harm boundary story).
- **ZNE was worse than raw on 3/3 hardware circuits** (e.g. layered_random
  0.016 -> 0.260), consistent with the sim finding that folding-ZNE at a low
  fixed shot budget loses at low depth/low noise, and with published
  hardware ZNE regressions (Russo 2023, Koster 2026 — LITERATURE section 4).
- Sample caveat: n=3 circuits, one device, one seed, 1024 shots — motivating
  preliminary evidence only, per the gap list (LITERATURE section 5.2).

---

## 7. Disclosures these results inherit

1. **Lagos readout cap compression** (section 1 realized-features table):
   Lagos's readout dial realizes x1.28/x1.44, not x1.5/x2.0; its 2q dial is
   exact. REM-vs-scale trends on Lagos mix cap artifacts with physics; prefer
   plotting against realized rates.
2. **x1.0 vs scaled noise-model family switch**: plain backends run
   `from_backend` composite channels; scaled variants run synthetic
   depolarizing+readout. The 1.0 -> 1.5 step changes noise character, not just
   strength (ZNE is the most sensitive to this).
3. **Conditioned circuit ensembles**: all families are conditioned on
   |ideal| >= 0.25; random families keep an atypically non-scrambling subset.
   Results do not transfer to typical scrambling circuits.
4. **argmin labels**: winners are argmin-|error| labels; near-ties count as
   decisions (Decision Kernels caveat, LITERATURE section 5.5). Section 5's
   flip-gap numbers quantify how non-tied the typical flip actually is.

---

## Appendix: reproducibility

All tables derive from `results\research\results.csv`,
`results\research\aggregated.csv`, `results\research\errors.log`, and
`results\hw_first_run\results.csv` with pandas only. Key operations:
`device`/`scale` parsed from `backend` by splitting on `@x`; winner shares are
`groupby(["device","scale"])["best_technique"].value_counts(normalize=True)`;
CDR-valid = `cdr_abs_error.notna()`; seed-flip = merge per-seed rows with
aggregated winners on (family, n_qubits, depth, backend) and compare labels;
hardware bridge recomputes the sim winner as argmin over
{raw,zne,rem}_abs_error only. Figures: `results\research\figs\*.png`.

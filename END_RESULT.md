# QEM-Selector — End Result (research run, 2026-07-22)

This is the plain-language summary of what the project found. Every number below
was produced by the pipeline, cross-checked by an independent analysis pass, and
then re-reproduced from scratch by a separate verifier agent (fresh code, zero
discrepancies — see `notes/final-verifier.md`). Full tables live in
`docs/ANALYSIS.md`; model metrics in `results/research/metrics.json` and
`results/research/metrics_cost_aware.json` (per-seed ablation in
`results/research/per_seed/`).

---

## 1. What the experiment was

Quantum computers are noisy, and there are several competing "error mitigation"
techniques that try to clean up the answer after the fact. The field agrees that
none of them is universally best — which one wins depends on the circuit, the
device, and the shot budget. This project asked: **can a plain ML classifier,
looking only at cheap static features (circuit shape + device calibration
numbers, no quantum execution at decision time), predict which technique will
win?** To find out, we built a benchmark: 180 small circuits (5 families, 2–5
qubits, depths 4/8/16) were each run in 9 simulated noise environments (3 IBM
fake devices — Manila, Jakarta, Lagos — each at noise scale x1.0 / x1.5 / x2.0),
with 3 random seeds and 4096 base shots: **1620 experiment units**. On every
unit, five strategies were run: `raw` (do nothing), `raw_plus` (do nothing at
11x shots — the equal-budget control), `zne`, `cdr`, and `rem`. Each result was
compared against the exact noise-free answer; the technique with the lowest
error became the label. Two label variants: `best_technique` (pure accuracy)
and a cost-aware variant that penalizes shot cost.

On top of that dataset we trained the selector (RandomForest and
GradientBoosting; the better of the two by cross-validated macro-F1) and
evaluated it the honest way: grouped cross-validation (seed-duplicates can never
sit in both train and test), plus three held-out generalization tests — **LOFO**
(leave one circuit family out: can it handle a new kind of circuit?), **LOBO**
(leave one noise scale out: interpolation), and **LODO** (leave one entire
device out, all its scales at once: a genuinely new noise environment). The
headline model trains on **seed-averaged labels** (540 groups); a per-seed
ablation (1620 rows) quantifies how much that averaging matters. Techniques
that would produce garbage refuse loudly instead of silently "winning" (CDR on
fully-Clifford or degenerate circuits, REM on near-singular readout
inversions), and every refusal is logged and accounted for. A small n=3
real-hardware run on ibm_marrakesh (Heron) bridges to reality.

---

## 2. The findings

**F1. The best mitigation technique is not fixed — it shifts predictably with
noise level, device, and circuit type, which is exactly the signal a selector
needs.** CDR's win share grows 59.3% → 61.7% → 65.7% as noise scales x1.0 →
x1.5 → x2.0, taken almost entirely from REM (33.7% → 25.4%). The mechanism:
raw error grows 0.366 → 0.475 with the noise dial, while CDR's grows only
0.084 → 0.096 — CDR is the noise-robust technique. On the readout-dominated
device (Lagos), the cost-aware label tells a different story: "do nothing" wins
31.7% → 49.4% of rows at high noise, because mitigation stops paying for its
extra shots there.

**F2. A classifier that never runs a circuit predicts the winning technique
well above baseline — including on circuit families and devices it has never
seen.** Headline (seed-averaged, 540 rows): `best_technique` grouped 5-fold CV
accuracy **0.796 ± 0.053** (macro-F1 0.417) vs majority baseline 0.594;
**LOFO 0.787** (new family), LOBO 0.893 (scale interpolation), **LODO 0.865**
(new device; folds Jakarta 0.967 / Lagos 0.694 / Manila 0.933). Cost-aware
label: CV **0.728 ± 0.133** (F1 0.583) vs baseline 0.437; LOFO 0.702; LOBO
0.783; LODO 0.704 (weakest single number in the project: the Lagos fold at
0.422). No class had to be dropped in any run. Every one of these numbers was
independently re-derived from scratch and matched to 4 decimals.

**F3. Per-seed winner labels are ~21% noise; seed-averaging removes it and
improves every held-out metric.** The per-seed winner disagrees with its
group's seed-averaged winner on 20.9% (best) / 21.6% (cost-aware) of the 1620
rows; only 58.0% / 52.8% of groups are unanimous across seeds. The flips are
genuine (median error gap 0.063 on the flipped seed — not ties), family-
dependent (ghz_plus 32.4% down to mirror 5.2%), and mostly cdr↔rem confusion.
Training on seed-averaged labels beats per-seed training on every held-out
metric: LOFO +0.075 / +0.104, LODO +0.049 / +0.016 (best / cost-aware). And
0/540 aggregated winners rest on partial seed coverage.

**F4. ZNE only wins in a narrow moderate-noise, deeper-circuit band — and its
failure pattern points the same direction as the analytic help-harm theory.**
Of 78 per-seed ZNE wins, only 30 beat the full 5-technique menu (the rest are
wins-by-forfeit where CDR/REM refused); 28/30 of the real wins sit at depth
8/16, concentrated at moderate noise (Jakarta x1.0/x1.5, Manila x1.0). ZNE wins
die out as noise rises on clean devices (Jakarta 15 → 11 → 5, Manila 6 → 1 → 0),
and the apparent Lagos@x2.0 "surge" (20 wins) is 100% refusal artifact. The
theory-consistent gradient: ZNE is worse than raw on 28.5% of depth-4 rows vs
~15% at depth 8/16, and on readout-dominated Lagos its mean improvement is a
negligible +0.006 (worse than raw on 41.1% of rows, vs 8–9% on the other
devices). On real hardware ZNE lost to raw on 3/3 circuits. Caveat: shots were
fixed at 4096, so this is the sim-side preview of the boundary, not the full
(noise × shots) map.

**F5. Throwing 11x more shots at the problem does nothing — raw's error is
bias, not shot noise — so mitigation's wins survive the equal-budget control.**
`raw_plus` (raw at 11x shots, matching CDR's budget) beats plain raw on only
49.7% of rows — a coin flip (paired difference −0.0003 ± 0.0138). Do-nothing
strategies never win on Manila or Jakarta (0/1080 rows); all 49 raw/raw_plus
wins are on ultra-noisy Lagos and mostly menu artifacts (on most of those rows
CDR and/or REM had refused).

**F6. CDR is the dominant technique, and honest refusal accounting makes it
look better, not worse.** Pooled mean |error|: raw 0.4205 / raw_plus 0.4202 /
zne 0.3728 / rem 0.1963 / **cdr 0.0893** (median 0.0249) — error-reduction
factors on rows each technique accepts: cdr 4.3x, rem 2.0x, zne 1.1x. CDR's
overall win share is 62.2%, but among the 1205 rows it accepts it wins
**83.7%** (79.6 / 83.0 / 88.3% by scale). Its 415 refusals are structural, not
random: fully-Clifford or degenerate-training circuits (ghz_plus 225 +
near_clifford 190, backend-independent), and those rows go mostly to REM (352).
REM's own 156 refusals are all on Lagos (8/46/102 by scale) — a survivor effect
that flattens its apparent per-scale error.

**F7. The fake-backend simulations are ~9–15x noisier than today's real
hardware, and the selector's one hardware miss is exactly the low-noise regime
the sim grid never covers.** On 3 identical circuits, ibm_marrakesh (Heron) raw
error was 0.016–0.031 vs sim means 0.168–0.479 (ratios 8.7x / 15.3x / 10.5x).
Winner agreement 2/3: both mirror circuits agree on rem; the miss is
layered_random, where hardware raw wins (0.016) and REM makes it worse (0.046)
while the sim says rem — a regime (near-noiseless) that simply does not exist
in the simulated grid. n=3 is an anecdote, not a study — but it is exactly the
right anecdote to motivate the hardware boundary test below.

---

## 3. What this means for the paper

The paper plan is one paper (see `docs/RESEARCH_ANGLES.md`): the execution-free
selector is the contribution, Angle 3 (learned ZNE-refusal vs. the analytic
help-harm boundary) is the intellectual headline, Angle 2 (CDR regressor choice
as a selectable technique) is a supporting sim-only section.

- **The core selector claim** is now fully loaded: F2 provides the CV / LOFO /
  LODO numbers for both labels vs. baselines, F3 provides the label-noise
  methodology result (seed-averaging as a protocol contribution), F5 provides
  the equal-budget control reviewers will demand, and F6 provides the honest
  refusal-accounting story. Quote LODO (not LOBO) for "new noise environment"
  claims, and always give the baseline next to the accuracy.
- **Angle 3 (headline)** is fed by F4: at fixed shots, the selector's data
  already shows ZNE's win region collapsing exactly where the finite-shot
  help-harm theory (Scavino, arXiv:2605.08251) says it should — with depth,
  device noise character, and noise scale. What is missing is the shot-budget
  axis: the paper's centerpiece overlay needs a (noise × shots) sweep in sim
  (compute the analytic ΔMSE = 0 curve, plot the selector's learned ZNE-refusal
  region on the same axes), with ZNE reconfigured to Scavino's fixed-Richardson
  variant first — otherwise the comparison is apples-to-oranges.
- **The ~9.3 remaining free QPU-minutes** should buy Angle 3's hardware half,
  per RESEARCH_ANGLES: known-answer circuits (mirror / near-Clifford, where the
  ideal value is known) at 2–3 shot budgets on Heron, measuring the empirical
  MSE crossing between ZNE and raw, overlaid with the sim-anchored analytic
  curve and the selector's refusal region. F7 makes this doubly valuable: real
  Heron hardware sits in the low-noise regime the sim grid misses, so the
  boundary test simultaneously patches the selector's known blind spot. Scope
  it as motivating preliminary evidence, one device.
- **Angle 2** is fed by F6 and F1: CDR's dominance and noise-robustness is the
  motivation for caring about its regressor at all, and its structural refusal
  map defines where a regressor swap is even applicable. The novel figure — the
  2D (training-set size × non-Clifford fraction) linear-vs-nonlinear crossover
  heatmap — is pure simulation, zero QPU cost, anchored to Korolev's
  "regularized-linear-usually-wins" result (arXiv:2606.02697). Do not spend QPU
  minutes here.
- **F7 itself** is the sim-to-real transfer subsection ("first sim-to-real
  transfer test of a *selection policy*", per `docs/LITERATURE.md` §3) —
  presented as n=3 motivating evidence, never as a validated transfer study.

---

## 4. Honest limitations

1. **Everything except 3 circuits is simulated noise.** The "devices" are Aer
   simulators loaded with calibration snapshots of retired Falcon-era IBM
   machines; they capture error rates and topology but not drift, crosstalk, or
   non-Markovian effects — and F7 shows they are ~9–15x noisier than a current
   Heron device. Every claim in §2 except F7 is a claim about simulated noise.
2. **The labels are argmin-|error|.** "Best technique" means lowest absolute
   error on one observable at one shot budget. Decision Kernels
   (arXiv:2607.02888) shows accuracy gains need not improve downstream
   decisions — small error gaps can be decision-flat. Disclosed as a limitation;
   significance-aware tie labels remain future work.
3. **The noise dial is a controlled approximation, not physics.** Scaled
   variants replace the composite x1.0 noise model with synthetic
   depolarizing + readout channels (character changes at the first scaling
   step), and the 0.45 readout cap compresses Lagos to realized ~x1.28 / x1.44
   at nominal x1.5 / x2.0. Trends on Lagos must always carry this caveat; quote
   realized rates, not nominal scales.
4. **The hardware bridge is n=3 circuits, one device, 1024 shots.** Motivating
   preliminary evidence only. ZNE there was off-the-shelf folding-based dZNE —
   the variant IBM itself has moved away from on Heron — so the ZNE-negative is
   variant-specific, not a verdict on ZNE.
5. Further disclosures the paper must carry: circuit families are *conditioned*
   ensembles (|ideal| ≥ 0.25 rejection sampling — results do not transfer to
   fully scrambling circuits); features are angle-blind (seed variants share
   feature vectors — handled by grouped CV); the reported model is the better
   of two picked on the same CV (mild winner's curse); macro-F1 is much lower
   than accuracy (0.417 on the headline run) because minority classes (zne,
   raw) are genuinely hard to predict.

---

## 5. Your next 5 actions

1. **Draft the paper skeleton** following `docs/RESEARCH_ANGLES.md` structure
   (selector = contribution; Angle 2 = sim section; Angle 3 = validation
   headline), pulling numbers only from `results/research/metrics*.json` and
   `docs/ANALYSIS.md`, and wording claims strictly per `docs/LITERATURE.md` §3
   (the DO-NOT-CLAIM list is a hard rule set).
2. **Build the Angle 3 sim overlay:** extend the sweep with a shot-budget axis
   (e.g. 256 / 1024 / 4096 / 16384 shots) for zne-vs-raw, align the ZNE
   implementation to Scavino's fixed-Richardson variant, compute the analytic
   ΔMSE = 0 boundary, and overlay the selector's learned refusal region in the
   (noise × shots) plane. This is the paper's centerpiece figure.
3. **Spend the ~9.3 QPU-minutes on the hardware boundary test:** known-answer
   mirror / near-Clifford circuits at 2–3 shot budgets on a Heron device
   through the existing gated flow (`test_hardware_connection.py` →
   `estimate_hardware_cost.py` → `hardware_confirmed: true` → run), measuring
   the empirical ZNE help-harm crossing to overlay on the sim curve.
4. **Produce the Angle 2 crossover heatmap** (sim only, zero QPU): CDR
   linear-vs-nonlinear regressor error difference over (training-set size ×
   non-Clifford fraction), across the 5 families, reproducing Korolev's
   linear-wins anchor.
5. **Re-run the literature/novelty scan immediately before submission** (the
   three closest papers are Apr–Jul 2026) **and get an experienced eye on the
   methodology** — QOSF mentorship application or a university supervisor; this
   repo, with its verified negative controls, is the application artifact.

# Novelty scan — ANGLE 2 (CDR-regressor-as-selectable-technique) — "cdr-embed"

Run 2026-07-22 by novelty searcher "cdr-embed". Hostile-reviewer mode.
All arXiv IDs below were fetched and confirmed to resolve.

## The question
Angle 2 reframed: NOT "which regressor is best for CDR" (Korolev closed that).
Instead — add nonlinear-CDR as an additional selectable technique inside our
per-circuit QEM selector, and CHARACTERIZE WHEN the selector should pick
nonlinear-CDR vs linear-CDR, across 5 heterogeneous non-VQE families, on real
hardware, with a (train-size x non-Clifford-fraction) overfitting map, using
Korolev's "Ridge usually wins" as a sanity anchor.

Four sub-claims to test for scooping:
(a) treat CDR regressor CHOICE as a selection/decision problem
(b) benchmark CDR regressor variants across HETEROGENEOUS non-VQE families
(c) do CDR regressor swaps on REAL hardware
(d) map CDR overfitting vs training-set size / non-Clifford fraction

## VERDICT: PARTIALLY-COVERED (reframed angle survives; naive angle is closed)

The naive "swap CDR's linear fit for sklearn regressors and see which wins" is
CLOSED by Korolev (June 2026) and echoed by NIL. But every one of the four
reframe pillars has meaningful open space. No paper embeds the regressor choice
as a per-circuit SELECTION target; none does it on real hardware; none covers
heterogeneous NON-VQE families with a regressor comparison; and the 2D
overfitting map exists only as separate 1D slices for linear/quadratic models.

## The threats, by pillar

### Threat #1 — Korolev et al. 2026, arXiv:2606.02697 (CONFIRMED, sim-only)
"ML-Based QEM for Variational Algorithms." Benchmarks 7 regressors
(Ridge, Lasso, Random Forest, SVM, KNN, MLP, XGBoost) on near-Clifford training
data. THIS IS THE PAPER THAT CLOSES THE NAIVE ANGLE.
- Circuit families: ONLY TwoLocal ansatz (ring CNOT) for the Sherrington-
  Kirkpatrick Hamiltonian. n=12 main, n=6 for statistics. NOT heterogeneous, VQE-only.
- SIMULATION ONLY (Qiskit; depolarizing/Pauli/composite synthetic noise). No hardware.
- Headline: "Ridge regression ... generally achieves superior error mitigation";
  XGBoost only "occasionally outperform[s] Ridge" at high noise (p~0.1). Nonlinear
  advantage vanishes below ~5% error for the 12q setup.
- Non-Clifford fraction: tested at {0.2,0.4,0.6,0.8}; finding = "Ridge regression
  performance does not depend significantly on the fraction of non-Clifford gates."
- Training-set size: FIXED at 1500 samples. NO learning curves, NO overfitting-vs-N.
=> Scoops: regressor benchmarking + "Ridge usually wins" (use as sanity anchor).
=> Leaves open: selection framing, heterogeneous non-VQE, hardware, train-size scaling,
   and the 2D (N x k) overfitting map for the REGRESSOR CHOICE specifically.

### Threat #2 — NIL, Chen et al. 2025, arXiv:2512.12578 (CONFIRMED, sim-only)
"Scalable QEM with Neighbor-Informed Learning." Compares linear / Lasso / neural
network. NN "did not show an advantage over linear models." Settles on Lasso
PRAGMATICALLY ("best trade-off"), NOT as a per-circuit decision.
- Families: 1D/2D TFI, UCC (LiH, F2), hardware-efficient ansatz, up to 100q.
  Broader than Korolev but still structured/variational, NOT non-VQE-heterogeneous
  (no GHZ / QFT / random / mirror families).
- SIMULATION ONLY (stim Clifford simulators). No hardware.
- Scaling: theoretical O(ln N / eps^2) sample complexity; empirically monotone
  MSE decrease, "without apparent overfitting" — i.e. they argue overfitting is
  NOT a problem, they do not MAP where it becomes one.
=> Partially covers pillar (b) (multi-family regressor comparison) but sim-only and
   the regressor choice is an implementation detail, not a selection target.

### Threat #3 — Zhao et al. 2025, arXiv:2511.03556 (CONFIRMED, sim-only)
"Energy sampling + NCE enhanced CDR." Keeps linear + quadratic scikit-learn
LinearRegression (NO neural nets / ML zoo). NCE adds non-Clifford count k as an
extra regression input.
- DOES study BOTH training-set size (N~50 convergence) AND non-Clifford fraction k
  dependence — this is the closest hit on pillar (d).
- BUT: only linear/quadratic models (not the "when does nonlinear overfit" question),
  single H4 molecule / tUPS ansatz (VQE), SIMULATION ONLY (ibm_torino fake backend).
=> The individual axes of our overfitting map (N-scaling, k-dependence) are already
   plotted for linear/quadratic CDR on one VQE molecule in sim. Our contribution must
   be the 2D (N x k) map SPECIFICALLY for the nonlinear-vs-linear crossover, on
   heterogeneous families, on hardware. Cite Zhao so we don't claim N-scaling as new.

### Threat #4 — Scavino 2026, arXiv:2606.21686 (CONFIRMED, theory+sim)
"Finite-shot operating windows for PEC and CDR." Derives MSE boundaries giving a
"CDR-dominant operating window" (upper bound ~ 1/(delta1^2 p)) vs no-mitigation and
PEC, as a function of noise level and shot budget. QAOA + Pauli observables, closed-
form 2-qubit + sim. STANDARD LINEAR CDR ONLY; no regressor variants; maps vs
noise+shots, NOT train-size/non-Clifford-fraction; characterization, not selection.
=> Adjacent: shows "characterize when CDR dominates" is an active research shape
   (same author family as the ZNE help-harm boundary in Angle 3). A reviewer will
   ask how our regressor-choice map differs from an operating-window analysis —
   answer: ours is data-driven regressor SELECTION, theirs is analytic CDR-vs-PEC.

### Threat #5 — ML-QEM, Liao et al. 2024, arXiv:2309.17368 (Nat Mach Intell) — the hardware precedent
Benchmarks linear / random forest / MLP / GNN on DIVERSE circuit classes over
multiple device-noise profiles, on REAL IBM hardware up to 100q; RF best; ~2x
runtime saving. THIS IS THE STRONGEST ADJACENT HARDWARE PRECEDENT — but it learns
to MIMIC digital ZNE (predict the ZNE-mitigated value), it is NOT a CDR regressor
swap and does NOT use near-Clifford training data. RF-on-diverse-circuits-on-
hardware exists; RF-as-a-CDR-regressor-on-hardware does not.
=> A hostile reviewer WILL say "nonlinear ML regressors on diverse circuits on real
   IBM hardware is done (ML-QEM)." Our rebuttal: different training data
   (near-Clifford CDR, not ZNE-folds) and different task (selectable technique in a
   selector, not a standalone ZNE surrogate).

### Cleared sub-angle — Gaussian-process / kernel CDR: genuinely CLEAR
Searched explicitly. No near-Clifford CDR paper uses Gaussian-process or kernel
regression. (Hits were unrelated quantum-kernel GP or Decision-Kernels theory.)
If we want a defensible "new regressor" beyond Korolev's zoo, GP/kernel-CDR is open.

## What remains OPEN for Angle 2 (defensible contributions)
1. Regressor choice as a per-circuit SELECTION TARGET inside a technique selector —
   NOBODY does this. Korolev/NIL recommend one regressor globally. CLEAR.
2. Heterogeneous NON-VQE families (GHZ / QFT / random / mirror + one variational)
   with a CDR regressor comparison — Korolev is VQE-only, NIL is variational-only.
   Mostly OPEN.
3. CDR regressor swaps on REAL hardware — all regressor-comparison papers (Korolev,
   NIL, Zhao) are simulation-only. OPEN for CDR specifically (ML-QEM is the adjacent
   hardware precedent, but for ZNE-mimicry).
4. 2D (train-size x non-Clifford-fraction) overfitting/crossover map for the
   nonlinear-vs-linear decision — the two axes exist separately (Zhao N-scaling;
   Korolev k-insensitivity of Ridge) but the joint map, and specifically "when does
   nonlinear overfit vs help," is not plotted. OPEN.
5. Gaussian-process/kernel CDR regressor — nonexistent. OPEN (optional stretch).

## Required framing / citations (do not get desk-rejected)
- Reproduce Korolev's "Ridge usually wins / nonlinear only helps at high noise" as a
  sanity anchor; cite 2606.02697 prominently as the paper that closed the naive angle.
- Cite Zhao 2511.03556 for N~50 convergence and k-dependence so we don't claim
  training-set-size scaling as novel.
- Cite NIL 2512.12578 for multi-family regressor comparison (and NN-no-advantage).
- Cite Scavino 2606.21686 as the analytic CDR operating-window analysis and
  distinguish our data-driven regressor selection from it.
- Cite ML-QEM 2309.17368 as the RF-on-hardware precedent and state the two
  differences (near-Clifford training data; selection task).
- Do NOT claim "first to apply ML regressors to CDR / near-Clifford data" (FALSE,
  Korolev + NIL). Correct claim: "first to embed the CDR regressor choice as a
  selection target and characterize when nonlinear-CDR beats linear-CDR across
  heterogeneous non-VQE families and on real hardware."

# Deep Reader — Batch 3 (literature overlap check)

Date: 2026-07-22. Sources: arXiv abs + HTML/ar5iv full texts. Our contribution axes referenced below:
(a) learned per-circuit QEM technique selector w/ grouped-CV + LOFO eval, (b) noise-strength dimension via scaled backends,
(c) equal-budget raw_plus fairness baseline, (d) sim-to-real selector transfer test on 2026 Heron hardware, (e) CDR regressor swaps.

---

## 1. Zhao et al. 2025 — "QEM using energy sampling and extrapolation enhanced CDR" (arXiv:2511.03556, v3 Feb 2026)

**What they did.** VQE chemistry on the H4 molecule with a tiled-UPS ansatz, simulated with the ibm_torino (133q) noise
model in AerSimulator at the infinite-shot limit — NO real hardware, NO shot noise. Base CDR = OLS linear regression
f = a1*X_noisy + a2. Two variants: **ES-CDR** keeps only the lowest-energy training circuits (data selection, same linear
model); **NCE-CDR** adds the number of retained non-Clifford parameters k as a second regression input and upgrades to a
quadratic multivariate least-squares model f = a1*X^2 + a2*k^2 + a3*kX + a4*X + a5*k + a6.

**Key numbers.** 2-layer tUPS (18 non-Clifford params, k=4, up to N=153 training circuits, converges ~N=50): CDR ~0.15 Ha
error, ES-CDR ~0.10 Ha, NCE-CDR ~0.05 Ha. 3-layer: CDR ~0.40 Ha, ES ~0.28 Ha, NCE ~0.13 Ha. Chemical accuracy (1.6 mHa)
never reached. ES gives ~50% error cut at small training-set sizes.

**Overlap verdict.** Touches only (e), and only half-way: they change *inputs* (add k) and *data selection* (energy filter)
and go from linear to a hand-crafted quadratic polynomial — still least squares, no nonparametric/ensemble regressor (no
RF/GB), no cross-validation methodology, no comparison to ZNE or any other technique, no selection, no hardware, no shot
budget accounting (infinite shots!). Must-cite for our (e): it establishes that feature-augmented nonlinear regression
beats vanilla CDR, so our (e) claim should be phrased as "systematic swap to nonparametric ML regressors (RF/GB) under
finite equal budgets", not "first to go beyond linear CDR". Leaves (a)-(d) fully open.

---

## 2. Liao, Wang, Sitdikov, Salcedo, Seif, Minev — "Machine learning for practical QEM" (arXiv:2309.17368; Nat. Mach. Intell. 6, 1478-1486, 2024)

**What they did.** IBM's flagship ML-QEM. Trained classical models (linear regression, **random forest — consistently
best**, MLP, GNN) to map (noisy expectation value + circuit features: native-gate counts, #2q gates, observable as sparse
Pauli rep, optional device noise params) -> mitigated expectation value. Two regimes: (i) tractable circuits, trained on
exact ideal values; (ii) beyond-classical: **mimic digital ZNE** (gate folding, noise factors {1,3}, linear extrapolation),
so the RF reproduces ZNE's output at ~zero extra quantum cost at inference.

**Key numbers.** 100-qubit Trotterized 1D TFIM on ibm_brisbane, up to 1,980 CNOTs; RF-mimicking-ZNE matches ZNE accuracy
with >=2x lower runtime overhead. Training: 50 circuits/Trotter step (500 total) at tractable scale; 10/step at 100q.
Generalization: trained on <~2% of Pauli observables, transfers to the rest with lower error than ZNE; depth extrapolation
beyond training degrades under coherent noise; Appendix D shows MLP fine-tuning to a drifted noise profile needs ~300 extra
circuits.

**Overlap verdict.** Closest cousin, but orthogonal output: their model REPLACES a mitigation technique (predicts the
mitigated value); ours SELECTS among techniques (predicts which of raw/raw_plus/ZNE/CDR/REM wins). They never compare or
choose among techniques — ZNE is fixed as the mimicry target, and RF-mimicking-ZNE by construction cannot beat ZNE (they
say so). Mandatory citation for: RF-as-workhorse, circuit+backend feature engineering, and the cost-amortization pitch
(theirs is per-shot inference cost, ours is per-technique-execution cost — distinguish carefully). Their noise-drift
fine-tuning (App. D) is the nearest thing to our (d), but it is "retrain the mimic after drift", not "test whether a
selector trained on calibration snapshots transfers to real hardware". Leaves (a), (b), (c), (d), (e) open.

---

## 3. Chen et al. 2025 — "Scalable QEM with Neighbor-Informed Learning" (arXiv:2512.12578, Dec 2025)

**What they did.** Framework (NIL) predicting the ideal expectation value of a target circuit as a learned linear
combination of noisy outputs from "neighbor" circuits (structural variants: gate insertions, noise-amplified copies —
unifies ZNE/PEC/CDR-style constructions). Training-set innovation: instead of substituting each rotation R_P(theta) with
uniformly random Cliffords, use only {R_P(0), R_P(pi/2), R_P(pi), R_P(3pi/2)} — a "rotation 2-design" capturing 2nd-order
statistics. Combine function fitted with **Lasso** (they compared linear regression, Lasso, and neural nets; Lasso won on
accuracy/efficiency trade-off). Simulation only (depolarizing noise, 0.001 1q / 0.01 2q). Training set size scales
O(ln(N)/eps^2) in the number of neighbors N.

**Key numbers.** ~4 orders of magnitude better than random-Clifford training on LiH circuits; ~2 orders of magnitude error
reduction vs standard ZNE on VQE benchmarks; demonstrated at 100+ qubits, 20+ layers. No hardware.

**Overlap verdict.** Training-DATA innovation for the CDR family; partially touches (e) because they DID run a small
regressor comparison (linear vs Lasso vs NN) inside their framework — so "nobody has compared regressors in learning-based
QEM" is not a safe claim. However: their comparison is within NIL (not vanilla CDR), Lasso is still linear-in-features, no
RF/GB, no shot-budget fairness, no hardware, and no technique selection. Leaves (a)-(d) open and (e) mostly open (our
ensemble-regressor-in-vanilla-CDR-under-equal-budget angle survives, but cite this and phrase precisely).

---

## 4. Strikis, Qin, Chen, Benjamin, Li — "Learning-based QEM" (PRX Quantum 2, 040330, 2021; arXiv:2005.07601)

**What they did.** Learns **quasi-probability compensation coefficients** q(P) over Pauli-gate insertions, ab initio,
without an error model: run Clifford-substituted variants of the target circuit on hardware, compute their ideal values
classically (Gottesman-Knill), minimize squared error of the compensated output via least squares / gradient descent.
The learned object is a sampling distribution applied to the primary (non-Clifford) circuit — there is NO noisy->ideal
regression map and no per-circuit prediction model.

**Key numbers.** Needs |T| ~ 3x the number of significant error locations in training circuits (e.g., 255 Clifford
circuits for |SigE|=85). H2 VQE on ibmq_santiago: 5.6% error unmitigated -> 0.7% mitigated. 2q DQCp circuit improved on
three IBMQ devices (Yorktown, Ourense, Santiago). Emulated 8q/8-layer devices: 4-5x better than tomography-based PEC;
error-rescaling factor plateaus for >=9 qubits.

**Overlap verdict.** Ancestor/family citation only (origin of the Clifford-training-circuit idea alongside CDR). Zero
overlap with (a)-(d); no overlap with (e) since nothing resembling a regressor exists to swap. Standard related-work cite.

---

## 5. M. Liao, Zhu, Chiribella, Yang — "Noise-agnostic QEM with data augmented neural models" (DAEM) (arXiv:2311.01727; npj Quantum Info 11, 8, 2025)

**What they did.** Neural mitigation WITHOUT noise-free training labels: build "fiducial" circuits that ideally implement
identity (each 1q gate R replaced by sqrt(R)†·sqrt(R)-style canceling pairs; CNOTs kept) so labels come from measuring
known input states on the SAME noisy hardware at several artificially varied noise levels; an MLP (U-Net for CV/Wigner
functions) then extrapolates the target circuit's noiseless statistics. Per-circuit-skeleton training; transfers across
parameter values of the same skeleton without retraining.

**Key numbers.** OriginQ cloud hardware, 4q circuits (U3+CZ): average MAE — DAEM 0.067 vs ZNE 0.259 vs CDR 0.095. 4q TFIM
VQE simulations: beats ZNE and CDR under phase/amplitude damping and non-Markovian noise. CV Kerr dynamics: fidelity
~0.3 -> >0.9.

**Overlap verdict.** Adjacent neural-QEM related work; no technique selection, no scaled-noise feature dimension, no
budget-fairness baseline, no CDR regressor swap (it sidesteps CDR's classical-simulation step entirely). Useful supporting
data point for our hardware story: on their real device, plain ZNE was the WORST of the three (0.259) and CDR-family beat
it — consistent with our ibm_marrakesh finding that ZNE can hurt. Leaves (a)-(e) open.

---

## 6. Xu et al. 2025 — "Physics-inspired ML for QEM" (NNAS) (arXiv:2501.04558)

**What they did.** Neural Noise Accumulation Surrogate: embedding stage (device specs, noise type/rates, circuit params,
optionally noisy measurements) -> RNN "uniform neural accumulator" whose hidden state H_l recursively tracks layer-wise
noise accumulation -> attention-based extractor producing per-layer noise-impact factors r_l used in a closed-form
mitigation formula y_em = y_noisy / (prod(1-p_j) + r_l). Architecture-side ML-QEM: physics-structured NN, not a new
training-set or regressor-in-CDR idea. Simulation only (6q and 10q; classically computed labels).

**Key numbers.** 6q QAOA-type circuits, depths 1-20: 65.85% MAE reduction vs noisy, 35.12% vs best standard QEM
(ZNE/PEC/CDR); deep circuits (>15 steps): 79.42% / 55.63%. GHZ metrology (1-10q): 52.81-84.95% RMSE cut vs noisy, ~2 orders
of magnitude vs an RF baseline. Trains on ~100 sequences; >=10x less data than RF/MLP/GNN baselines (up to 90% dataset
reduction).

**Overlap verdict.** Compares against RF/MLP/GNN as mitigation-model baselines (the Liao-et-al model family) but everything
stays in the "predict the mitigated value" paradigm. No technique selection, no scaled backends as a learned feature, no
equal-budget baseline, no hardware, no CDR-internal regressor work. Leaves (a)-(e) open; cite as ML-QEM architecture
related work.

---

## Batch-3 bottom line

- Nobody in this batch does (a) technique SELECTION, (b) noise-strength-scaled backends as a training dimension,
  (c) an equal-shot-budget raw_plus fairness baseline, or (d) a sim-to-real selector transfer test. These remain ours.
- (e) is the contested axis: Zhao et al. (feature-augmented quadratic LS inside vanilla CDR) and Chen et al. (linear vs
  Lasso vs NN comparison inside NIL) both chip at it. Safe claim: "first systematic comparison of nonparametric ensemble
  regressors (RF/GB) as drop-in CDR regressors under finite, equal shot budgets, evaluated with grouped CV" — cite both.
- Liao et al. (NMI 2024) is the mandatory framing citation: RF as the workhorse model and cost-amortization argument, but
  output = mitigated value (replace), ours = technique choice (select). Their App. D noise-drift fine-tuning is the
  nearest prior to our sim-to-real concern — cite when motivating (d).
- DAEM's hardware table (ZNE worst at 0.259 MAE, CDR 0.095) independently corroborates our "ZNE hurts on real hardware at
  low shot counts" observation.

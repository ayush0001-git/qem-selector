# Deep Reader — Batch 2 (overlap analysis)

Date: 2026-07-22. Six papers flagged as potential overlap, read (abstract + available
HTML/methods/results). Verdicts scored against OUR five contribution axes:
(a) learned per-circuit technique SELECTOR with honest grouped/LOFO/LODO eval;
(b) noise-strength dimension via scaled backends;
(c) equal-budget raw_plus fairness baseline;
(d) sim-to-real transfer test of the selector on 2026 Heron hardware;
(e) CDR regressor swaps.

Key framing point that recurs below: every one of these papers is a **learn-to-MITIGATE**
(regression / distribution correction) or a **build-a-better-CDR** paper. NONE of them is a
**learn-to-SELECT-a-technique-from-static-features-without-spending-shots** paper. Our axis (a)
survives all six. The real erosion is on axis (e): the "swap CDR's linear fit for other
regressors" idea is now substantially covered by papers 4 and (partly) 3 — it is no longer a
novel headline, only a per-circuit / real-hardware / overfitting-regime refinement.

---

## 1. Deep Learning Approaches to Quantum Error Mitigation (arXiv:2601.14226, Jan 2026)
Placidi, Williams, Rinaldi, Mills, Cîrstoiu, Eccles, Duncan (Quantinuum-adjacent authors).

**What they did.** Systematically compare deep-learning architectures — fully connected NNs
through transformers — for correcting the noisy output *probability distribution* of measured
circuits toward the ideal distribution. Seq2seq attention models win. Trained/tested on both
simulated and REAL IBM QPU data up to 5 qubits, across several depths. Ablations on which input
features matter (circuit / device properties / noisy output statistics), cross-circuit-family
generalization, and transfer learning to a different IBM QPU.

**Key numbers.** No hard error numbers in the abstract; they claim their mitigated distributions
beat "baseline error mitigation techniques" across depths, and that transfer across similar
same-architecture devices works without full retraining. (Distribution-level correction, not a
single expectation value.)

**Overlap verdict.** This is a REGRESSION / distribution-correction model, NOT a selector — they
predict a mitigated distribution, they do not pick which of ZNE/CDR/REM to run. Overlaps ONLY on
our feature side (which inputs help) and on the sim-to-real transfer theme (they do real IBM data
+ cross-device transfer; we do a selector transfer test to Heron). Leaves fully open: (a) the
technique-selection framing, (b) the scaled-noise axis, (c) equal-budget raw_plus, (e) CDR
regressor swap. Their transfer result is a mild pre-emption of the *novelty* of "does an
ML-QEM model transfer to new hardware" — but they transfer a mitigator, we transfer a selector,
and their device pairs are same-architecture IBM, not a fake-backend->Heron distribution shift.
Cite as closest neighbor on the feature-ablation + transfer questions; NOT a selector competitor.

## 2. Configurable Readout Error Mitigation in Quantum Workflows (Electronics 11:2983, 2022)
Beisel, Barzen, Leymann, Truger, Weder, Yussupov (U. Stuttgart, QuantME/MODULO group).

**What they did.** A software-engineering / workflow paper, not an ML paper. Extends QuantME
(Quantum Modeling Extension for BPMN/BPEL workflow languages) with a configurable REM task:
surveys existing readout-error-mitigation methods, catalogs their general and method-specific
configuration options, and adds a Validator with hand-written validation RULES so a workflow
modeler picks a REM method consistent with device/requirement constraints; a Transformer then
refines the choice into an executable native-BPMN fragment.

**Key numbers.** None quantitative in the ML sense — this is a methodology/architecture + config-
option catalog with validation criteria, no accuracy benchmark, no learned model.

**Overlap verdict.** "Selection" exists, so it superficially touches axis (a) — but it is
RULE-BASED (requirement/validation rules), has NO learning, NO features, and operates ONLY WITHIN
the readout-mitigation family (it never arbitrates REM vs ZNE vs CDR). It is the closest prior
"automatic selection of a mitigation method" reference and MUST be cited to sharpen our
contribution, but it does not touch (b) noise-scaling, (c) equal-budget, (d) sim-to-real, (e)
regressor swaps, and it does not do learned cross-technique selection. Frame ours as: learned,
feature-driven, cross-technique, shots-free — vs their rule-based, within-REM, config-driven.

## 3. GEM: Scalable QEM with Physically Informed Graph Neural Networks (arXiv:2604.16815, 2026)
Wang, Wu, Liu, He, Shang, Guo, Chen.

**What they did.** Encode the circuit as an attributed graph (nodes = qubits carrying local T1/T2
+ readout error; edges = two-qubit gate errors / coupling) and train a GNN to model error
propagation and correct the observable, with a dual-branch affine correction for physical
consistency. Tested on 10- and 16-qubit random circuits on superconducting processors; headline
is zero-shot transfer from 10q-trained model to 16q.

**Key numbers (16-qubit, Table 2, mean MAE):** GEM 0.0903 (best) < CDR 0.0951 < GEM-without-edges
0.1036 < ZNE 0.1201 < MLP 0.1243 < Noisy 0.1297. So GEM beats CDR by ~5% and ZNE by ~25% MAE;
plain MLP barely beats noisy. 10q results only shown graphically (Figs 4-5). "Comparable to CDR at
small scales, better MAE + stability in zero-shot transfer to larger systems."

**Overlap verdict.** Another learn-to-MITIGATE entrant (regression on the observable), NOT a
selector. It IS a strong new "technique" a selector could later arbitrate over, and it is directly
relevant to axis (e) in spirit (a nonlinear/graph regressor beating linear CDR — same thesis as
our CDR-swap idea, but they build a standalone mitigator, not a swap inside the CDR training-set
pipeline). Leaves open: (a) selection framing, (b) our controlled scaled-noise axis (they vary
qubit count, not a noise-strength dial), (c) raw_plus, (d) selector sim-to-real. Use in related
work as evidence that nonlinear maps beat linear CDR (motivates our swap) AND as a candidate
future technique in the selector menu. Note their MLP baseline being weak (0.1243, worse than CDR)
is a useful cautionary datapoint for our overfitting-regime analysis.

## 4. Machine Learning-based QEM for Variational Algorithms (arXiv:2606.02697, 2026)
Korolev, Lakhmanskiy, Rabinovich.

**What they did.** THE regressor-swap experiment, in a VQE setting. Generate training data from
near-Clifford circuits and benchmark six regressor families — linear regression with L1/L2
(Ridge/Lasso), Random Forest, SVM, KNN, MLP, XGBoost — as the noisy->ideal map for VQE on the
Sherrington-Kirkpatrick Hamiltonian, up to 12 qubits (6q for validation). Simulated noise only
(depolarizing, Pauli, composite at p in {0.01,0.05,0.1}); 1,500 training samples/config; models
made transferable across target Hamiltonians. NO real hardware, NO per-circuit-family mitiq-CDR
sweep, NO selector.

**Key numbers.** Ridge on near-Clifford wins most regimes (median suppression 8.0x depolarizing /
7.4x Pauli at p=0.05). XGBoost competitive/better only at high noise (its piecewise-constant fit
suits the Clifford dataset; ~1.9-2.0x). ZNE beats ML at low noise (p=0.01: ZNE 44.4x vs Ridge
8.2x depolarizing) but ML wins at high noise (p=0.1: Ridge 3.5x vs ZNE 2.6x).

**Overlap verdict.** This is the SINGLE most overlapping paper for axis (e). It closes any naive
"first to try ML regressors on Clifford training data" claim — that headline is GONE. What it
LEAVES open for us: (i) per-circuit mitiq-style CDR swaps across our 5 heterogeneous circuit
FAMILIES (they are VQE/SK-only), (ii) REAL hardware (they are simulation-only — our Heron data is
a genuine gap they don't fill), (iii) explicit OVERFITTING-regime characterization / when nonlinear
maps hurt (they report winners but not a systematic overfit analysis), and critically (iv) the
entire SELECTOR contribution (a) — they compare regressors, they do not learn to pick a technique
from static features. Also their finding "Ridge (linear) usually wins, nonlinear only helps at high
noise" is exactly the skeptical result we should expect and cite; it reframes our CDR-swap as
"characterize WHEN the swap helps," not "the swap wins." Must cite prominently; reposition axis (e)
as a refinement, not a discovery.

## 5. Improving the efficiency of learning-based error mitigation (Czarnik et al., Quantum 9, 1727)
(arXiv:2204.07109, 2022; published 2025). Czarnik, McKerns, Sornborger, Cincio — the CDR authors.

**What they did.** Original-CDR-authors' follow-up on making CDR cheaper, NOT changing the
regressor. Two levers: (1) smarter training-DATA selection that fixes the pathology where training
expectation values cluster near zero (which starves the fit of signal), and (2) exploiting problem
symmetries. Keeps the LINEAR fit. Benchmarked on IBM Toronto + noise-model sims (XY-model
long-range correlators; LiH ground-state energy with Ourense noise).

**Key numbers.** ~10x cheaper at equal accuracy vs standard CDR; ~factor-10 improvement over
unmitigated at a 2x10^5-shot budget; "orders of magnitude improvements in frugality."

**Overlap verdict.** Touches axis (e) only tangentially and in a way that HELPS us: they modify the
training DATA and keep the linear map, so the regressor axis is explicitly left open — and their
central observation (training expectation values clustering near zero degrades the linear fit) is a
direct physical MOTIVATION for trying nonlinear/robust regressors (our swap) and for our low-signal
|ideal| screening. Does not touch (a) selection, (b) our scaled-noise dial (they don't sweep noise
strength as a controlled axis), (c) raw_plus, (d) selector transfer. Cite as: (i) authority that
the regressor swap is unexplored by the CDR originators, (ii) the mechanism that justifies it.

## 6. Extension of Clifford Data Regression Methods for QEM (arXiv:2411.16653, 2024)
Pérez-Guijarro, Pagès-Zamora, Fonollosa (UPC Barcelona).

**What they did.** Two CIRCUIT-CONSTRUCTION variants of CDR: (1) multiple copies of the original
circuit, (2) adding a layer of single-qubit rotations (plus an insertion+ZNE combination), with
theoretical complexity/error-scaling analysis (error ~ O(sqrt((J+1)/S) + 1/sqrt(N))) and numerics
(QFT circuits) showing reduced RMSE. The estimator stays LINEAR — ridge regression
argmin ||f - Phi alpha||^2 + mu||alpha||^2; they mention the kernel trick only as a compute
optimization, NOT as a nonlinear regressor swap. NN/RF/kernel regressors are NOT used.

**Key numbers.** Theory-heavy; concrete RMSE tables were not extractable from the HTML we could
reach (Section 5 numerics on QFT show "reduced RMSE" for the variants vs standard CDR, magnitudes
not pinned down). Treat quantitatively as "improves on standard CDR on QFT, numbers unverified."

**Overlap verdict.** Purely circuit-CONSTRUCTION changes to CDR — orthogonal to our regressor axis
(e), which they explicitly leave open (linear fit retained). Does not touch (a) selection, (b)
noise-scaling, (c) raw_plus, (d) sim-to-real. Cite as evidence that "extending CDR" so far means
new circuit constructions, NOT new regressors — sharpening that OUR (e) is the regressor axis
nobody in the CDR-extension line has taken. Lowest overlap of the six.

---

## Net assessment for the paper
- Axis (a) learned cross-technique feature-driven selector: **UNTOUCHED** by all six. Closest is
  paper 2 (rule-based, within-REM only) — cite to contrast, not a competitor.
- Axis (b) scaled-noise dial, (c) equal-budget raw_plus, (d) selector sim-to-real to Heron:
  **UNTOUCHED** by all six. (Paper 1 does mitigator transfer on real IBM, paper 4 sweeps noise p
  but as a regression benchmark, not a controlled selector axis.)
- Axis (e) CDR regressor swap: **HEAVILY ERODED**. Paper 4 already benchmarks Ridge/Lasso/RF/SVM/
  KNN/MLP/XGBoost on near-Clifford data (VQE, sim-only); paper 3 shows a GNN beating linear CDR.
  Reposition (e) as: per-circuit-FAMILY swaps in the mitiq-CDR pipeline + REAL-hardware +
  overfitting-regime characterization + feeding the swapped regressor as one technique into the
  selector. Do NOT claim novelty of "ML regressors for CDR"; papers 5 & 6 confirm the CDR line
  itself hasn't swapped the regressor, but paper 4 has (outside mitiq/per-family). Cite 4 and 3
  prominently and frame honestly.

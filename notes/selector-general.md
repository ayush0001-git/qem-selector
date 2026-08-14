# Novelty re-scan — "selector-general" (BOTH angles)

Run 2026-07-22 by novelty searcher `selector-general`. Hostile-reviewer re-scan of
the umbrella claim that could subsume Angle 2 (nonlinear-CDR as a selectable technique)
and Angle 3 (selector boundary vs ZNE help-harm theory): **a per-circuit, learned,
execution-free selector across QEM technique families.** 13 web queries + 5 paper
fetches. Every arXiv ID below was fetched and resolved.

## Verdict: PARTIALLY-COVERED (the exact learned/execution-free/family/transfer selector is still OPEN)

The *idea* of "select/adapt a QEM technique" is occupied by adjacent work (Mitiq
Calibrator, GSC-QEMit, Beisel) — so we must NOT claim selection-of-QEM is new. But no
2026 paper does the specific thing: **predict the technique family from static circuit+
backend features, without spending shots, with grouped/LOFO/LODO holdout and a
sim-to-real transfer test.** That instantiation survives the re-scan intact. Both
sub-pillars (nonlinear-CDR-as-selection-target; selector-boundary-vs-theory) are clear.

## What the re-scan ADDED beyond docs/LITERATURE.md

1. **QEMOS** — "a scalable quantum error mitigation method to overcome qubit
   sensitivity", Mach. Learn.: Sci. Technol. 2026, DOI 10.1088/2632-2153/ae16fc
   (https://iopscience.iop.org/article/10.1088/2632-2153/ae16fc). Search snippets say
   it "recommends error mitigation methods" — FALSE ALARM. Fetched the article: QEMOS
   is a random-forest **ML-MITIGATOR** that *performs* mitigation via post-processing
   (same class as ML-QEM), trained on circuit+backend+output features, run on real HW
   (tianyan-176). It does NOT select among ZNE/CDR/REM and is NOT execution-free
   (needs the noisy output distribution as a feature). Its actual novelty is decoupling
   from qubit-count so it generalizes to more qubits than trained on (5–9q → 2–13q).
   → Add as an ML-QEM cousin; its qubit-count-transfer trick is a useful comparator for
   our transfer story, NOT a threat to the selector task.

2. **QEM-Bench / QEMFormer** (ICML 2025 poster, https://icml.cc/virtual/2025/poster/45382)
   — a benchmark + transformer baseline for learning-based *mitigators*. Not a selector.

3. **QAGT-MLP** (arXiv:2511.03119), **SSQEM semi-supervised QEM**,
   **Data-driven adaptive QEM for probability distributions** (arXiv:2511.13231) — all
   ML-MITIGATORS, none selects among families.

4. **Pi-QEM / Pauli Weight Term Selection** (arXiv:2606.31195, 30 Jun 2026, fetched) —
   selects Hamiltonian/Pauli *training terms* WITHIN an ML-QEM model, not the technique.
   Not a selector, no theory-boundary check.

5. **bnZNE / Verifiable Benchmark Circuits** (arXiv:2603.10224) — a benchmark-circuit
   improvement to ZNE. Not a selector.

## Re-verification of load-bearing citations (field moves monthly)

- **Korolev et al. arXiv:2606.02697** (1 Jun 2026) — RESOLVED. NOTE: the *abstract alone*
  reads as a single "ML-QEM protocol beats ZNE" story and does NOT mention the regressor
  benchmark. Only the **full body** (fetched https://arxiv.org/html/2606.02697) confirms
  it benchmarks **8 regressors** (linear, Ridge, Lasso, RF, SVM, KNN, MLP, XGBoost) on
  near-Clifford CDR data. Headline: **Ridge wins overall** ("strong regularization and
  numerical stability"); **XGBoost occasionally beats Ridge only at high noise (p=0.1,
  depolarizing/Pauli)**. **Sim-only, VQE-only, ≤12q.** So docs/LITERATURE.md is CORRECT
  and the abstract's spin hides the regressor comparison — cite the body, not the abstract.
  Angle-2 reframe stands: Korolev closes "which regressor for CDR (VQE, sim)" but NOT
  (i) regressor-choice as a selection target, (ii) heterogeneous non-VQE families,
  (iii) real hardware, (iv) the train-size × fraction_non_clifford overfitting map.

- **GSC-QEMit arXiv:2604.24551** (27 Apr 2026) — RESOLVED. Confirmed: selects mitigation
  **intensity** (graded levels) via **online contextual bandit (Thompson sampling)** over
  **streaming telemetry**, **Qiskit Aer sim only**. NOT technique-family, NOT static
  features, NOT offline supervised, NO LOFO/LODO. Remains the closest-in-spirit threat and
  the key differentiation paragraph — unchanged from the lit file.

- **Decision Kernels arXiv:2607.02888** (Scavino) — recurs as the nearest "select QEM by
  geometry/decision-quality, not MSE" idea. It is a *theory of how to score* selection,
  **builds no selector**, and does NOT compare a *learned* boundary to an *analytic* one.
  So it does not touch Angle 3's specific novelty (data-driven selector boundary vs the
  analytic ZNE help-harm boundary) — but it IS the paper a reviewer cites to attack our
  argmin-|error| labels. Already on the DO-NOT-CLAIM list; keep the disclosure.

## Bottom line for framing (both angles)

- The umbrella **learned, feature-driven, execution-free family-selector** is UNSCOOPED.
  Keep the lit file's precise wording — "no method **predicts** the technique family from
  static features **without spending shots**" — never "no tool selects across QEM".
- **Angle 2** is CLEAR with the reframe: nobody embeds a regressor-choice (incl. a
  nonlinear-CDR option) as a selection target across heterogeneous families on real HW
  with an overfitting-regime map. Reproduce Korolev's Ridge-wins from the body as the
  sanity anchor.
- **Angle 3** is CLEAR: no paper validates a data-driven QEM selector's *learned* decision
  boundary against an *analytic* mitigation help-harm boundary (Scavino 2605.08251 supplies
  the theory ground truth; Decision Kernels theorizes selection but builds nothing and
  makes no theory-vs-learned-boundary comparison).
- The surrounding **ML-mitigator** space is CROWDED and getting more so monthly (QEMOS,
  QEM-Bench/QEMFormer, QAGT-MLP, Pi-QEM). A reviewer WILL ask "isn't this ML-QEM?" — the
  one-line answer stays: they learn to *be* a cheaper mitigator; we learn *which* mitigator
  to run, and the mitigation is still real Mitiq ZNE/CDR/REM. Re-run this scan at submission.

# Literature search: CDR regressor landscape ("cdr-variants")

Date: 2026-07-22. Searcher task: does any published work already swap CDR's linear
regression for non-linear / ML regressors, i.e. does it close our planned novel angle
(PROJECT_STATUS §5.3)? Method: 8 distinct WebSearch queries + WebFetch on ~10
abstracts/HTML pages. Every claim below is from a fetched page, not from memory.

## BOTTOM LINE (verdict first)

**The naive version of the angle is CLOSED as of June 2026; a sharpened version is
still open.**

- **arXiv:2606.02697 (Korolev, Lakhmanskiy, Rabinovich, June 2026)** benchmarks
  Ridge/Lasso linear regression, Random Forest, SVM, KNN, MLP and **XGBoost** trained
  on (near-)Clifford circuit data for VQE error mitigation. That is exactly
  "ML regressors on Clifford training data". Their headline: **regularized LINEAR
  (Ridge) on near-Clifford data wins most regimes; XGBoost competitive** — the final
  map is literally linear, f(P) = MP + b. So "we are the first to try sklearn
  regressors inside CDR" is no longer a defensible claim, and their result predicts
  the likely outcome of our ablation (linear + regularization is hard to beat).
- **Everything else in the CDR-variants literature modifies the TRAINING DATA or
  CIRCUITS, not the regressor** (Czarnik 2204.07109 = data selection + symmetries;
  Pérez-Guijarro 2411.16653 = circuit copies / extra rotation layer;
  Zhao 2511.03556 = energy-based training-circuit selection + an extra regression
  input; Chen 2512.12578 = 2-design neighbor circuits). All keep (near-)linear fits.
  mitiq itself only ships linear fit functions (`linear_fit_function`,
  `linear_fit_function_no_intercept`, scipy `curve_fit` underneath).

**What 2606.02697 does NOT do — the open re-framing for our paper:**
1. It is VQE-specific: one mitigation map learned per ansatz, generalizing across
   *variational parameters*, ≤12 qubits, Sherrington-Kirkpatrick Hamiltonian,
   simulated noise models only. NOT per-circuit mitiq-style CDR across heterogeneous
   circuit families (our layered_random / ghz / mirror / hw_efficient / near_clifford
   grid), NOT real hardware.
2. It reports *which* regressor wins, not a regime analysis of **when and why
   nonlinear maps help vs overfit** as a function of training-set size and
   `fraction_non_clifford` — exactly the characterization our step-3 plan specifies.
3. Nobody couples regressor choice to a **technique selector** (our core artifact):
   treating "CDR-with-regressor-X" as additional techniques the selector can
   recommend from static features, and testing sim-to-real transfer of that choice,
   is untouched.
4. Our real-hardware finding (ZNE harmful at 1024 shots on Heron; REM winning on
   mirror circuits) plus Köster & Mauerer 2605.29872 (ZNE conclusions flip with
   parameters and drift) supports a hardware-validated, statistically careful
   framing that none of the regressor papers have.

**Recommended framing change:** not "first ML regressor in CDR" but "a systematic
regime study of the CDR regression ansatz (linear vs nonlinear, training-set size,
non-Clifford fraction) integrated into a shot-free technique selector, validated on
real Heron hardware" — and cite 2606.02697 as the closest prior work, reproducing
its Ridge-wins result as a sanity anchor if our data agrees.

---

## Papers (classified)

### direct-overlap

1. **Machine Learning-based Quantum Error Mitigation for Variational Algorithms** —
   Korolev, Lakhmanskiy, Rabinovich — arXiv:2606.02697 (Jun 2026).
   https://arxiv.org/abs/2606.02697
   Benchmarks linear (L1/L2), Random Forest, SVM, KNN, MLP, XGBoost regressors
   trained on (near-)Clifford data (parametrized gates replaced by random Cliffords,
   optional Haar layer) for VQE on SK Hamiltonian ≤12 qubits; several-fold error
   suppression, beats ZNE in high noise; Ridge on near-Clifford data best overall,
   XGBoost's piecewise-constant fit noted as well-suited to Clifford data.
   **Verdict: closes the naive regressor-swap claim (in the VQE setting); leaves
   per-circuit CDR across circuit families, real hardware, overfitting-regime
   analysis, and selector integration open.**

### adjacent (CDR variants that do NOT touch the regressor — angle left open by each)

2. **Improving the efficiency of learning-based error mitigation** — Czarnik,
   Maziarz(?), Coles, Cincio et al. — arXiv:2204.07109; Quantum 9, 1727 (2025;
   preprint 2022). https://arxiv.org/abs/2204.07109
   CDR follow-up by the original authors: smarter training-data selection (fixes
   clustering of training expectation values near zero) + problem symmetries;
   ~10x cheaper at equal accuracy on IBM Toronto (XY-model correlators). Modifies
   data, keeps the linear fit. Verdict: leaves regressor swap open; their
   "training data clustering" observation is a direct motivation for nonlinear maps.

3. **Extension of Clifford Data Regression Methods for Quantum Error Mitigation** —
   Pérez-Guijarro, Pagès-Zamora, Fonollosa — arXiv:2411.16653 (Nov 2024).
   https://arxiv.org/abs/2411.16653
   Two CDR variants: (a) multiple copies of the original circuit, (b) an added layer
   of single-qubit rotations; theory (complexity, error scaling) + numerics.
   Circuit-construction changes only — no NN/RF/kernel regressors. Verdict: open.

4. **Quantum error mitigation using energy sampling and extrapolation enhanced
   Clifford data regression** — Zhao, Kjellgren, Coriani, Kongsted, Sauer, Ziems —
   arXiv:2511.03556 (Nov 2025). https://arxiv.org/abs/2511.03556
   VQE/quantum-chemistry CDR: Energy Sampling (keep only lowest-energy training
   circuits) + Non-Clifford Extrapolation (adds #non-Clifford parameters as an extra
   regression INPUT so the map extrapolates toward the target circuit). Enriches
   the regression's inputs/data, does not replace the regression model class.
   Verdict: open, but NCE is a half-step toward "feature-augmented CDR" — cite it.

5. **Scalable Quantum Error Mitigation with Neighbor-Informed Learning** — Chen,
   Cheng, Gao, Lin, Zhang, Wei, Ji — arXiv:2512.12578 (Dec 2025).
   https://arxiv.org/abs/2512.12578
   Learns ideal output of a target circuit from noisy outputs of structurally related
   "neighbor" circuits; replaces the conventional random-Clifford-substitution
   training set with a 2-design construction; training-set size scales
   logarithmically in #neighbors. Training-data innovation, regressor unspecified in
   abstract. Verdict: open on the regressor axis.

6. **Machine learning for practical quantum error mitigation** — Liao, Wang,
   Sitdikov, Salcedo, Seif, Minev (IBM) — arXiv:2309.17368; Nature Machine
   Intelligence 6, 1478–1486 (2024). https://arxiv.org/abs/2309.17368
   Random forests, linear regression, MLPs, GNNs trained to MIMIC mitigation
   (reference: digital ZNE) so mitigated values come nearly for free at runtime;
   up to 100 qubits on real IBM devices. Not CDR-internal (different training
   philosophy — mimic an existing technique), but it is the flagship "RF as
   mitigation model" paper and MUST be cited. Verdict: adjacent, leaves CDR
   regressor swap open; also relevant to our selector (they amortize mitigation cost,
   we amortize technique CHOICE cost).

7. **Learning-based quantum error mitigation** — Strikis, Qin, Chen, Benjamin, Li —
   PRX Quantum 2, 040330 (2021).
   https://www.semanticscholar.org/paper/7d86c9b323febee941439b34ec5777ed8ed3c002
   Learns quasi-probability compensation coefficients ab initio from Clifford-
   substituted variants of the target circuit. Learning-based QEM ancestor, but the
   learned object is a QP distribution, not a noisy->ideal regression map.
   Verdict: open; standard citation for the "learning-based QEM" family.

8. **Noise-agnostic quantum error mitigation with data augmented neural models** —
   M. Liao, Zhu, Chiribella, Yang — arXiv:2311.01727; npj Quantum Information 11, 8
   (2025). https://arxiv.org/abs/2311.01727
   Neural error-mitigation model trained WITHOUT noise-free data via a quantum data
   augmentation trick; works across circuits/many-body/CV systems, tested on real
   hardware. Neural-QEM, not CDR-structured. Verdict: adjacent.

9. **Physics-inspired Machine Learning for Quantum Error Mitigation (NNAS)** — Xu,
   Xue, Chen, Ding, Li, Zhou, Huang, Bao — arXiv:2501.04558 (Jan 2025).
   https://arxiv.org/abs/2501.04558
   Neural Noise Accumulation Surrogate: NN architecture encoding layer-wise noise
   accumulation structure; >50% error reduction on deeper circuits, smaller training
   sets. Architecture-side innovation for ML-QEM. Verdict: adjacent.

10. **Large-scale quantum approximate optimization on non-planar graphs with machine
    learning noise mitigation** — Sack, Egger (IBM) — arXiv:2307.14427; Phys. Rev.
    Research 6, 013223 (2024). https://arxiv.org/abs/2307.14427
    Feed-forward NN mitigates QAOA at 40 qubits (958 CX gates); mitigates SAMPLES,
    not just expectation values. Application-scale ML-QEM. Verdict: adjacent.

### supporting (methodology / selector premise)

11. **Unified approach to data-driven quantum error mitigation (vnCDR)** — Lowe,
    Gordon, Czarnik, Arrasmith, Coles, Cincio — arXiv:2011.01157; Phys. Rev.
    Research 3, 033098 (2021). https://arxiv.org/abs/2011.01157
    CDR + variable noise levels in the training data (unifies ZNE and CDR); beats
    both (e.g. 8-qubit Ising energy: 33x over raw, 20x over ZNE, 1.8x over CDR).
    Still a (multi-)linear fit. The canonical "CDR variant" to cite alongside the
    original; also a candidate technique for our benchmark's future work list.

12. **Unifying and benchmarking state-of-the-art quantum error mitigation
    techniques (UNITED)** — Bultrini, Gordon, Czarnik, Arrasmith, Cerezo, Coles,
    Cincio — arXiv:2107.13470; Quantum 7, 1034 (2023).
    https://arxiv.org/abs/2107.13470
    Benchmarks ZNE / CDR / vnCDR / virtual distillation under trapped-ion noise:
    **which technique wins depends strongly on the shot budget** (UNITED only wins
    at 10^10 shots). This is the single best literature support for our selector
    premise ("no universal winner -> predict the winner from features"), and for our
    hardware observation that ZNE loses at small shot budgets.

13. **Claim against Measurement: Statistical Artefacts in Quantum Error Mitigation
    Benchmarks** — Köster, Mauerer — arXiv:2605.29872 (May 2026).
    https://arxiv.org/abs/2605.29872
    132-config ZNE sweep: scale factors / extrapolation choice / calibration flip
    conclusions between "significant improvement" and "significant degradation";
    72-hour drift study: same ZNE config's effect size varies >3x with WHEN it runs;
    only 25% of 81 reviewed QEM papers use proper inferential statistics. Directly
    supports (a) our ZNE-made-it-worse hardware result being real and expected, and
    (b) our seed-averaged, grouped-CV, honest-stats methodology. Cite prominently.

14. **Mitiq CDR options documentation** — Unitary Foundation, current stable docs.
    https://mitiq.readthedocs.io/en/stable/guide/cdr-3-options.html
    Ground truth for the baseline we modify: `fit_function` accepts only
    linear forms (`linear_fit_function`, `linear_fit_function_no_intercept`) fitted
    via scipy `curve_fit`; no sklearn/nonlinear regressor support shipped. Documents
    that the tool ecosystem itself has not absorbed regressor swapping.

### background

15. **Error mitigation with Clifford quantum-circuit data (original CDR)** —
    Czarnik, Arrasmith, Coles, Cincio — arXiv:2005.10189; Quantum 5, 592 (2021).
    https://arxiv.org/abs/2005.10189
    The origin: near-Clifford training circuits resembling the target, classical
    simulation of ideal values, **explicitly a linear ansatz** fit noisy->ideal;
    order-of-magnitude error reduction on 16-qubit IBMQ ground-state problem and
    64-qubit noisy simulator. The linearity of the ansatz is stated by design —
    which is what makes a principled ablation of it a legitimate question.

16. **Quantum Error Mitigation (review)** — Cai, Babbush, Benjamin, Endo, Huggins,
    Li, McClean, O'Brien — arXiv:2210.00921; Rev. Mod. Phys. 95, 045005 (2023).
    https://arxiv.org/abs/2210.00921
    The standard field review (covers ZNE, PEC, REM, learning-based methods incl.
    CDR). Use for the related-work section's framing paragraph.

---

## Search log (queries run)

1. "Czarnik Clifford data regression error mitigation arXiv"
2. "variable noise Clifford data regression vnCDR unified data-driven error mitigation"
3. "Clifford data regression neural network nonlinear regression quantum error mitigation"
4. '"Clifford data regression" random forest OR "kernel regression" OR "Gaussian process" arxiv' (no CDR hits — no GP/kernel-CDR paper appears to exist)
5. "learning-based quantum error mitigation Strikis survey neural network error mitigation near-Clifford training"
6. 'site:arxiv.org "Clifford data regression" improved OR variant OR nonlinear OR machine learning 2024 2025'
7. "Czarnik improving efficiency learning-based error mitigation arxiv 2204 near-Clifford training circuits"
8. '"Clifford data" quantum error mitigation "XGBoost" OR "gradient boosting" OR "support vector regression"'
9. "mitiq CDR fit_function nonlinear regression custom regressor documentation"

Fetched (abstract or HTML): 2411.16653, 2511.03556, 2606.02697 (abs + full HTML),
2311.01727, 2309.17368, 2501.04558, 2512.12578, 2605.29872, 2310.13382 (excluded
from final list — DNN postprocessing, not Clifford-trained), 2307.14427, 2107.13470.

Notable non-findings: no published paper applies Gaussian-process or kernel-ridge
regression inside per-circuit CDR; no paper studies regressor choice as a function
of training-set size / fraction_non_clifford; no paper connects CDR-variant choice
to a technique-selection model.

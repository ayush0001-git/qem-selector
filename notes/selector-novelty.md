# Novelty threat scan — "ML selector for QEM techniques" (agent: selector-novelty)

Date: 2026-07-22. Method: 11 distinct web searches (arXiv-focused, varied phrasing:
"technique selection", "adaptive", "recommendation", "algorithm selection",
"bandit", "learning to select", "automated", mitiq Calibrator, workflow selection)
+ WebFetch of abstracts/docs for the 8 most threatening hits.

## VERDICT (one paragraph)

**No direct overlap found.** As of 2026-07-22 I could not find any published work
that trains a supervised model to predict, from *static circuit + backend features
and with zero additional quantum executions at selection time*, which of several
QEM technique families (ZNE / CDR / REM / raw) will give the lowest error for a
given (circuit, device) pair. The two nearest threats are (1) **GSC-QEMit**
(arXiv:2604.24551, IJCNN 2026) — a contextual-bandit policy that adaptively picks
mitigation *intensity levels* (NONE/MODERATE/SEVERE) from streaming device
telemetry, i.e., online, telemetry-driven, and not a choice among distinct
technique families — and (2) **Mitiq's Calibrator** — which picks a strategy by
*running* benchmark experiments on the backend (spends shots; primarily tunes ZNE
hyperparameters; no ML/prediction). Our delineation sentence writes itself:
*offline-trained selector, cross-technique-family, predicts from cheap static
features without spending a single shot at recommendation time, with cost-aware
labels, an equal-budget control (raw_plus), and a sim-to-real hardware transfer
test.* Caveat: the field is moving fast (3 of the closest papers are from
April–July 2026); re-run this scan immediately before submission.

## Classified hits

### Closest threats (adjacent — MUST cite and differentiate)

1. **GSC-QEMit: A Telemetry-Driven Hierarchical Forecast-and-Bandit Framework for
   Adaptive Quantum Error Mitigation** — arXiv:2604.24551 (Apr 2026, accepted
   IEEE/INNS IJCNN 2026). https://arxiv.org/abs/2604.24551
   - ACTUALLY DID: GHSOM clusters streaming telemetry into contexts; a Gaussian-
     process forecaster predicts near-horizon fidelity; a Thompson-sampling
     contextual bandit selects among graded mitigation *intensity levels*
     (NONE/MODERATE/SEVERE). Qiskit Aer simulation only (GHZ/QFT/Grover under
     nonstationary drift). +9.0% mean logical fidelity, −35% intervention cost
     vs static-severe.
   - vs US: it learns *how much* mitigation, online from telemetry; we learn
     *which technique family*, offline from static circuit+backend features with
     no execution at decision time. Different action space, different information
     regime. This is the single most important paper to cite/differentiate.

2. **Mitiq Calibrator / calibration module** — docs + Unitary Foundation blog
   (2023–). https://mitiq.readthedocs.io/en/stable/guide/calibrators.html ,
   https://unitary.foundation/posts/calibration/ ,
   https://mitiq.readthedocs.io/en/stable/examples/calibration-tutorial.html
   - ACTUALLY DOES: runs a series of benchmark experiments (Settings ->
     BenchmarkProblems + Strategies) *on the backend*, averages improvement
     across circuits, returns the best-performing Strategy. Docs state support
     centers on ZNE (scale factors, extrapolation, folding methods). No ML, no
     feature-based prediction — it is empirical trial-and-error, costs shots.
   - vs US: our selector predicts without spending shots and spans technique
     families (ZNE vs CDR vs REM vs raw). PROJECT_STATUS §5.4's planned framing
     ("predicting from static features without spending shots") survives contact
     with the current docs.

3. **Machine Learning for Practical Quantum Error Mitigation (ML-QEM)** — Liao,
   Wang, et al. (IBM), arXiv:2309.17368; Nature Machine Intelligence 6,
   1478–1486 (2024). https://arxiv.org/abs/2309.17368
   - ACTUALLY DID: ML models (linear regression, RF, MLP, GNN) *predict mitigated
     expectation values* (regression), trained to mimic ZNE at much lower runtime
     cost; up to 100 qubits on IBM hardware. Random forest consistently best.
   - vs US: ML *performs* the mitigation; it never chooses among techniques. The
     shared machinery (circuit/device features + RF) means a reviewer WILL bring
     this up — frame ours as the *algorithm-selection* layer that could sit on
     top of ML-QEM or any technique. Related IBM patent: US 12,481,908
     "Performing quantum error mitigation at runtime using trained machine
     learning model"
     (https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12481908)
     — claims cover training GNN/MLP/RF/OLS models to mitigate results at
     runtime, i.e., the ML-QEM idea, NOT technique selection.

### Adjacent (compare/benchmark or learn-to-mitigate; no learned selector)

4. **Volumetric Benchmarking of Error Mitigation with Qermit** — Cirstoiu et al.
   (Quantinuum), arXiv:2204.09725; Quantum 7, 1059 (2023).
   https://arxiv.org/abs/2204.09725
   - Benchmarks QEM protocols (ZNE, CDR, PEC, SPAM...) volumetrically across
     hardware to "identify situations in which their use is beneficial" — the
     *manual* version of our question. No predictive model. Also useful: their
     finding that predicted vs practical performance of QEM often disconnects
     supports our sim-to-real shift hypothesis.

5. **Decision Kernels for Quantum Error Mitigation: Why Accuracy Gains Need Not
   Improve Downstream Decisions** — Scavino, arXiv:2607.02888 (Jul 2026).
   https://arxiv.org/abs/2607.02888
   - Theory: QEM benchmarking mismatch between expectation-value accuracy and
     downstream decisions; advocates selecting QEM methods via "residual gap
     geometry", shows CDR can be decision-flat and PEC can worsen decision risk.
     No trained selector. Fresh (July 2026) — cite as evidence that
     "which-technique" selection is an open, recognized problem.

6. **Q-LEAR: A Machine Learning-Based Error Mitigation Approach For Reliable
   Software Development On IBM's Quantum Computers** — arXiv:2404.12892; FSE
   Companion 2024. https://arxiv.org/abs/2404.12892
   - ML with a novel feature set *corrects outputs* (learns error, predicts
     corrected values) across 8 IBM machines + simulators. Again: mitigation via
     ML, not selection among techniques.

7. **Deep Learning Approaches to Quantum Error Mitigation** — arXiv:2601.14226
   (Jan 2026). https://arxiv.org/abs/2601.14226
   - Compares FCN → transformer architectures for mitigating output
     distributions; studies which *input features* (circuit, device properties,
     noisy outputs) matter. Feature overlap with us, task is regression/
     distribution correction.

8. **Physics-inspired Machine Learning for Quantum Error Mitigation** —
   arXiv:2501.04558 (Jan 2025). https://arxiv.org/abs/2501.04558
   - Head-to-head ZNE/PEC/CDR vs ML-QEM (RF) comparisons — an example of the
     growing compare-QEM-methods literature; no selector.

9. **GEM: Scalable Quantum Error Mitigation with Physically Informed Graph
   Neural Networks** — arXiv:2604.16815 (Apr 2026).
   https://arxiv.org/abs/2604.16815
   - GNN beats CDR/ZNE on 16-qubit random circuits (MAE 0.090 vs 0.095/0.120).
     Another learn-to-mitigate entry; also a candidate future "technique" our
     selector could arbitrate over.

10. **Configurable Readout Error Mitigation in Quantum Workflows** — Beisel,
    Barzen, Leymann et al., Electronics 11(19):2983 (2022).
    https://www.mdpi.com/2079-9292/11/19/2983
    - Rule/requirements-based automated *configuration and selection of REM
      methods* inside BPMN quantum workflows (QuantME). Selection exists but is
      hand-engineered rules, within the REM family only, no learning.

### Supporting (methodological analogs + evidence for our hardware findings)

11. **Predicting Good Quantum Circuit Compilation Options (MQT Predictor)** —
    Quetschlich, Burgholzer, Wille, arXiv:2210.08027; IEEE QSW 2023.
    https://arxiv.org/abs/2210.08027
    - Supervised classifier (RF best) predicts best device/compiler/settings
      from circuit features: ~75% accuracy, >95% top-3, 3000 circuits.
      EXACTLY our method template, applied to compilation instead of mitigation.
      Cite as the algorithm-selection precedent; our contribution = same
      paradigm brought to QEM + cost-awareness + sim-to-real hardware test.
      (Preempt the "incremental" review by owning this analogy explicitly.)

12. **Benchmarking Error Mitigation: Artefactual Improvements in Zero-Noise
    Extrapolation** — Köster & Mauerer, arXiv:2607.09360 (Jul 10, 2026).
    https://arxiv.org/abs/2607.09360
    - Richardson ZNE can collapse into a fixed rescaling of one noisy
      measurement, yielding bogus apparent improvements; their garbage-folding
      negative control "improved" more than real folding; on IQM hardware ZNE
      overshot ideal by up to 21%. DIRECTLY supports our ibm_marrakesh
      observation (ZNE worse than raw on all 3 circuits at 1024 shots) and
      motivates a selector that can recommend "raw".

13. **Best practices for quantum error mitigation with digital zero-noise
    extrapolation** — Majumdar et al., arXiv:2307.05203 (2023).
    https://arxiv.org/abs/2307.05203
    - ZNE outcome hinges on scale-factor/extrapolation choices; costly
      trial-and-error; works only when noise is stable. Motivation for
      automated, learned selection.

### Background (foundations; terminology traps)

14. **Learning-based quantum error mitigation** — Strikis, Qin, Chen, Benjamin,
    Li, arXiv:2005.07601; PRX Quantum 2, 040330 (2021).
    https://arxiv.org/abs/2005.07601
    - TERMINOLOGY TRAP: "learning-based QEM" here means learning quasi-
      probability correction parameters for the circuit — nothing to do with
      selecting among techniques. Don't let the title scare us; do disambiguate
      our title/abstract from this established phrase.

15. **Error mitigation with Clifford quantum-circuit data (CDR)** — Czarnik,
    Arrasmith, Coles, Cincio, arXiv:2005.10189; Quantum 5, 592 (2021).
    https://arxiv.org/abs/2005.10189
    - Foundation of our CDR arm (and the vnCDR/eCDR follow-up line, e.g.
      arXiv:2511.03556 combining CDR+ZNE). Background citation.

## Searches that came back clean (evidence of absence)

- "predict the best error mitigation" / "selecting the best error mitigation" —
  only ML-QEM regression hits.
- "recommender / recommendation system" + QEM — only a circuit-encoding
  recommender (quantumzeitgeist.com note) and compilation-option prediction.
- "multi-armed / contextual bandit" + error mitigation technique selection —
  only GSC-QEMit (intensity levels) and unrelated quantum-bandit theory.
- "algorithm selection / meta-learning" + QEM — only ML-QEM variants
  (arXiv:2606.02697 VQA-tailored ML-QEM, arXiv:2606.31195 Pi-QEM training-
  observable selection — both regression-side, not technique selection).
- "learning to select / learned selector" + QEM — nothing beyond the above.

## Framing recommendations for the paper

1. Novelty sentence: first *offline-trained, execution-free* selector across QEM
   technique *families* (raw/ZNE/CDR/REM) from static circuit+backend features,
   with cost-aware labels and equal-budget controls, validated for sim-to-real
   transfer on a Heron-class device.
2. Must-cite-and-differentiate set: GSC-QEMit, Mitiq Calibrator, ML-QEM (+ IBM
   patent), Qermit, MQT Predictor.
3. Own the MQT-Predictor analogy proactively (algorithm selection paradigm);
   our added value = QEM domain, cost model, honest refusal-aware labels,
   hardware transfer experiment.
4. Our ZNE-worse-on-hardware result is corroborated by arXiv:2607.09360 — cite
   it in the hardware section; it strengthens (not undermines) the negative
   result.
5. Re-run this scan right before submission — GSC-QEMit (Apr 2026), Decision
   Kernels (Jul 2026) and the ZNE-artefacts paper (Jul 2026) show the space is
   heating up in exactly our direction.

# Literature search: QEM benchmark / comparison studies

Searcher label: "benchmarks". Date: 2026-07-22. Method: 8 WebSearch queries (varied
phrasing, arXiv-focused) + WebFetch on the 9 most promising papers (ar5iv HTML where
possible to get real numbers). Classification is threat-to-our-novelty:
direct-overlap / adjacent / supporting / background.

Our ground-truth numbers to compare against (from PROJECT_STATUS + hardware run):
- Sim (small run, Manila/Lagos fakes): mean |err| raw 0.423, zne 0.391, cdr 0.040
  (11x shots, non-Clifford only), rem 0.102. CDR ~4x-10x better than raw on Manila.
- Real HW (ibm_marrakesh Heron, 2026-07-22, 1024 shots): raw err only 0.016-0.031;
  ZNE WORSE on all 3 circuits; REM on mirror 0.027 -> 0.004 (~7x); raw won once.

---

## DIRECT-OVERLAP (benchmark comparisons of multiple QEM techniques, which-wins-when)

### 1. Russo, Mari, Shammah, LaRose, Zeng — "Testing platform-independent quantum
   error mitigation on noisy quantum computers" (IEEE Trans. Quantum Eng. 2023;
   arXiv:2210.07194)
- https://arxiv.org/abs/2210.07194
- WHAT THEY DID (verified via ar5iv full text): applied ZNE (linear + Richardson)
  and PEC via mitiq to RB circuits and mirror circuits on IBM (Lima, Kolkata),
  IonQ Harmony, Rigetti Aspen-M2 + depolarizing simulators. Defined a
  resource-normalized "improvement factor" mu.
- NUMBERS: n=3: IBM mu ~1-4; Rigetti mostly mu~1 (no help); IonQ several mu<1
  (MITIGATION MADE IT WORSE), esp. Richardson ZNE. PEC mu ~1-2. n=12 (Kolkata)
  mu ~1-3; RB improved more than mirror circuits. Overall "between 1x and 7x".
- RELEVANCE TO US: (a) their mirror-circuit protocol matches ours; (b) mu<1 cases
  on real hardware are published precedent for our "ZNE worse on Heron" result —
  platform-dependent, hence a selector is motivated; (c) their improvement factors
  (1-7x) bracket our REM mirror result (0.027->0.004 ~ 7x). They do NOT build any
  predictor — purely empirical benchmark. THE closest benchmark paper to cite.

### 2. Cirstoiu, Dilkes, Mills, Sivarajah, Duncan — "Volumetric Benchmarking of
   Error Mitigation with Qermit" (Quantum 7, 1059, 2023; arXiv:2204.09725)
- https://arxiv.org/abs/2204.09725 / https://quantum-journal.org (Quantum 7, 1059)
- WHAT THEY DID (verified via ar5iv): volumetric (width x depth grid) benchmark of
  ZNE (exponential fit, folding) vs CDR (linear fit, near-Clifford training) on
  ibm_lagos and ibmq_casablanca; random circuits + Pauli-gadget circuits; ships
  Qermit, an open-source composable QEM package (tket-based).
- NUMBERS: random circuits -> ZNE beat CDR; Pauli-gadget circuits -> CDR beat ZNE
  (fewer non-Clifford gates). Median relative error ~0.5 for ZNE on 4-layer Pauli
  circuits on casablanca; both methods notably worse on ibm_lagos than casablanca
  (device-dependence again). Even at 2x median error reduction, at least one
  sampled circuit was NOT improved by mitigation.
- RELEVANCE: which-wins-when depends on circuit family AND device — this is
  exactly the structure our selector learns. They stop at the empirical map; no
  prediction from static features. Also: their "mitigation sometimes hurts
  individual circuits" = our per-circuit variance story.

### 3. Bultrini, Gordon, Czarnik, Arrasmith, Coles, Cincio — "Unifying and
   benchmarking state-of-the-art quantum error mitigation techniques" (UNITED)
   (Quantum 7, 1034, 2023; arXiv:2107.13470)
- https://arxiv.org/abs/2107.13470
- WHAT THEY DID (verified via ar5iv): benchmarked ZNE, CDR, vnCDR, VD and their
  unified UNITED method on random circuits + QAOA Max-Cut under a realistic
  trapped-ion noise model (simulation only), sweeping qubit count, depth, and
  total shot budget 10^5..10^10.
- NUMBERS: shot budget decides the winner — 10^5 shots: ZNE best; 10^6-10^8:
  vnCDR best; 10^10: UNITED best. Up to 20x improvement over noisy result (RQC,
  10 qubits, 10^10 shots); 5-10x for deep circuits; VD dominates QAOA at
  10^5-10^8.
- RELEVANCE: the canonical "no single technique wins" citation, with shot budget
  as an explicit axis (our cost-aware label / raw_plus control is the same
  concern). Simulation-only, no hardware, no selector. Our 1024-shot hardware ZNE
  failure is consistent with their "ZNE only wins in the low-shot regime — and
  even then modestly" picture.

### 4. Mitiq Calibrator (Unitary Foundation, software, 2023-)
- https://mitiq.readthedocs.io/en/stable/guide/calibrators.html and
  https://unitary.foundation/posts/calibration/
- WHAT IT DOES (verified from docs): `mitiq.Calibrator` RUNS a set of experiments
  on benchmark circuits on your backend, weighs technique/parameter performance,
  returns the best strategy. As of the docs consulted, calibration supports ZNE
  strategies (scale factors, extrapolation methods, folding methods) — i.e. it
  tunes within-ZNE parameters more than it arbitrates across technique families.
- RELEVANCE: THE delineation target named in PROJECT_STATUS §5.4. It selects by
  SPENDING SHOTS at calibration time; our selector predicts from static
  circuit+backend features with zero quantum execution. No paper claims the
  static-feature version — searches for a static-feature technique
  selector/recommender found nothing published (see "gap check" below).

---

## ADJACENT (ML-for-QEM or selection-flavored, but not a static-feature selector)

### 5. Liao, Wang, Sitdikov, Salcedo, Seif, Minev (IBM) — "Machine learning for
   practical quantum error mitigation" (Nature Machine Intelligence 6, 1478-1486,
   2024; arXiv:2309.17368)
- https://www.nature.com/articles/s42256-024-00927-2
- WHAT THEY DID (verified): trained linear regression, random forests, MLPs, GNNs
  to PREDICT MITIGATED EXPECTATION VALUES (mimicking/replacing ZNE), on diverse
  circuit classes, sim + real devices up to 100 qubits. ML matches digital ZNE
  accuracy at much lower runtime cost (fewer circuit executions at inference).
- RELEVANCE: closest big-name ML+QEM paper, and it uses RF like us — but the ML
  REPLACES the mitigation output; it does not choose among techniques. Cite to
  position: "ML as mitigator (Liao) vs ML as meta-selector over mitigation
  techniques (us)". Their framing "reduce cost of mitigation" is also our
  cost-aware angle.

### 6. Scavino — "Decision Kernels for Quantum Error Mitigation: Why Accuracy
   Gains Need Not Improve Downstream Decisions" (arXiv:2607.02888, July 2026)
- https://arxiv.org/html/2607.02888
- WHAT IT DID (verified via abstract fetch): theory + Aer simulations arguing QEM
  should be evaluated/selected by downstream DECISION quality (argmin, ranking,
  top-k), not expectation-value MSE; shows CDR can improve MSE while being
  "decision-flat" and PEC can improve accuracy while worsening decision risk via
  sampling overhead. Decision-aware method selection modestly beat accuracy-based
  selection on held-out tests.
- RELEVANCE: the closest-in-spirit 2026 work on CHOOSING a QEM method — but it
  selects via decision-theoretic evaluation of run data, not prediction from
  static features. Single-author, brand new, unrefereed — treat with caution, but
  its "selection criterion" framing overlaps our label-definition choices
  (best_technique = argmin is exactly the downstream decision it studies).

### 7. Placidi, Williams, Rinaldi, Mills, Cirstoiu, Eccles, Duncan — "Deep
   Learning Approaches to Quantum Error Mitigation" (arXiv:2601.14226, Jan 2026)
- https://arxiv.org/abs/2601.14226
- WHAT THEY DID (verified): seq2seq attention models correct noisy output
  DISTRIBUTIONS toward ideal, on IBM QPUs up to 5 qubits, sim + real data;
  studied which circuit/device/noise features matter and cross-device
  generalization ("same-architecture transfer works without full retraining").
- RELEVANCE: same team as Qermit. Their feature-importance and cross-device
  generalization analysis is the nearest published analogue of our LODO question,
  but again the model IS the mitigator, not a selector. Note: 5-qubit scale =
  ours.

### 8. Koester, Mauerer — "Benchmarking Error Mitigation: Artefactual Improvements
   in Zero-Noise Extrapolation" (arXiv:2607.09360, July 2026)
- https://arxiv.org/abs/2607.09360
- WHAT THEY DID (verified): on IQM Euro-Q-Exa hardware, showed Richardson-ZNE
  "improvements" can be artefacts: when amplified noise passes beyond usable
  signal, extrapolation collapses to a fixed rescaling of one noisy measurement;
  reported estimates overshoot ideal by up to 21%. Includes a "garbage-folding"
  negative control (bigger apparent improvement than real folding!) and a
  reporting checklist for ZNE benchmarks.
- RELEVANCE: strong, very recent support for our hardware finding that ZNE can
  hurt/mislead — and a methodology warning we should self-apply (our
  known-answer mirror circuits are exactly the kind of verifiable control they
  recommend). Cite in the hardware-results discussion.

---

## SUPPORTING (numbers/methodology we compare against or reuse)

### 9. Czarnik, Arrasmith, Coles, Cincio — "Error mitigation with Clifford
   quantum-circuit data" (CDR original) (Quantum 5, 592, 2021; arXiv:2005.10189)
- https://arxiv.org/abs/2005.10189
- NUMBERS: order-of-magnitude (~10x) error reduction for ground-state-energy
  observables on 16 qubits of an IBMQ device and a 64-qubit noisy simulator.
- RELEVANCE: our CDR ~10x on Manila sim (0.209 raw -> 0.012, non-Clifford rows,
  11x shots) is the same order as the original paper's claim — good sanity match.

### 10. Czarnik, Gordon, et al. — "Improving the efficiency of learning-based
   error mitigation" (Quantum 9, 1727, 2025; arXiv:2204.07109)
- https://arxiv.org/abs/2204.07109
- NUMBERS (from abstract/search verification): ~10x cheaper than original CDR at
  equal accuracy; 10x improvement over unmitigated with total budget as small as
  2x10^5 shots (XY-model correlators, IBM Toronto).
- RELEVANCE: shot-budget-aware CDR — the reference point for our CDR 11x
  SHOT_MULTIPLIER honesty and the raw_plus control.

### 11. Nation, Kang, Sundaresan, Gambetta (IBM) — "Scalable Mitigation of
   Measurement Errors on Quantum Computers" (M3) (PRX Quantum 2, 040326, 2021;
   arXiv:2108.12518)
- https://arxiv.org/abs/2108.12518
- NUMBERS: readout error is the dominant noise for low-depth circuits ("even a
  few percent is debilitating"); M3 beats tensored/least-squares mitigation on
  GHZ up to 42 qubits (Brooklyn); ~7 ms vs ~3 s per circuit runtime.
- RELEVANCE: the production REM reference. Our REM mirror-circuit win on Heron
  (0.027 -> 0.004) is consistent with "REM dominates for short circuits". Also a
  reminder that IBM's stack applies M3-style REM by default via primitives —
  worth checking whether our SamplerV2 path already had TREX/M3 off (we ran
  optimization_level=0; resilience settings should be stated in the paper).

### 12. Kim, Wood, et al. (IBM) — "Best practices for quantum error mitigation
   with digital zero-noise extrapolation" (arXiv:2307.05203, 2023)
- https://arxiv.org/abs/2307.05203
- WHAT: IBM's practical guide: folding choices, extrapolation-fit choices, when
  digital ZNE is reliable vs not (deep circuits, low signal).
- RELEVANCE: checklist to diagnose WHY our 1024-shot ZNE lost on Heron (their
  key failure modes: shot noise swamping the fold signal at low shots, and
  extrapolation model mismatch — both plausibly ours).

### 13. "Folding-Free Zero-Noise Extrapolation by Layout-induced Noise Diversity"
   (arXiv:2603.13949, 2026)
- https://arxiv.org/pdf/2603.13949
- NUMBERS (from search verification): 133-qubit Heron, 50-qubit EfficientSU2:
  unmitigated 16.84% from ideal -> ~6% with FF-ZNE; Bell-state ZNE 0.938->0.953;
  ZNE effective for ISA depth ~20-100.
- RELEVANCE: Heron-era magnitude anchor. Their Bell-state raw value 0.938 (err
  ~0.06 at 50q scale) vs our 5q raw errs 0.016-0.031 — consistent with "Heron is
  clean at small scale". Also shows ZNE CAN work on Heron with better protocol —
  our negative ZNE result is a statement about folding-ZNE@1024 shots, not ZNE
  per se; say so in the paper.

### 14. "Few-Shot Cross-Device Transfer for Quantum Noise Modeling on Real
   Hardware" (arXiv:2604.24397, 2026)
- https://arxiv.org/html/2604.24397
- WHAT (from search verification): transfer of learned noise models across
  devices; zero-shot KL 1.67 -> 1.19 with K=20 fine-tune samples (28.6% better);
  identifies CX gate error as the strongest cross-device mismatch feature,
  readout error secondary.
- RELEVANCE: direct published support for our sim-to-real / distribution-shift
  hypothesis AND for which features carry the shift (2q gate error + readout
  error — both in our feature set). Suggests a few-shot fine-tune of the
  selector on a handful of real-hardware rows as a cheap experiment.

### 15. Cai, Babbush, Benjamin, Endo, Huggins, Li, McClean, O'Brien — "Quantum
   Error Mitigation" (Rev. Mod. Phys. 95, 045005, 2023; arXiv:2210.00921)
- https://arxiv.org/abs/2210.00921
- WHAT: the field's authoritative survey — taxonomy of ZNE/PEC/learning-based/
  symmetry/purification methods, in-principle efficacy, sampling-overhead
  scaling, hardware demonstrations.
- RELEVANCE: background citation for the intro + the technique taxonomy our
  selector's classes come from. (No which-wins-when benchmark numbers of its
  own.)

Also noted, lower priority (not fetched in depth):
- "Hypothesis Testing for Error Mitigation: How to Evaluate Error Mitigation"
  (arXiv:2301.02690, 2023) — statistical methodology for claiming "mitigation
  helped"; relevant to our seed-averaged labels / significance question (§6.5).
- Proctor et al., mirror-circuit benchmarking ("Measuring the capabilities of
  quantum computers", Nature Physics 18, 75 (2022); arXiv:2008.11294) —
  provenance of our mirror_circuit family. (Cited from prior knowledge; verify
  exact numbers before quoting.)
- "Error-mitigation aware benchmarking strategy for quantum optimization
  problems" (arXiv:2601.18680, 2026) — QAOA-specific; skim before writing if the
  paper claims breadth.

---

## GAP CHECK (the novelty question from PROJECT_STATUS §5.4)

Multiple searches ("predicting best QEM technique selection classifier",
"'quantum error mitigation' recommender predict which technique classifier
without executing circuits", "...static features") surfaced NO paper that trains
a classifier on static circuit+backend features to predict the best mitigation
technique without quantum execution. Everything found either:
 (a) benchmarks techniques empirically and reports which-won-when (Russo, Qermit,
     UNITED),
 (b) uses ML AS the mitigator (Liao/ML-QEM, Placidi deep-learning, npj QI 2025
     "Noise-agnostic QEM with data augmented neural models"
     https://www.nature.com/articles/s41534-025-00960-y), or
 (c) selects by running calibration experiments (mitiq Calibrator) or by
     decision-theoretic evaluation of run data (Scavino 2026).
The static-feature, zero-shot-at-selection-time selector, with LOFO/LODO
generalization tests and a real-hardware transfer check, still looks unclaimed
as of 2026-07-22. Closest threats: Scavino (2607.02888) for the selection
FRAMING, Liao et al. for ML+QEM mindshare. Write the delineation sentence
against both.

## MAGNITUDE SANITY TABLE (theirs vs ours)

| Source | Setting | Reported | Ours (comparable) |
|---|---|---|---|
| Russo 2023 | ZNE/PEC, IBM/IonQ/Rigetti HW | mu 1-7x, some mu<1 | ZNE<1x on Heron (worse) — precedented |
| UNITED 2023 | sim, 10^5 shots | ZNE best but modest | ZNE weak at 1024 shots — consistent |
| UNITED 2023 | sim, 10^10 shots | vnCDR/UNITED 20x | CDR ~10-17x on Manila sim @11x shots — same order |
| CDR original 2021 | 16q IBMQ HW | ~10x | CDR sim ~10x; CDR not yet run on our HW |
| M3 2021 | GHZ up to 42q | REM largest gain, low-depth | REM 7x on mirror@Heron — consistent |
| FF-ZNE 2026 | Heron 50q | raw 16.8%->6% | raw 1.6-3.1% @5q Heron — cleaner at small n, consistent |
| Koester 2026 | IQM HW | ZNE overshoot up to 21% | ZNE worse on all 3 circuits — same failure family |

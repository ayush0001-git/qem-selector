# Literature search: ZNE failure modes on modern hardware ("zne-limits")

Searched 2026-07-22. Question: we measured ZNE making results WORSE on ibm_marrakesh
(Heron, raw errors 0.016-0.031, 1024 shots, all 3 circuits). Is that known, expected,
or underexplored?

## Bottom line

**Known and expected in principle; our specific angle survives.** The mechanism we
hypothesized (folding+extrapolation variance/bias > removed gate noise when the device
is clean and the shot budget is small) is exactly what 2022-2026 theory predicts, and
2025-2026 papers now demonstrate it empirically — including a May-2026 paper that
literally computes the "help-harm boundary" of finite-shot ZNE. IBM's own docs say ZNE
is "not guaranteed to produce an unbiased result" and that gate folding is "potentially
inaccurate and might lead to incorrect results"; Mitiq's docs say the low-noise points
may be too noisy for extrapolation to beat unmitigated. So we must NOT frame "ZNE can
hurt" as a discovery. What nobody in this list does: predict, from cheap static
circuit+backend features and with no extra QPU spend, WHEN ZNE will hurt on a given
modern device — and quantify the sim-to-real transfer failure of a selector trained on
calibration-snapshot noise. That framing is still open. Cite the papers below as the
expected-behavior baseline and position our result as (a) an independent confirmation
on Heron and (b) motivation for feature-based selection.

## DIRECT OVERLAP (must cite, must differentiate)

1. **The finite-shot help-harm boundary of zero-noise extrapolation** — V. Scavino
   Alfaro, arXiv:2605.08251 (May 2026). https://arxiv.org/abs/2605.08251
   Defines the mean-squared-error crossing where fixed Richardson ZNE flips from
   harmful to helpful at finite shots: ZNE trades noise bias for variance inflated by
   Richardson coefficients + shot splitting. Explicitly: ZNE is counterproductive at
   low shots, shallow circuits, low noise — our exact regime (1024 shots, clean
   Heron). Theory + Aer sims + IBM hardware checks. This is the closest single paper
   to our observation. Differentiator: it derives a boundary for a given
   circuit/observable; it does not learn a cross-circuit selector or study sim-to-real
   transfer.

2. **Benchmarking Error Mitigation: Artefactual Improvements in Zero-Noise
   Extrapolation** — D. Köster, W. Mauerer, arXiv:2607.09360 (Jul 2026).
   https://arxiv.org/abs/2607.09360
   On IQM Euro-Q-Exa: when amplified noise exceeds usable signal, the extrapolation
   collapses into a fixed rescaling of one noisy measurement — a fake "improvement"
   independent of amplification, overshooting the ideal by up to 21%. Proposes a
   matched-cost "garbage-folding" negative control + reporting checklist. Directly
   relevant to our benchmark hygiene: our raw_plus equal-budget control is the same
   philosophy; we should consider adding their negative control for the paper.

3. **Claim against Measurement: Statistical Artefacts in Quantum Error Mitigation
   Benchmarks** — D. Köster, W. Mauerer, arXiv:2605.29872 (May 2026).
   https://arxiv.org/abs/2605.29872
   132-configuration sweep: scale-factor/extrapolant/calibration choices flip ZNE
   conclusions from "significant improvement" to "significant degradation". 72-hour
   drift study: same ZNE config shows >3x different effect size depending on WHEN it
   runs. Reviewed 81 QEM papers; only 25% use inferential statistics. Supports both
   our "ZNE hurt today" result being time/config-contingent AND our drift/sim-to-real
   hypothesis. Sets the reporting bar our paper should meet.

## ADJACENT (same failure mechanism or same hardware class)

4. **Direct Analysis of Zero-Noise Extrapolation: Polynomial Methods, Error Bounds...**
   — P. Mohammadipour, X. Li, arXiv:2502.20673; Quantum 9, 1909 (2025).
   https://arxiv.org/abs/2502.20673
   Rigorous bias+variance bounds for Richardson/polynomial ZNE: extrapolation
   coefficients grow rapidly with node count, exponentially amplifying measurement
   (shot) noise; sample-complexity estimates. The theory behind "folding+extrapolation
   noise > removed gate noise at 1024 shots".

5. **Optimization of Richardson extrapolation for quantum error mitigation** —
   M. Krebsbach, B. Trauzettel, A. Calzona, arXiv:2201.08080; PRA 106, 062436 (2022).
   https://arxiv.org/abs/2201.08080
   Bias-variance analysis of node placement; shows statistical-uncertainty blowup is
   controllable but real. Standard citation for the variance cost of extrapolation.

6. **Best practices for quantum error mitigation with digital zero-noise
   extrapolation** — Majumdar, Rivero, Metz, Hasan, Wang (IBM), arXiv:2307.05203;
   IEEE QCE 2023. https://arxiv.org/abs/2307.05203
   Practitioner guidance across the whole dZNE workflow (folding choice, scale
   factors, extrapolant, composition with other QEM) from simulators + real hardware.
   Cite when justifying our ZNE hyperparameters.

7. **Trapped-ion SU(2) matrix-model simulation on Quantinuum H2-2** —
   arXiv:2604.14094 (2026). https://arxiv.org/abs/2604.14094
   Independent real-hardware data point on a *clean* device: ZNE cut error 72% at
   early times but produced results WORSE than raw at later times. Same qualitative
   phenomenon as ours, different platform (trapped ion), reported in passing rather
   than studied.

8. **Error mitigation, optimization, and extrapolation on a trapped-ion testbed** —
   arXiv:2307.07027; PRA 110, 032416 (2024). https://arxiv.org/abs/2307.07027
   Two of three physical noise-scaling methods failed to scale noise in an
   extrapolatable way at all — evidence that ZNE's core assumption (controlled,
   extrapolatable amplification) can silently fail on real devices.

9. **Reliable high-accuracy error mitigation for utility-scale quantum circuits
   (QESEM)** — Aharonov et al. (Qedma), arXiv:2508.10997 (Aug 2025; rev. Apr 2026).
   https://arxiv.org/abs/2508.10997
   Commercial-grade mitigation validated on IBM Heron + IonQ; explicitly frames ZNE
   as a heuristic "lacking accuracy guarantees" and shows multiple ZNE variants
   underperforming. Practitioner-on-Heron evidence that vanilla ZNE is not the
   default-best on this hardware class.

10. **Machine learning for practical quantum error mitigation** — Liao et al. (IBM),
    arXiv:2309.17368; Nature Machine Intelligence (2024).
    https://arxiv.org/abs/2309.17368
    Random forest mimics ZNE at ~40-50% lower quantum overhead, up to 100 qubits on
    ibm_brisbane. Adjacent to our project overall (ML replaces mitigation output);
    our selector instead predicts WHICH technique wins from static features — keep
    the delineation explicit when citing.

## SUPPORTING (sampling-cost / fundamental-limit theory)

11. **Fundamental limits of quantum error mitigation** — Takagi, Endo, Minagawa, Gu,
    arXiv:2109.04457; npj Quantum Information 8, 114 (2022).
    https://arxiv.org/abs/2109.04457
    Sampling overhead for mitigating layered depolarizing noise scales exponentially
    with depth for general protocols.

12. **Universal sampling lower bounds for quantum error mitigation** — Takagi, Tajima,
    Gu, arXiv:2208.09178; PRL 131, 210602 (2023). https://arxiv.org/abs/2208.09178
    Protocol-agnostic lower bounds (covers nonlinear postprocessing and future
    methods): required samples grow exponentially with depth.

13. **Exponentially tighter bounds on limitations of quantum error mitigation** —
    Quek, Stilck França, Khatri, Meyer, Eisert, arXiv:2210.11505; Nature Physics
    (2024). https://arxiv.org/abs/2210.11505
    ZNE specifically needs samples scaling exponentially in the number of gates in
    the observable's light cone; framework covers ZNE, PEC, CDR, virtual
    distillation. The "why mitigation cannot be free" anchor citation.

14. **Probabilistic error cancellation with sparse Pauli-Lindblad models** — van den
    Berg, Minev, Kandala, Temme (IBM), arXiv:2201.09866; Nature Physics 19 (2023).
    https://arxiv.org/abs/2201.09866
    The gamma sampling-overhead formalism (overhead ~ gamma^2, gamma exponential in
    circuit noise). Basis of IBM's noise-learning stack and of PEA.

## BACKGROUND (guidance docs + provenance)

15. **IBM Qiskit Runtime docs, "Configure error mitigation"** (accessed 2026-07-22).
    https://quantum.cloud.ibm.com/docs/en/guides/configure-error-mitigation
    ZNE "is not guaranteed to produce an unbiased result"; default ZNE = 3 noise
    factors => ~3x overhead; gate folding is "potentially inaccurate and might lead
    to incorrect results" (why IBM built PEA); resilience level 1 = TREX only,
    level 2 adds ZNE + twirling. Note: IBM's own default (level 1) does NOT enable
    ZNE — consistent with our finding that readout-side mitigation is the safer
    default on clean Heron devices.

16. **Mitiq docs, "When should I use ZNE?"** (v0.48+/1.0, accessed 2026-07-22).
    https://mitiq.readthedocs.io/en/stable/guide/zne-2-use-case.html
    States that on real devices of nontrivial depth "the lowest error points may be
    too noisy for the extrapolation to show improvement over the unmitigated result"
    and that a mismatched low-degree fit yields large bias. Our own toolchain's docs
    predict our observation.

17. **Digital zero-noise extrapolation for quantum error mitigation** —
    Giurgica-Tiron, Hindy, LaRose, Mari, Zeng, arXiv:2005.10921; IEEE QCE 2020.
    https://arxiv.org/abs/2005.10921
    The unitary-folding dZNE framework we actually run (via mitiq), including the
    variance dependence on extrapolation choice.

18. **Utility-scale error mitigation with probabilistic error amplification** — IBM
    Quantum tutorial (Heron-era).
    https://quantum.cloud.ibm.com/docs/en/tutorials/probabilistic-error-amplification
    IBM's current recommended ZNE implementation at utility scale is PEA (learned
    noise injection), not gate folding — "gate folding requires large stretch factors
    that greatly limit the depth"; PEA gives an error bound apart from extrapolation
    bias. Relevant caveat for our paper: our ZNE = folding-based dZNE, which is the
    variant IBM itself has moved away from on Heron.

## Implications for our paper

- Frame the Heron ZNE result as *confirming* the finite-shot help-harm regime
  (cite #1, #4, #16) on current-generation hardware, not as a surprise.
- Adopt benchmark hygiene from #2/#3: matched-cost negative control (we have
  raw_plus; consider garbage-folding too), report config sensitivity, note drift.
- The selector story is the novelty: none of these predict per-circuit ZNE
  harm/help from static features, and none quantify calibration-snapshot ->
  live-device transfer failure of such a predictor (our LODO / sim-to-real axis).
- Cost-aware labels get theoretical cover from #11-13 (mitigation cost must be
  priced in) — cite when defending the sqrt shot-penalty label.
- When quoting our ZNE numbers, disclose we used folding-based dZNE at 1024 shots;
  IBM's PEA-based ZNE (#18) might behave differently — honest limitation + future
  work.

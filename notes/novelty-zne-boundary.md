# Novelty scan — ANGLE 3: selector's learned boundary vs. analytic ZNE help-harm boundary

Searcher label: `zne-boundary`. Run 2026-07-22. Hostile-reviewer mode.

**The angle-3 claim under test:** the fact "ZNE hurts on clean low-shot devices" is
KNOWN physics. Our reframe: use the analytic help-harm boundary as GROUND TRUTH and
test whether our purely data-driven selector INDEPENDENTLY learns to refuse ZNE in
exactly the theory-predicted regime — i.e., does the selector's *learned* decision
boundary match the *analytic* ZNE help-harm boundary, on simulation AND on real Heron
hardware? Has anyone done selector-vs-theory-boundary for QEM, or anything QEM-adjacent?

## VERDICT: CLEAR (the boundary-comparison synthesis is unclaimed) — but not "wide open"

No paper compares a learned / data-driven QEM selector's decision boundary against the
analytic ZNE help-harm boundary (or any analytic mitigation-failure threshold). Every
ingredient exists separately; the synthesis — a data-driven selector's learned refuse-ZNE
boundary validated against an analytic help-harm boundary, on sim + real hardware — is
open. Two *conceptual* near-anticipations exist (Decision Kernels theory; GSC-QEMit's
baked-in low-noise heuristic); neither performs the learned-boundary-vs-analytic-boundary
comparison. Word the claim around the *comparison*, not around "nobody knew ZNE can hurt."

## What each threatening hit ACTUALLY did (fetched, not abstract-trusted)

1. **Scavino Alfaro, "The finite-shot help-harm boundary of ZNE"** — arXiv:2605.08251.
   VERIFIED (abs resolves; title/author confirmed). This is our GROUND-TRUTH boundary.
   Purely analytic + Qiskit Aer simulation; IBM Quantum used only for "checks /
   diagnostics", NOT a hardware study. NO machine learning, NO trained classifier, NO
   data-driven selector anywhere. It derives the boundary; it does not test whether any
   learned model recovers it. => This paper *enables* angle 3 (gives us the analytic
   curve to compare against); it does not scoop it. Cite as the theory we validate against.

2. **"Data-driven adaptive quantum error mitigation for probability distribution"** —
   Shimazu, Endo, Hakkaku, Saito, arXiv:2511.13231. VERIFIED (title/authors confirmed).
   "Data-driven adaptive" sounds threatening but is NOT a learned when-to-apply predictor.
   It does two things: (a) N-version programming that excludes outlier QEM outputs, and
   (b) consistency-based extrapolation-point *selection* that picks the L-of-K
   extrapolation combo with lowest variance, bitstring-wise. It selects among extrapolation
   configs BY VARIANCE, not a learned classifier; NO help-harm-boundary comparison; NO real
   hardware. Adjacent (data-driven QEM selection) but a different object. Not a scoop.

3. **Niroula, Gopalakrishnan, Gullans, "Error Mitigation Thresholds in Noisy Random
   Quantum Circuits"** — arXiv:2302.04278 (PRB, qsmz-9kkh). VERIFIED. A *second* analytic
   mitigation-failure boundary (a genuine phase transition below/above which PEC & tensor-
   network mitigation succeed/fail, via an Imry-Ma statistical-mechanics argument under
   imperfect noise characterization). Purely analytic + numerical physics; NO ML, NO
   selector, NO learned-vs-theory comparison. => A SECOND ground-truth boundary we could
   cite/compare against (broadens angle 3 beyond just Scavino's finite-shot ZNE boundary).
   Companion newer paper arXiv:2510.07512 "Error correction phase transition in noisy
   random quantum circuits" is the same physics-threshold lineage — also not a learned
   selector.

4. **GSC-QEMit** (Szachara et al., arXiv:2604.24551, IJCNN 2026) — the closest structural
   threat, already in LITERATURE.md. Re-confirmed here: it is a telemetry-driven
   forecast + contextual bandit choosing mitigation *intensity* (NONE/MODERATE/SEVERE
   — abstract levels, NOT real ZNE/CDR/REM families), Qiskit Aer only, NO hardware. It
   bootstraps the bandit via imitation learning to embed the expert heuristic "avoid heavy
   correction when noise is low." That heuristic is the *conceptual cousin* of "refuse ZNE
   on clean devices" — but it is HAND-CODED as an expert prior, not INDEPENDENTLY LEARNED,
   and is NEVER compared to an analytic help-harm boundary. This is the paper a reviewer
   will wave at us; our differentiation: (i) our selector *learns* the refuse region from
   data rather than being told it; (ii) we *validate the learned region against an analytic
   boundary*; (iii) real Heron hardware; (iv) real technique families.

5. **Decision Kernels for QEM** (Scavino, arXiv:2607.02888) — the other conceptual
   near-anticipation. Provides THEORY for when mitigation should be declined and finds
   "experiments repeatedly recommend retaining Raw" — i.e. theory-says-decline-ZNE-here
   already exists as an idea. But it builds NO selector and makes NO learned-boundary-vs-
   analytic-boundary comparison. It is also a DO-NOT-CLAIM caveat for angle 2's labels
   (already flagged in LITERATURE.md §3).

## The methodology "learned boundary vs analytic threshold" is NOT novel in the abstract

In the ADJACENT field of quantum error *correction* (decoding), comparing a learned
classifier's decision boundary to an analytic threshold (e.g. neural-net / SVM decoders
vs the MWPM threshold) is established practice. This does NOT scoop angle 3 (QEC decoding
!= QEM technique selection, and the object compared is a correction-threshold not a
mitigation help-harm boundary), but it means we must NOT claim "first ever to compare a
learned boundary to an analytic threshold" in the abstract. Claim the QEM-*specific*
version: "first to test whether a data-driven QEM *selector* independently reproduces the
analytic ZNE help-harm boundary."

## Also seen, not threats
- ML-QEM (Liao 2309.17368), Korolev (2606.02697), GEM (2604.16815), Q-LEAR (2404.12892),
  Placidi (2601.14226): all learn to *perform* mitigation; none learn *when to refuse* it
  or compares to a theory boundary. (All already in LITERATURE.md.)
- Pauli-weight term selection for ML-QEM (arXiv:2606.31195): feature/term selection for a
  mitigator, not a when-to-apply predictor.
- Various 2026 Heron QEM papers (ZEPE, QESEM, PEA): new mitigators / benchmarks, no learned
  when-to-mitigate boundary.

## What remains OPEN (the defensible angle-3 contribution)
A purely data-driven, feature-driven QEM selector whose *learned* decision to refuse ZNE
is quantitatively compared against the *analytic* finite-shot help-harm boundary (Scavino
2605.08251) — and optionally the error-mitigation threshold (Niroula 2302.04278) — as
ground truth, demonstrated on BOTH simulation AND real IBM Heron hardware. Nobody has done
this. Frame it as an *external theoretical validation* of the selector (the selector's
learned boundary agreeing with independently-derived physics is EVIDENCE the selector
learned real structure, not spurious correlations) rather than as discovering that ZNE can
hurt (which is known — cite Scavino, Russo, Köster-Mauerer, Mohammadipour-Li).

### Hostile-reviewer caveats to pre-empt
- "ZNE hurts at low shots / clean devices is already known" — YES; we cite it and use it as
  ground truth, we do not claim it as a finding (already a DO-NOT-CLAIM in LITERATURE.md §3).
- "GSC-QEMit already avoids heavy mitigation at low noise" — but as a hand-coded imitation
  prior, not learned-then-validated-against-theory; and no hardware/no families.
- "Learned-boundary-vs-analytic-threshold is old (QEC decoders)" — true methodologically;
  scope the claim to the QEM-selector-specific instance.
- Boundary-match is only as good as the analytic boundary's assumptions (fixed Richardson,
  finite-shot). Disclose that Scavino's boundary is for a specific ZNE variant; our
  off-the-shelf folding-dZNE@1024-shots must match that variant for the comparison to be
  fair (ties into LITERATURE.md gap §5.3).

# Deep Reader — Batch 4 (literature overlap check)

Written 2026-07-22. Six papers flagged as potential overlap with the QEM-Selector
work. For each: what they ACTUALLY did (not abstract spin), extractable numbers,
and a skeptical verdict on which parts of OUR contribution they cover vs leave open.

Our contribution, as the yardstick:
- (a) learned per-circuit technique selector (raw/ZNE/CDR/REM) from STATIC features,
      picked WITHOUT spending shots, with honest grouped/LOFO/LODO eval
- (b) noise-strength dimension via scaled fake backends (@x1.5, @x2.0)
- (c) equal-budget `raw_plus` (raw at 11x shots) fairness baseline
- (d) sim-to-real transfer test of the sim-trained selector on 2026 Heron hardware
- (e) CDR regressor swaps (linear -> polynomial / sklearn regressors)

---

## 1. Sack & Egger — "Large-scale QAOA on non-planar graphs with ML noise mitigation"
PRR 6, 013223 (2024) · arXiv:2307.14427 · https://arxiv.org/abs/2307.14427

**What they actually did.** A feed-forward neural network (FFNN) is trained as a
learned noise-inverse MAP: input layer has n + n(n-1)/2 neurons (single-qubit
<Z_i> and two-qubit <Z_iZ_j> correlators), output layer |E| neurons (one per graph
edge), hidden layer = average of the two. It is retrained PER QAOA optimization
iteration on ~3000 (300 for ibm_kyiv) classically-simulable "random product state"
training circuits (superposition swapped for random X-partition; RZ replaced by
barriers to keep noise structure but stay simulable), 1024 shots each. It mitigates
EXPECTATION VALUES, not bitstrings — the abstract's "need to mitigate samples, not
only expectation values" is a stated LIMITATION/future direction, not what their
method does. Applied to depth-2 QAOA on non-planar random-regular graphs up to 40
nodes via SAT-based swap-network qubit mapping, on IBM Brisbane/Kyiv/Nazca (Eagle).

**Key numbers.** 40q graph = 958 ECR + 1021 1q + 1275 RZ gates; mitigated approx
ratio a(mu) 58.3% (58.6% pulse-efficient) vs 72.4% noiseless; best cut 44/60
(0.786), seen 988/12288 shots. 10q 71.6% vs 82.8% noiseless; 20q 64.0%; 30q 59.6%.
Path fidelity 95.9% (10q) -> 76.5% (40q). Sim: mean energy -3.78+/-2.12 (mitigated)
vs -2.63+/-2.01 (unmitigated), 20 runs; FFNN R^2 = 71.8% (10-node).

**Overlap verdict.** This is ML-QEM of the "learn a correction map" flavor (a
nonlinear cousin of CDR) — it uses ML to REPLACE a mitigation technique, NOT to
CHOOSE among techniques. It does not touch our selector (a): no feature-based
technique choice, no shot-free prediction — its FFNN spends ~3000x1024 shots
retraining every iteration. Covers none of (b) noise-strength axis (one hardware
noise level), (c) equal-budget control, (d) sim-to-real selector transfer (it is
all real hardware, no sim-trained model transferred). Its closest tie is to (e):
its FFNN is essentially a nonlinear/ML generalization of a CDR-style fit, so it is
a REFERENCE POINT / inspiration for our CDR-regressor-swap angle, not overlap.
Value: the canonical application-scale ML-QEM citation (40q, 958 2q gates) proving
learned mitigation maps work at scale — establishes ML-QEM legitimacy but does NOT
pre-empt a static-feature selector.

---

## 2. Scavino Alfaro — "The finite-shot help-harm boundary of zero-noise extrapolation"
(2026) · arXiv:2605.08251 · https://arxiv.org/abs/2605.08251

**What they actually did.** Analytic derivation (local expansion) of a finite-shot
"help-harm boundary": the lower local-MSE crossing where FIXED Richardson ZNE flips
from harmful to helpful. The boundary is governed by the first squared-bias
improvement (ZNE's benefit) vs the first excess-variance penalty (Richardson-
coefficient variance inflation + shot splitting), producing one of three regimes:
a shrinking power law, a budget threshold, or NO shrinking lower boundary. Validated
on Qiskit Aer (variance-exponent fits) and IBM Quantum, separating deterministic
stabilizer measurements from variational energy measurements; readout-regime
diagnostics bound measurement-protocol / hardware-traceability limits. 22 pp, 4 figs.

**Key numbers.** Primarily analytic (boundary = first squared-bias improvement vs
first excess-variance penalty). Few hardware scalars in the abstract; deterministic-
stabilizer vs variational-energy separation confirmed by variance-exponent fits.

**Overlap verdict.** This is the closest THEORETICAL explanation of our headline
empirical finding (ZNE hurts at low shots, shallow circuits, low noise). But it
computes a PER-OBSERVABLE, per-circuit analytic boundary for a SINGLE technique
("use ZNE here or not") — it does NOT learn a cross-circuit selector from static
features (a), does not span multiple techniques (no CDR/REM), has no scaled-noise
design axis (b), no equal-budget raw_plus control (c), and frames Aer+IBM as
validation rather than a sim-to-real selector-transfer test (d). It STRENGTHENS our
story: it is the physics/analytic boundary that our learned selector approximates
empirically, generalized across techniques and circuits WITHOUT computing per-
observable bias/variance. Cite as "the analytic single-observable boundary; our
selector is the learned multi-technique, feature-based, shot-free approximation."
Leaves the whole selector problem open.

---

## 3. Koster & Mauerer — "Benchmarking Error Mitigation: Artefactual Improvements in ZNE"
(2026) · arXiv:2607.09360 · https://arxiv.org/abs/2607.09360

**What they actually did.** On IQM Euro-Q-Exa (54q, qubits 8-11), 4-qubit Trotter
circuit, parity <Z^4> observable (floor 0), scale factors {1,3,5}, 4096 shots: show
that once amplified noise drives E(lambda) to the observable floor, Richardson ZNE
stops reflecting physics and COLLAPSES into a fixed rescaling of a single noisy
measurement, manufacturing "improvement." Proposes (i) a matched-cost GARBAGE-
FOLDING negative control — inserted folds that do NOT reduce to identity
(U(U^dU) != U vs genuine U(U^dU)=U), identical gate counts at every lambda but
signal destroyed — which reports a LARGER apparent improvement than genuine folding;
(ii) a zero-cost negative-probability diagnostic W_neg = sum over states with
P_hat(s)<0 of |P_hat(s)|, computable from counts already collected; (iii) a 3-item
ZNE reporting checklist.

**Key numbers.** Overshoot up to 21%: at depth d=3, E_hat(0)=1.16 vs E_ideal=0.96.
Scale factors {1,3,5}; 4096 shots; IQM Euro-Q-Exa 54q, qubits 8-11; 4q QTC.
Checklist: (1) report whether E(lambda) still retains signal at lambda>1 or has hit
the observable floor; (2) report per-basis-state negative-probability weight W_neg
alongside E^(0); (3) report improvement vs the physically attainable maximum and
flag overshoots.

**Overlap verdict.** Benchmark-hygiene / negative-control paper, NOT a selector. It
VALIDATES our methodology philosophy: our `raw_plus` equal-budget control (c) is the
same SPECIES of matched-cost negative control as their garbage-folding, and our real-
hardware result "ZNE made results WORSE on all 3 circuits" sits in exactly the
artifact regime they formalize. It raises the bar our paper must clear (report
signal-at-lambda, W_neg, overshoot flags) and offers a control we could add
(garbage-folding). It does NOT cover (a) selector, (b) scaled-sim noise axis, (d)
sim-to-real, or (e) CDR. Overlap with (c) is partial/complementary: theirs is a
garbage-SIGNAL control for ZNE specifically; ours is an equal-SHOT-BUDGET control
across techniques. Caveat on transferability: our Heron finding is at 1024 shots
with LOW raw error (0.016-0.031) — a low-shot variance mechanism, related to but
not identical to their signal-floor collapse (they amplify noise past the floor;
we barely have bias to remove). Cite as the hygiene standard our design anticipates;
consider adopting the checklist + garbage-folding control.

---

## 4. Koster & Mauerer — "Claim against Measurement: Statistical Artefacts in QEM Benchmarks"
(2026) · arXiv:2605.29872 · https://arxiv.org/abs/2605.29872

**What they actually did.** Meta-study + two empirical demonstrations. Survey of 81
recent QEM papers: only 15 (25%) use inferential statistics, 25 (42%) report
uncertainty only descriptively. A 132-configuration sweep over {scale factors x
extrapolants x hardware calibration settings} shows these choices are ACTIVE, not
incidental — they flip ZNE conclusions from statistically significant IMPROVEMENT to
statistically significant DEGRADATION for the same underlying method. A 72-hour
real-hardware drift study shows the SAME ZNE config's effect size varies >3x
depending solely on WHEN it runs, and drift drastically cuts the effective number of
independent observations (false illusion of consistent effectiveness).

**Key numbers.** 81 papers surveyed; 15 (25%) inferential; 25 (42%) descriptive-only;
132 configurations; 72-hour study; effect size >3x from temporal drift alone.

**Overlap verdict.** Statistical-honesty / reproducibility paper, not a selector.
Directly SUPPORTS our sim-to-real hypothesis (d): their >3x drift-driven effect-size
swing is the temporal analogue of our claim that a selector trained on calibration
SNAPSHOTS may not transfer to today's cleaner Heron hardware; their 132-config
sensitivity mirrors why our honest grouped/LOFO/LODO eval and equal-budget control
matter. Sets minimum reporting standards our paper should adopt: report CIs/effect
sizes not point estimates, robustness across ZNE knobs, longitudinal drift,
inferential tests. Covers NONE of our positive contributions — no learned selector
(a), no scaled-sim axis (b), no raw_plus baseline (c), no CDR swaps (e) — and studies
ONE ZNE config's stability, not cross-circuit technique choice. Overlap = methodo-
logical guardrails only: it defines the bar; we clear a different research question.
Our own +/-0.09 fold-noise disclosure and grouped CV already align with its asks.

---

## 5. Mohammadipour & Li — "Direct Analysis of Zero-Noise Extrapolation: Polynomial Methods, Error Bounds"
Quantum 9, 1909 (2025) · arXiv:2502.20673 · https://arxiv.org/abs/2502.20673

**What they actually did.** Rigorous bias+variance theory for Richardson/polynomial
ZNE. Prove the Lagrange-coefficient l1-norm ||gamma||_1 (= the variance amplification
factor) grows EXPONENTIALLY in the number of nodes n. Give explicit bias bound,
variance bound, sample-complexity estimates, and propose a polynomial LEAST-SQUARES
alternative (fit degree m < n) that turns the variance growth from exponential to
polynomial and curbs overfitting; also extend to joint noise + Trotter-step scaling.

**Key numbers.** Equidistant nodes: ||gamma||_1 = O((2Be/(B-1))^n). Chebyshev nodes:
O(kappa^{2n}), kappa=(sqrt(B)+1)/(sqrt(B)-1), B = max noise amplification. Variance:
Var[p_n(0)] <= (sigma^2/N_S)(sum|gamma_j|)^2. Sample complexity (Chebyshev):
N_S = Omega(alpha^2 * eps^{-(2+4 log kappa)} * log(2/delta)); equidistant exponent
-(2 + 4Be/(B-1)) is far worse. Bias: |f(0)-p_n(0)| <= C M^{n+1}/(n+1)! * prod x_j;
n = Omega(log(1/eps)) nodes suffice. Least-squares: ||gamma||_1 = O(kappa^{2m}),
m = Omega(log(1/eps)) — polynomial not exponential.

**Overlap verdict.** This is the THEORETICAL MECHANISM behind our finding "folding+
extrapolation noise > removed gate noise at 1024 shots." Their Var <= (sigma^2/N_S)*
Lambda^2, with Lambda blowing up (exponentially for equidistant nodes), is exactly
why ZNE inflates variance at low shots and therefore HURTS when the removed bias is
small — our clean-Heron regime. But it is pure single-technique estimator analysis:
covers none of our contributions directly — no learned selector (a), no cross-
technique comparison, no static-feature shot-free prediction, no scaled-noise design
axis (b), no equal-budget cross-technique control (c), no sim-to-real (d). Its
polynomial-least-squares idea is ADJACENT to our (e) CDR-fit-swap ablation in spirit
(replace an interpolating/linear fit with a regularized regressor) — worth citing
when we motivate (e). Best used as the citation that justifies WHY our selector
correctly avoids ZNE at low shots. Leaves the entire selector problem open.

---

## 6. Krebsbach, Trauzettel, Calzona — "Optimization of Richardson extrapolation for QEM"
PRA 106, 062436 (2022) · arXiv:2201.08080 · https://arxiv.org/abs/2201.08080

**What they actually did.** In-depth bias-variance analysis of Richardson NODE
PLACEMENT. With optimal shot allocation N_j proportional to |gamma_j|, variance =
(sigma^2/N_tot) * Lambda^2 where Lambda = sum|gamma_j| is the minimal measurement
overhead (Lambda^2 >= 1). Key result: Lambda can be controlled INDEPENDENTLY of node
count n — removing the exponential variance blowup that had limited prior practice to
n<=4. Propose novel "tilted Chebyshev" nodes that shrink the bias constant C_n vs
standard node families at no extra cost.

**Key numbers.** Var[R_hat_n] = (sigma^2/N_tot) * Lambda^2, Lambda = sum|gamma_j|,
Lambda^2 >= 1 minimal overhead. Bias = (-1)^n E^{(n+1)}(xi) C_n/(n+1)!. Tilted-
Chebyshev nodes at n=7, Lambda=10: bias constant C_n ~1.25x smaller than extremal
Chebyshev, ~2x smaller than exponential, ~35x smaller than linear nodes. Prior
linear-node variance grew exponentially with n (practically n<=4).

**Overlap verdict.** Foundational ZNE-variance-cost citation, purely about optimizing
ONE technique's estimator (node placement + shot allocation). Standard reference for
"extrapolation inflates statistical uncertainty," so it backs our raw_plus / equal-
budget concern and our finding that ZNE's variance cost can exceed its bias benefit.
Orthogonal to every one of our contributions: no selector (a), no multi-technique
benchmark, no scaled-sim axis (b), no equal-budget cross-technique control (c) (it
optimizes ZNE's own overhead Lambda, not a fairness baseline vs raw), no sim-to-real
(d), no CDR swap (e). IMPORTANT CAVEAT for our narrative: it shows ZNE variance CAN
be tamed (Lambda controlled independent of n, tilted-Chebyshev nodes) — meaning our
negative ZNE result is partly an artifact of mitiq DEFAULT node placement (3x shots,
default/linear-ish nodes). Our "ZNE hurts" story should honestly note that optimized
node placement per this paper would narrow (not necessarily erase) the gap. Leaves
the selector question fully open.

---

## Cross-cutting takeaways for the paper

1. NONE of the six builds a learned, static-feature, shot-free technique SELECTOR
   across raw/ZNE/CDR/REM. Our contribution (a) survives the batch intact. Sack &
   Egger is ML-QEM-as-correction-map (replace a technique), not selection.

2. Three papers (2, 5, 6) give us the THEORY for our empirical "ZNE hurts at low
   shots/low noise" result. Scavino Alfaro (2) is the finite-shot MSE boundary;
   Mohammadipour-Li (5) and Krebsbach (6) are the variance-amplification mechanism
   (Var ~ Lambda^2/N_S, Lambda grows with nodes). Cite all three to ground our
   real-hardware ZNE regression physically instead of just empirically.

3. Two papers (3, 4) set the BENCHMARK-HYGIENE bar. Our raw_plus (c), honest grouped/
   LOFO/LODO eval, and drift/sim-to-real hypothesis (d) already align with their
   asks. Actionable adds for the paper: (i) adopt their reporting checklist
   (signal-at-lambda, negative-probability weight W_neg, overshoot flags);
   (ii) consider a garbage-folding negative control alongside raw_plus;
   (iii) report effect sizes + CIs and be explicit about winner's-curse / drift.

4. HONESTY CAVEAT surfaced by paper 6: our ZNE used mitiq defaults. Optimized node
   placement + shot allocation (Krebsbach; Mohammadipour-Li least-squares) could
   improve ZNE's variance and partly reduce our "ZNE always worse on Heron" gap. The
   paper must state that our ZNE is the OFF-THE-SHELF configuration, not the best
   possible ZNE — otherwise a reviewer citing [2201.08080]/[2502.20673] will.

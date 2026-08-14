# Angle 3 deep-dive — "boundary-recovery": does a usable ANALYTIC ZNE help-harm boundary exist?

Searcher label: **boundary-recovery**. Date: 2026-07-22.
Scope: verify whether Angle 3 is *executable* — i.e. whether there is a concrete,
computable analytic mitigation-failure boundary we can overlay on our selector's
learned decision surface as ground truth. Hostile-reviewer stance throughout.

**One-line verdict: YES, an explicit closed-form help-harm boundary exists (and is now
a whole theory program, not one paper). Angle 3 is executable. The boundary is
computable in SIMULATION and against KNOWN-ANSWER benchmark circuits, but NOT from
static features alone — the bias side needs the ideal value μ₀ — which is actually what
makes the "does the feature-only selector independently recover it" test meaningful.**

All arXiv IDs below were fetched (abs + html) and resolve with the titles shown.

---

## 1. The primary target: Scavino Alfaro, arXiv:2605.08251 (May 2026)

"The finite-shot help-harm boundary of zero-noise extrapolation", V. Scavino Alfaro.
Fetched abs + `arxiv.org/html/2605.08251`. Has a Zenodo validation supplement v1.0.1.

**It DOES give an explicit closed form.** The MSE decomposition and boundary:

- `MSE(μ̂) = Bias(μ̂)² + Var(μ̂)`
- Help-harm boundary defined by the zero crossing of the MSE difference:
  `ΔMSE(ε,B) = MSE_noisy(ε,B) − MSE_ZNE(ε,B) = 0`
- Local-expansion form:
  `ΔMSE(ε,B) = D_p·ε^(2p) − K_q·ε^q / B + R_p(ε,B)`
  - `ε` = noise strength, `B` = shot budget, `p` = leading bias power (observable/noise
    property), `q` = effective measurement-variance exponent (protocol property).
  - `D_p` = leading squared-bias improvement coefficient.
  - `K_q` = variance-penalty constant (Richardson coefficients × scale factors × shot
    allocation).
- **Two closed-form regimes** (this is the "shrinking power law / budget threshold / no
  boundary" trichotomy from the abstract):
  - **Budget threshold** (q = 2p): `B* = K_q / D_p`. Below `B*` ZNE harms, above it helps.
  - **Shrinking power law** (0 ≤ q < 2p):
    `ε*(B) ∼ (K_q/D_p)^(1/(2p−q)) · B^(−1/(2p−q))`.
  - No shrinking lower boundary when the variance exponent is too large.

So we have an actual curve in the **(ε, B) = (noise-strength × shot-budget) plane** we can
plot. That is exactly the plane a "refuse ZNE at low shots / low noise" decision boundary
lives in.

**Hardware validation already in the paper.** Its validation supplement uses Qiskit Aer +
variance-exponent fits and "readout-regime diagnostics and IBM Quantum checks." So Scavino
*already validated that his boundary predicts the measured MSE crossing*, including on IBM
hardware. What he did NOT do: test whether a *data-driven feature selector* recovers it
(that gap = our angle).

### The load-bearing caveat (determines HOW we execute, not IF)

`K_q` (variance side) is **computable a priori** — Mohammadipour & Li (below) confirm the
Richardson/Lagrange coefficients depend only on the chosen noise-scale factors.

`D_p` (bias-improvement side) is **NOT** computable from static pre-execution features —
it "depends on unmitigated and mitigated bias expansions," i.e. it needs the ideal value
μ₀ (or a noise-model bias model). WebFetch quote: *D_p "requires knowing the ideal
expectation value μ₀ ... though bias coefficients can sometimes be estimated from data."*

Consequence for Angle 3:
- **In simulation** we know μ₀ and the noise model → we can compute `D_p`, `K_q` → plot the
  analytic ΔMSE=0 curve → overlay the selector's learned refuse-ZNE region. Fully executable.
- **On real Heron hardware** we cannot compute `D_p` blind. BUT with **known-answer
  benchmark circuits** (mirror circuits, Clifford/near-Clifford) we know μ₀, so we can
  measure the **empirical** help-harm boundary directly (grid over noise-scale × shots,
  find where measured MSE_ZNE crosses measured MSE_noisy). Executable. The clean-story
  framing on hardware is *selector-vs-empirical-boundary*, with the analytic curve as a
  simulation-anchored overlay.
- **The independence is the point.** The selector sees only static features (execution-free);
  the theory boundary needs μ₀, which the selector never gets. So "does the feature-only
  selector's boundary coincide with the μ₀-dependent analytic boundary" is a genuine,
  non-circular validation — not the selector regurgitating an input.

---

## 2. The mechanism paper: Mohammadipour & Li, arXiv:2502.20673 (Feb 2025, rev Nov 2025)

"Direct Analysis of Zero-Noise Extrapolation: Polynomial Methods, Error Bounds, and
Simultaneous Physical-Algorithmic Error Mitigation", Quantum 9:1909. Fetched html.

Gives the **a-priori-computable variance side** the Scavino boundary needs:

- `Var[p̂_n(0)] ≤ (σ²/N_s)·(Σ_j |γ_j|)²`, with `γ_j = L_j(0) = Π_{k≠j} x_k/(x_k − x_j)`
  the Lagrange coefficients — **depend only on noise-scale nodes {x_j}, not on bias/μ₀**.
- Coefficient growth: `‖γ‖₁ = O(κ^{2n})` (Chebyshev nodes, κ=(√B+1)/(√B−1)) vs
  `O((2Be/(B−1))^n)` (equidistant, exponentially worse).
- Sample complexity (Thm 5): `N_s = Ω(α²·ε^{−(2+4log κ)}·log(2/δ))`.
- Bias bound (Thm 2) needs unknown derivative constants C, M → confirms the bias side is
  the non-a-priori part, consistent with Scavino's `D_p` caveat.

**Use:** this is what lets us actually numerically build the `K_q` half of the boundary for
our specific ZNE config (nodes, allocation). It does NOT itself give an MSE-crossing
inequality — WebFetch: "The paper does not provide an explicit formula where MSE exceeds the
unmitigated estimator's MSE." So the *boundary* comes from Scavino; the *variance
ingredients* come from here.

---

## 3. Köster & Mauerer, arXiv:2607.09360 (Jul 2026) — NO analytic boundary

"Benchmarking Error Mitigation: Artefactual Improvements in ZNE." **Purely empirical.**
No closed-form threshold. Contribution = a *failure mode* (Richardson ZNE collapses to a
fixed rescaling of one noisy measurement in the deep/high-amplification regime → bogus
apparent improvement, overshoot up to 21% on IQM), plus the **garbage-folding matched-cost
negative control** and a reporting checklist. This is depth-driven and confirms the physics,
but supplies no formula. Do NOT cite it as "the boundary"; cite it as (a) empirical
corroboration of harm and (b) the negative-control idea.

---

## 4. The full boundary PROGRAM (this is bigger than one paper — reviewers will know it)

Scavino has published a **series** of finite-shot operating-window / help-harm papers in
2026. All fetched and confirmed:

| arXiv | Title | Boundary form given |
|-------|-------|---------------------|
| 2605.08251 | Finite-shot help-harm boundary of **ZNE** | `ΔMSE=D_p ε^{2p} − K_q ε^q/B`; `B*=K_q/D_p`; power law `ε*(B)` |
| 2606.21686 | Finite-shot operating windows for **PEC and CDR** | crossover `B_{PEC=CDR} ∝ 1/(δ₁² p)`, `δ₁`=first-order CDR calibration mismatch |
| 2606.15464 | Certified operating windows for **Virtual Distillation & Symmetry Verification** | MSE law w/ non-asymptotic remainders; validated `p^{-2}` window scale |
| 2607.02888 | **Decision Kernels for QEM** (why accuracy ≠ decision quality) | quotient-space theory, marginal no-go + QEM-pullback theorems; does *decision-aware selection* vs accuracy-based on static held-out data |

Reading: there is now an analytic help-harm/operating-window boundary for **every** major
QEM family (ZNE, PEC, CDR, VD, SV). Great news for Angle 3's supply of ground truth — we can
overlay boundaries for multiple selectable techniques, not just ZNE. Bad news for framing:
the boundary is **not our discovery** and is heavily developed; we must cite the whole
program and position ourselves strictly as "data-driven selector *recovers* it," never as
introducing it.

**The one real adjacency threat = Decision Kernels (2607.02888).** Scavino there already
formalizes QEM *selection* (argmin/ranking) and shows "decision-aware selection modestly
reduces static held-out failure vs accuracy-based selection." That is theory-driven
selection on static held-out data — uncomfortably close to our selection framing. But it is
NOT a data-driven, feature-based, execution-free ML selector whose *learned decision
boundary* is compared to the *analytic help-harm boundary* on sim + hardware. That specific
comparison remains unclaimed.

---

## 5. Did anyone already do "learned selector boundary vs theory boundary" for QEM?

Ran 8 distinct hostile queries (selector-vs-boundary, ML-recovers-theory-boundary, ZNE
phase diagram/regime map, predict-ZNE-help-from-features classifier, experimental 2D
shots×noise harm map). **No hit.** Nobody:
- trains a classifier to predict ZNE help/harm from static circuit/device features, or
- overlays a learned QEM decision surface on the Scavino analytic boundary, or
- maps the help-harm boundary as a 2D regime diagram *and* compares it to a learned policy.

**Methodological precedent to CITE (different domain, not a scoop):** in quantum *many-body
phase* classification, showing that an ML classifier's learned decision boundary matches the
theoretical phase boundary is an established validation motif (e.g. neural detection of
quantum phases, arXiv:2205.12966). Cite this so the *method* ("learned boundary vs analytic
boundary") doesn't look unprecedented — it just hasn't been applied to QEM technique
selection.

---

## 6. Executability verdict for Angle 3

**Executable = YES.** A usable analytic boundary exists (§1), its computable ingredients
exist (§2), and the selector-recovers-boundary comparison is unclaimed (§5).

Concrete plan the boundary supports:
1. Fix a circuit family + observable + noise structure → the boundary is the ΔMSE=0 curve in
   the **(noise-scale ε, shot-budget B)** plane. Our selector already sweeps noise-scale and
   shots, so it spans this plane — take 2D slices with other features held fixed.
2. **Simulation:** compute analytic boundary from known μ₀ + noise model (`D_p`, `K_q`);
   overlay the selector's learned "raw/raw_plus vs ZNE" decision region; report boundary
   agreement (e.g. IoU / signed distance of the selector's ZNE-refuse frontier to ΔMSE=0).
3. **Real Heron:** use known-answer benchmark circuits (mirror/Clifford) → measure the
   *empirical* MSE-crossing boundary directly; overlay selector; the analytic curve rides
   along as a sim-anchored reference. Frame hardware as motivating evidence (small n), per
   the existing SEV-1/SEV-2 caveats in LITERATURE.md.

### Hostile-reviewer landmines (must design around these)
- **ZNE-config mismatch.** Scavino's boundary assumes *fixed Richardson* coefficients with a
  specific per-noise-level shot allocation. Our Mitiq off-the-shelf **folding ZNE at 1024
  shots** must be configured to match those assumptions (fixed nodes, split shots) or the
  overlay is apples-to-oranges. This is the single biggest execution constraint.
- **Depth is not a clean third axis.** The closed form is 2D in (ε, B). Circuit depth enters
  only *through* ε and the bias coefficients `D_p`; the depth-driven collapse (Köster-
  Mauerer) is empirical, outside the formula. So sell it as a *(noise × shots)* boundary per
  family, not a 3D (shots × noise × depth) surface. Don't claim depth is an explicit boundary axis.
- **μ₀-dependence of D_p.** Anticipate "your ground truth needs the ideal value you claim not
  to have." Answer: selector is feature-only/execution-free; boundary computed in sim (μ₀
  known) or from known-answer benchmark circuits on HW; the independence is the validation,
  not a leak.
- **Scavino already checked IBM.** He validated the *theory*; he did not test whether a
  *data-driven selector* recovers it. State this delta explicitly and cite the whole program.

### Classification
**CLEAR (open)** for the specific reframe — a purely data-driven, feature-driven,
execution-free QEM selector whose *learned* decision boundary is validated against the
*analytic* help-harm boundary, on simulation and real Heron hardware. Caveat it heavily:
the boundary theory is mature and crowded (Scavino ×4), and Decision Kernels already does
theory-driven selection, so the novelty is *narrow and must be worded exactly* — "the
selector *independently learns* the theory-predicted boundary," never "we derive/discover
the boundary."

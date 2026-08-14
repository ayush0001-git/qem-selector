# Two Reframed Research Angles — Adversarial Novelty Verdict & Plan

Written 2026-07-22 by the adversarial-judge pass. Audience: you (final-year AI/ML,
quantum beginner). Read `docs/LITERATURE.md` first — this file does **not** re-derive
the landscape; it answers one question for each of your two reframed angles:

> **Can we still claim it, exactly how do we word it, and what will a reviewer throw?**

I personally re-fetched the three closest threats (Korolev 2606.02697, Scavino's
help-harm boundary 2605.08251, Decision Kernels 2607.02888) and ran five hostile
"try-to-reject-this" searches. Findings and the two nuances that changed are logged at
the bottom (§ "What re-verification changed"). Bottom line up front: **both angles
survive, both need surgical wording, and they are stronger as two sections of ONE
paper than as two papers.**

---

## Plain-language recap of the two angles

You already have a **selector**: given static features of a circuit + backend (no
quantum runs at decision time), a classifier predicts which QEM technique family
(raw / ZNE / CDR / REM) will give the lowest error. The two angles below are *extra
claims* you can bolt onto that selector.

- **Angle 2 (CDR, reframed):** add "CDR-with-a-nonlinear-regressor" as an *extra
  technique the selector can pick*, and map out *when* the selector should prefer
  nonlinear-CDR over plain linear-CDR.
- **Angle 3 (ZNE boundary):** there is a physics equation that says exactly when ZNE
  will *hurt* rather than help. Test whether your data-only selector, which never sees
  that equation, independently learns to refuse ZNE in the same region the equation
  predicts.

---

# ANGLE 2 — CDR regressor choice as a selectable technique

**Verdict: PARTIALLY-OPEN.** The naive version ("swap CDR's linear fit for sklearn
regressors, see which wins") is **closed** by Korolev et al. (June 2026). The reframed
version survives — but it is the *more incremental* of your two angles, so word it
tightly and lean on the sim sweep you already have running.

### The exact defensible claim (copy this wording)

> We embed the **CDR regressor choice** (linear vs. nonlinear) as an additional
> **per-circuit selection target** inside an execution-free QEM technique selector, and
> **characterize when** nonlinear-CDR beats linear-CDR across **five heterogeneous,
> largely non-VQE circuit families** and on **real IBM hardware**, mapping the
> crossover as a joint function of **training-set size × non-Clifford fraction**. We
> **reproduce Korolev et al.'s "regularized-linear-usually-wins" result as a sanity
> anchor**, not as a finding.

Notice what this sentence does **not** say: it does not say we *introduce* ML regressors
to CDR (Korolev/NIL did), and it does not say nonlinear wins (it usually loses — that is
the anchor).

### What is genuinely still open vs. what is closed

| Pillar | Status | Who closed / left it open |
|---|---|---|
| "Which regressor is best for CDR?" | **CLOSED** | Korolev 2606.02697 (7 regressors, Ridge wins); NIL 2512.12578 (NN no advantage) |
| Regressor choice as a **per-circuit** selection target | **OPEN** | Korolev picks **one regressor per noise model** by validation MSE — a *global* choice, not a per-circuit feature-driven one |
| **Heterogeneous non-VQE** families (GHZ / QFT / random / mirror) with a regressor comparison | **OPEN** | Korolev = VQE/SK only; NIL = variational only (TFI/UCC/HEA) |
| CDR regressor swaps on **real hardware** | **OPEN but contested** | Every regressor paper (Korolev, NIL, Zhao) is sim-only; **but ML-QEM already ran nonlinear regressors on diverse circuits on IBM hardware** — see rebuttal below |
| **2D (train-size × non-Clifford-fraction)** overfit/crossover map for nonlinear-vs-linear | **OPEN** | Zhao 2511.03556 plots the two axes *separately* for linear/quadratic only, one VQE molecule |
| Gaussian-process / kernel CDR regressor | **CLEAR (nonexistent)** | Confirmed again 2026-07-22: no near-Clifford CDR paper uses GP/kernel regression — optional "new regressor" stretch |

### The precise experiment to run

1. **Families (sim, from the running 1620-unit sweep):** GHZ / `ghz_plus`, QFT, random,
   mirror, plus one variational family. This is your heterogeneity pillar and it comes
   almost for free from `aggregated.csv` — no new QPU time.
2. **The overfitting map (sim):** a 2D grid — x-axis = CDR training-set size N (e.g.
   25 → 400), y-axis = `fraction_non_clifford` (0.1 → 0.9). At each cell, plot
   (linear-CDR error − nonlinear-CDR error). Colour = who wins. **This single heatmap
   is the novel figure** — nobody has plotted the nonlinear-vs-linear crossover jointly.
   Expect: nonlinear overfits at small N / low non-Clifford fraction, and only helps at
   large N + high noise (Korolev's anchor, now *mapped* instead of asserted).
3. **Hardware (spend little):** one or two circuits where the *sim* map says nonlinear
   should win vs. where linear should win, run on Heron, show the crossover holds (or
   report honestly if it doesn't). Frame as **motivating preliminary evidence**, n small.
4. **Selector integration:** add `cdr_nonlinear` as a class label the selector can emit;
   report how often the selector correctly routes to it and whether adding it improves
   overall selection accuracy.

### DO-NOT-CLAIM (these get you desk-rejected)

- ❌ "First to apply ML regressors to CDR / near-Clifford data." **FALSE** — Korolev,
  NIL.
- ❌ "We introduce nonlinear regression to CDR." **FALSE** — same.
- ❌ "Nonlinear CDR beats linear CDR." Usually **false**; the honest result is "linear is
  hard to beat except at large N + high noise."
- ❌ "First model-selection over CDR regressors." Korolev **does** do regressor
  model-selection (per noise model, by validation MSE). Say instead: *"first to make the
  regressor choice a **per-circuit, feature-driven** selection target."*
- ❌ Treating argmin-|error| labels as ground truth. Decision Kernels (2607.02888) shows
  **"Clifford-data regression can be decision-flat while improving MSE"** — disclose this.

### Must-cite (2–3)

1. **Korolev, Lakhmanskiy, Rabinovich 2026 — arXiv:2606.02697** — the paper that closed
   the naive angle; reproduce its Ridge-wins anchor. Cite the **body**, not the abstract
   (the abstract hides the 7-regressor benchmark).
   https://arxiv.org/abs/2606.02697
2. **Zhao et al. 2025 — arXiv:2511.03556** — already plots N-scaling (N≈50 convergence)
   and non-Clifford-fraction dependence for linear/quadratic CDR; cite so you don't claim
   the individual axes as new. https://arxiv.org/abs/2511.03556
3. **Liao et al. (ML-QEM) 2024 — arXiv:2309.17368** — the hardware precedent a reviewer
   will wave. **Rebuttal:** ML-QEM learns to *mimic ZNE* (predicts the ZNE-mitigated
   value from ZNE folds); it is **not** a CDR regressor trained on near-Clifford data and
   is **not** a selectable technique inside a selector. https://arxiv.org/abs/2309.17368
   *(Also cite NIL 2512.12578 for the multi-family / NN-no-advantage point.)*

---

# ANGLE 3 — Selector's learned ZNE-refusal region vs. the analytic help-harm boundary

**Verdict: CLEAR** (the specific synthesis is unclaimed) — and this is your **stronger,
more original** angle. Word it around the *boundary comparison*, never around
"discovering that ZNE can hurt" (that is known physics).

### Q: Is there a usable analytic help-harm boundary to test against? — **YES.**

Scavino Alfaro (arXiv:2605.08251, "The finite-shot help-harm boundary of ZNE", 7 May
2026) derives it in closed form. The mean-squared-error difference between mitigated and
raw is

> **ΔMSE(ε, B) = D_p · ε^(2p) − K_q · ε^q / B**

where ε is the noise strength, B the shot budget, the first term is the **squared-bias
improvement** from extrapolating, and the second is the **excess sampling variance** from
the Richardson coefficients + shot-splitting. The **help-harm boundary is the zero
crossing ΔMSE = 0**, which the paper shows collapses to one of three shapes (its abstract
names all three verbatim): a **shrinking power law** ε*(B) ∝ B^(−1/(2p−q)), a **budget
threshold** B* = K_q / D_p, or **no lower boundary**. This is a computable curve in the
**(noise-strength × shot-budget) plane — exactly the plane your selector spans.** The
variance side K_q is computable a priori (Mohammadipour & Li 2502.20673 give the Lagrange
coefficients from the noise-scale nodes). Verified 2026-07-22: purely analytic + Qiskit
Aer + IBM *diagnostic checks* (not a full hardware study), and **no ML / classifier /
data-driven selector anywhere** — it derives the curve, it never tests whether a learned
model recovers it. So it **enables** Angle 3; it does not scoop it.

**The one honest catch (this is a feature, not a bug):** the bias coefficient D_p needs
the *ideal* value μ₀, which your feature-only selector never sees. So the analytic curve
is computable **in simulation** (you know μ₀ and the noise model) and against
**known-answer benchmark circuits** on hardware (mirror / Clifford, where μ₀ is known),
**but not from static features alone**. That is precisely why "the feature-only selector
independently lands on the μ₀-dependent boundary" is a *non-circular* validation — the
selector cannot have cheated, because it never had μ₀.

### The exact defensible claim (copy this wording)

> Using the analytic finite-shot ZNE help-harm boundary (Scavino 2605.08251) as
> **external ground truth**, we test whether a purely **feature-driven, execution-free**
> QEM selector — which never observes the ideal value the boundary depends on —
> **independently learns to refuse ZNE in the theory-predicted regime**. We overlay the
> selector's **learned ZNE-refusal region** on the **analytic boundary** in the
> (noise × shots) plane, on **simulation** and on **real IBM Heron hardware** (via
> known-answer benchmark circuits). Agreement is evidence the selector learned real
> physics, not a dataset artifact.

### The precise experiment

1. **Simulation:** sweep (noise strength × shot budget). Compute the analytic boundary
   directly (you have μ₀ and the noise model). Plot the selector's learned ZNE-vs-raw
   decision boundary on the *same* axes. Report agreement (e.g. IoU / boundary distance).
2. **Hardware (Heron):** you cannot compute D_p from features, so use **known-answer
   circuits** (mirror / near-Clifford, μ₀ known) at 2–3 shot budgets to measure the
   *empirical* MSE-crossing boundary, and overlay the sim-anchored analytic curve. Show
   the selector's refusal region sits on the measured crossing.
3. **Alignment step (do not skip):** Scavino's boundary assumes **fixed Richardson
   coefficients with a specific per-level shot allocation**. Your off-the-shelf Mitiq
   **folding-dZNE at 1024 shots** must be reconfigured to match that variant, or the
   overlay is apples-to-oranges. This is the single biggest technical constraint.

### Risk: "isn't this obvious / already done?" — and how to defend

| Reviewer attack | Defense |
|---|---|
| "ZNE hurting at low shots on clean devices is already known." | Correct — we **use it as ground truth**, we do not claim it. Cite Scavino / Russo / Köster-Mauerer / Mohammadipour-Li. |
| "GSC-QEMit already avoids heavy mitigation at low noise." | That heuristic is **hand-coded as an imitation prior** (2604.24551), never *learned-then-validated*, no analytic-boundary comparison, no hardware, no real families. |
| "Decision Kernels already does theory-guided QEM selection." | 2607.02888 is **theory-driven selection via a decision kernel**, on static held-out data + a tiny hardware micro-cell. It builds **no feature-driven ML classifier** and makes **no learned-vs-analytic-boundary comparison**. Closest adjacency — cite it prominently and differentiate on "data-driven / feature-based / learned boundary vs. analytic line." |
| "Learned-boundary-vs-analytic-threshold is old (QEC decoders; phase-transition ML, arXiv:2205.12966)." | True *methodologically* — so **scope the claim** to the QEM-selector instance. Cite 2205.12966 as precedent so the *method* looks grounded, and claim only "first for a QEM technique selector." |
| "Your ground truth needs the μ₀ you claim not to have." | **State the feature-only independence explicitly** — that gap is exactly what makes the recovery non-trivial. |

### DO-NOT-CLAIM

- ❌ "We discovered ZNE can hurt." Known. You **recover** the boundary; you do not derive it.
- ❌ "First ever to compare a learned boundary to an analytic threshold." Too broad (QEC
  decoders, many-body phase classifiers do this). Scope to the **QEM-selector** instance.
- ❌ Selling a **3D (shots × noise × depth)** surface. The closed form is **2D in (ε, B)**;
  depth enters only through ε and the coefficients, and depth-driven ZNE collapse
  (Köster-Mauerer 2607.09360) is *empirical*, outside the formula. Sell a **per-family
  (noise × shots)** boundary.
- ❌ Claiming the Heron overlay "proves" agreement — with ~9 QPU-min it is **motivating
  preliminary evidence**, one device.

### Must-cite (2–3)

1. **Scavino Alfaro 2026 — arXiv:2605.08251** — the boundary you validate against (the
   ΔMSE = 0 curve). https://arxiv.org/abs/2605.08251
2. **Scavino 2026 (Decision Kernels) — arXiv:2607.02888** — nearest selection-framing
   adjacency; cite and differentiate. https://arxiv.org/abs/2607.02888
3. **Mohammadipour & Li 2025 — arXiv:2502.20673** — supplies the a-priori-computable
   variance side (K_q) and confirms the bias side needs μ₀.
   https://arxiv.org/abs/2502.20673
   *(Optional method-precedent: many-body learned-boundary-matches-theory, arXiv:2205.12966;
   second analytic boundary to broaden scope: Niroula et al. 2302.04278.)*

---

# VERDICT TABLE

| Angle | Status | The ONE sentence the paper can claim |
|---|---|---|
| **Angle 2 — CDR regressor as selectable technique** | **PARTIALLY-OPEN** (naive version closed by Korolev) | "First to make the CDR **regressor choice** a per-circuit, feature-driven **selection target**, and to **map** when nonlinear-CDR beats linear-CDR across heterogeneous non-VQE families and on real hardware — reproducing Korolev's linear-wins result as an anchor." |
| **Angle 3 — learned ZNE-refusal vs. analytic boundary** | **CLEAR** (synthesis unclaimed; not "wide open") | "First to show a purely **data-driven, execution-free** QEM selector **independently recovers the analytic finite-shot ZNE help-harm boundary** — validated on simulation and real IBM Heron hardware." |

---

# RECOMMENDATION

**Make it one paper: the execution-free QEM technique selector. Angle 3 is the
intellectual headline; Angle 2 is a supporting methods/ablation section. They are
complementary, not competing.**

Reasoning tied to your actual assets (the running **1620-unit sim sweep**, **one Heron
run already done**, **~9.3 free QPU-min left**):

- **Angle 3 is the stronger, cleaner novelty (CLEAR).** "A model that never sees the
  answer independently rediscovers a physics boundary" is a memorable, hard-to-scoop
  one-liner and doubles as *external validation that the whole selector learned real
  structure* — which strengthens the umbrella paper too. Its sim half is almost free
  (byproduct of the selector you are already training). Its weakness is that the punchy
  "on real Heron" half is the most QPU-hungry part.
- **Angle 2 is more deliverable but more incremental (PARTIALLY-OPEN).** Its two best
  pillars — heterogeneous families and the 2D overfitting map — are **pure simulation**
  and fall out of the sweep already running, with **zero** new QPU cost. Its hardware
  pillar is the most contested (ML-QEM already ran nonlinear regressors on diverse
  circuits on IBM hardware), so do not spend scarce QPU minutes trying to make it a
  standalone hardware claim.
- **Spend the 9.3 QPU-min on Angle 3's boundary overlay, not Angle 2's hardware.** One
  family of known-answer (mirror/Clifford) circuits at 2–3 shot budgets on Heron directly
  tests the paper's most original claim and buys the most novelty-per-minute. Angle 2's
  hardware pillar is already contested, so leave it as a sim result plus (if any minutes
  survive) a single nonlinear-vs-linear crossover point on hardware.
- **Structure:** (1) the selector + honest LOFO/LODO holdout = the contribution;
  (2) Angle 2 as a self-contained sim section — "add nonlinear-CDR as a selectable
  technique + the crossover map, anchored to Korolev"; (3) Angle 3 as the validation
  section and closing highlight — "the learned ZNE-refusal region lands on the analytic
  boundary, in sim and on Heron."
- **Non-negotiables both sections inherit:** align your folding-dZNE@1024 to Scavino's
  fixed-Richardson variant before the overlay; disclose argmin-|error| labels as a
  limitation (Decision Kernels); scope every hardware result as *motivating preliminary
  evidence*; re-run this novelty scan at submission — the closest threats are all
  April–July 2026.

---

# What re-verification changed (audit trail)

Re-fetched the three closest threats and ran five reject-this searches on 2026-07-22.
Nothing overturned the searchers' verdicts; two nuances sharpened the wording, and two
new items were checked and cleared.

- **Korolev 2606.02697 (re-fetched body).** Confirmed: 7 regressors, Ridge wins, XGBoost
  only occasionally at high noise (p=0.1, depolarizing/Pauli); N fixed at 1500 (no
  learning curves); non-Clifford fraction varied {0.2–0.8} with Ridge insensitive;
  simulation only; only TwoLocal/SK-VQE (+ an RY→RZYRZ transferability variant), **no
  GHZ/QFT/random/mirror**. **NUANCE:** their "model selection" = pick **one regressor per
  noise model** by lowest validation MSE via K-fold CV — a *global* selection, not
  per-circuit. Pillar (a) survives, but the claim must say **"per-circuit, feature-driven
  selection target,"** never just "model selection over regressors."
- **Scavino 2605.08251 (re-fetched).** Confirmed a **usable closed-form boundary exists**
  (ΔMSE = D_p ε^(2p) − K_q ε^q/B; three regimes named in the abstract). Analytic + Aer +
  IBM diagnostic checks only; **no ML/classifier/selector**. Enables Angle 3.
- **Decision Kernels 2607.02888 (re-fetched).** **NUANCE:** it is more than pure theory —
  it runs a decision-aware **selection** experiment on static held-out data **plus a
  hardware micro-cell probe**, and finds decision-aware selection helps "often by
  retaining Raw." Still **no feature-driven ML classifier** and **no learned-vs-analytic
  boundary comparison**, so Angle 3's synthesis is intact — but this is the closest
  adjacency and a DO-NOT-CLAIM source for argmin labels. Cite it prominently.
- **Killer queries (5) found no direct scoop.** New items checked and cleared:
  (i) arXiv:2607.01180 "Non-Clifford Benchmarking via Ensemble Feature Selection" —
  ridge + feature selection on ibm_kingston, but it estimates **gate process infidelity
  (benchmarking)**, not QEM technique selection; false alarm.
  (ii) Q-ANCHOR arXiv:2605.30075 — ZNE-guided *federated learning*, not a help-harm
  predictor.
  (iii) **Gaussian-process/kernel CDR remains nonexistent** — the "new regressor" stretch
  for Angle 2 is still CLEAR.

*Re-run before submission. Field is moving monthly.*

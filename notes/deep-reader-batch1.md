# Deep Reader — Batch 1 (literature overlap read)

Written 2026-07-22. Six papers flagged as potential overlap with QEM-Selector,
read at abstract+intro+methods+results depth where the source rendered. For each:
what they ACTUALLY did (not abstract spin), extractable numbers, and a precise,
skeptical overlap verdict against our five contributions:

- (a) learned per-circuit *technique-family* selector with honest grouped/LOFO eval
- (b) noise-strength dimension via scaled fake backends
- (c) equal-budget `raw_plus` fairness baseline
- (d) sim-to-real transfer test of the selector on 2026 Heron hardware
- (e) CDR regressor-swap experiments

Sourcing note: arXiv PDFs for 2604.24551 (GSC-QEMit) and 2404.12892 (Q-LEAR)
returned as binary; GSC-QEMit was fully recovered from the arXiv HTML
(`/html/2604.24551v1`), ML-QEM from ar5iv, Qermit from ar5iv, Decision Kernels
+ Calibrator from their pages. Q-LEAR's detailed mechanism could NOT be pulled
from a clean text source (HTML conversion failed, aimodels.fyi 403); its entry
below leans on the verbatim abstract + our task brief and flags the unverified bits.

---

## 1. GSC-QEMit (arXiv 2604.24551, IJCNN/WCCI 2026) — THE closest, but different axis

**URL:** https://arxiv.org/abs/2604.24551 (HTML: https://arxiv.org/html/2604.24551v1)

**What they actually did.** A three-stage online controller: (i) a Growing
Hierarchical Self-Organizing Map (GHSOM) clusters streaming device telemetry into
discrete "operating contexts"; (ii) a sparse-variational Gaussian process (SVGP)
forecasts short-horizon logical-fidelity degradation with uncertainty; (iii) a
cost-aware contextual multi-armed bandit (Thompson sampling) picks a mitigation
*action* each cycle. The action set is three **intensity levels**, verbatim:
"NONE: baseline execution with no additional mitigation; MODERATE: a mid-cost
intervention... SEVERE: a high-intensity intervention... at the highest overhead."
Crucially, in the actual simulator instantiation these are NOT distinct QEM
families — MODERATE = "structural redundancy-and-decode... replicating circuit
sampling and applying majority-vote decoding," SEVERE = "action-conditioned
scaling of the effective noise parameters." An abstract "intervention library
{Identity, Pauli Suppression, Surface Code}" is described but instantiated as
noise-scaling factors, not real ZNE/CDR/REM. GHSOM+SVGP are trained OFFLINE then
frozen; only the bandit posterior updates ONLINE from a 13-D telemetry vector
(normalized cycle time, code distance, p_eff, logical error rate, logical
fidelity, entropy, T-count, 2q-gate count, log error rates, recent-window mean
and variance of the logical error rate).

**Key numbers.** +9.0% relative average logical fidelity vs unmitigated (per-
benchmark +8.5% to +9.4% across 8 circuits: Bell chain, GHZ, CCX-heavy, T-sweep,
Grover, QFT, Bernstein-Vazirani, BV-oracle); ~35% lower aggregate intervention
cost vs always-SEVERE; controller selects NONE ~40% of cycles. Simulation only
(Qiskit Aer instrumented noise), NO hardware, qubit counts not stated in results.

**Overlap verdict.** This is the single paper a reviewer will name against us, and
it must be cited and differentiated carefully — but the overlap is shallower than
the title suggests. Shared: the *idea* of a learned policy that chooses a
mitigation action from features. DIFFERENT on essentially every axis of our
contribution: (a) they select **how much** mitigation (intensity NONE/MOD/SEVERE),
we select **which technique family** (ZNE vs CDR vs REM) — orthogonal decisions;
their "families" are synthetic noise-scaling stand-ins, not the real Mitiq
protocols we benchmark. Their input is **streaming runtime telemetry / drift**,
ours is **static pre-execution circuit+backend features** (no execution needed at
predict time). Their eval is online-regret on a synthetic Aer testbed with no
grouped/LOFO/LODO generalization protocol — nothing like our (a) honest held-out
family/device splits. (b) They vary noise over time via drift, not a controlled
scaled-backend axis. (c) No equal-budget accuracy control; their cost accounting
is intervention-count, not our shot-budget `raw_plus`. (d) No sim-to-real transfer
— they never touch hardware, so our Heron test is untouched. (e) No CDR internals.
Net: closest in spirit ("adaptive learned QEM selector"), but it answers a
different question (online intensity dosing under drift) and leaves our entire
offline-technique-family + honest-generalization + sim-to-real contribution open.

---

## 2. Mitiq Calibrator (docs, 2023-2026) — empirical, shot-spending, ZNE/PEC-centric

**URL:** https://mitiq.readthedocs.io/en/stable/guide/calibrators.html

**What it actually does.** A workflow object that RUNS a fixed set of benchmark
circuits on the backend under each candidate Strategy, measures the improvement
factor (noisy vs mitigated error) per circuit-strategy pair, and returns "the
strategy that performed best of those supplied." It is purely empirical — no ML,
no feature model, no prediction. Cost is real backend execution (docs show
`'noisy_executions': 100`). The strategy grid is dominated by ZNE hyperparameters
(scale factors 1.0/2.0/3.0; fold_global vs fold_gates_at_random; Richardson vs
Linear factory) via `ZNE_SETTINGS`, plus `PEC_SETTINGS` quasiprobability
representations. Built-in benchmark circuits are ghz / w / rb / mirror at 2 qubits.

**Key numbers.** None (documentation, not a study) — reports per-strategy
improvement factors it computes at run time.

**Overlap verdict.** This is the tool our whole framing is defined against, and the
delineation SURVIVES contact with the current docs. It selects a strategy by
**spending shots on the actual backend** (a fresh calibration run per device);
we **predict from static features without spending any shots**, which is the entire
point of (a). Its candidate space is essentially ZNE tuning (+ PEC), i.e. mostly
"which ZNE knobs," whereas we select across **technique families** and include REM.
It has no learned/transferable model — the calibration doesn't generalize to a new
circuit or device without re-running — so it cannot make an offline prediction and
has no notion of LOFO/LODO generalization (a), no scaled-noise axis (b), no
equal-budget accuracy control (c, its cost is just execution count), no sim-to-real
transfer study (d), and no CDR-internals work (e). Cite as: "prior art picks a
strategy by running experiments on the device; we predict the technique family
from cheap static features, amortizing that cost across circuits." One caveat to
state honestly: the Calibrator DOES already span >1 family (ZNE+PEC), so we should
say "existing selection is empirical and shot-spending," not "no tool selects
across families" — that stronger claim is false.

---

## 3. ML-QEM (arXiv 2309.17368, Nature Mach. Intell. 6:1478, 2024) — ML *performs* QEM, never selects

**URL:** https://arxiv.org/abs/2309.17368 (ar5iv: https://ar5iv.labs.arxiv.org/html/2309.17368)

**What they actually did.** IBM trains supervised models to PREDICT the
noise-free expectation value from the noisy one — the noisy value ⟨O⟩^noisy is the
input, the target is ⟨O⟩^mit. Models: linear regression, random forest, MLP, GNN.
Features: native-gate counts (parameterized gates binned by angle), the Pauli
observable in sparse-Pauli representation, and optional device noise data (gate
errors, T1/T2, readout errors; the GNN puts these on graph nodes). The models are
trained to **mimic digital ZNE** at lower runtime cost; for >~large circuits they
mimic other scalable QEM outputs. Explicitly a *mitigator*, not a *selector*:
"Performing mimicry does not allow the ML-QEM model to outperform the mimicked QEM
method by its nature."

**Key numbers.** Up to 100 qubits on hardware (Trotter circuits, up to 1,500
CNOTs). Random forest is best ("consistently outperforms the other ML-QEM models,
MLP closely following"). Overhead reduction vs ZNE: "50% lower runtime quantum
resource overhead" / "~25% lower overall and 50% lower at runtime." Strong under
interpolation; extrapolation degrades when coherent noise is present.

**Key numbers we'll be compared on:** their RF + circuit-feature machinery is
near-identical to ours, so this is the "isn't this just ML-QEM?" reviewer risk.

**Overlap verdict.** Heavy *machinery* overlap (RF, circuit + noise features),
ZERO *task* overlap. ML-QEM's model output IS the mitigated number (it replaces
running ZNE); our model's output is a **categorical technique choice**, and the
actual mitigation is still done by real Mitiq ZNE/CDR/REM. They pick no technique
— they hard-code "mimic ZNE." So all five of our contributions are untouched: (a)
we select among families with grouped/LOFO/LODO honesty (they have one fixed
target and report interpolation/extrapolation, a different split semantics); (b)
no controlled scaled-noise selector axis; (c) no equal-budget raw baseline (their
"cost" is shots-to-mimic-ZNE, a different quantity); (d) they run hardware to 100q
but never test whether a *selector* transfers sim→real — that question only exists
if you're selecting; (e) no CDR-regressor ablation. Differentiator sentence:
"ML-QEM learns to *be* a cheaper ZNE; we learn *which* mitigator to use." The
related IBM patent (task brief cites US 12,481,908) covers the same runtime-
mitigation idea, likewise not selection — worth a one-line acknowledgement, not a
threat.

---

## 4. Qermit volumetric benchmarking (arXiv 2204.09725, Quantum 7:1059, 2023) — manual benchmarking, no model

**URL:** https://arxiv.org/abs/2204.09725 (ar5iv: https://ar5iv.labs.arxiv.org/html/2204.09725)

**What they actually did.** Quantinuum's systematic *volumetric* benchmark
(width × depth grid) of QEM protocols on real superconducting hardware +
matched classical simulation, to find "the situations in which their use is
beneficial." Main experiments benchmark **ZNE and CDR** (Qermit also *implements*
PEC, frame randomisation, SPAM correction, but those aren't in the headline
study). No predictive model, no classifier, no selector — the volumetric plots are
read by a human. Ships Qermit, an open-source graph-based QEM composition library.

**Key numbers / findings.** "CDR generally outperforms ZNE on structured Pauli
circuits and slightly underperforms on random SU(4) circuits" even at low shot
budget (~10^5/value); ZNE does well on random circuits under depolarizing sim.
Devices: ibmq_lagos and ibmq_casablanca. Two central qualitative results directly
useful to us: (1) "emulation largely overestimates the performance... for
significantly larger circuit sizes compared to those accessible with real
hardware" — a predicted-vs-practical disconnect; (2) on two devices of comparable
error rates, "performance of error mitigation drastically decreased on ibm_lagos,"
attributed to noise bias / different noise profiles.

**Overlap verdict.** This is the *manual* version of our exact question and is
almost entirely complementary — it establishes that per-circuit/per-device QEM
benefit varies and must be characterized, which is our motivation, but it stops at
human-read plots. It provides NO learned selector (a) — this is the gap we fill;
no controlled noise-strength dial (b) (it uses whatever the two real devices give);
no equal-budget raw control (c); no selector-transfer study (d). Two things to
harvest rather than differentiate: (i) their emulation-overestimates-hardware
finding is independent support for our **sim-to-real distribution-shift
hypothesis** (d) — cite it exactly there; (ii) their ZNE-vs-CDR circuit-dependence
(Pauli vs SU(4)) is external evidence that the technique-selection signal we're
learning is real physics, not an artifact. Note honestly: they benchmark only
ZNE+CDR in the study (no REM head-to-head), and only 2 devices.

---

## 5. Decision Kernels (arXiv 2607.02888, July 2026) — theory that selection is the right frame; no selector

**URL:** https://arxiv.org/abs/2607.02888

**What they actually did.** A finite-shot *theory* paper arguing that QEM is
usually benchmarked by expectation-value accuracy, but many near-term workflows use
those values only for **downstream decisions** (argmin selection, ranking, top-k,
optimizer-step acceptance, phase labeling), which depend only on *gaps*, not
absolute values. They build a quotient-space "residual gap law / decision kernel"
formalism (the kernel = device noise pulled back through the mitigation map) and
prove several results (quotient factorization, a marginal no-go theorem, a QEM
pullback theorem, Gaussian decision-risk formulas, a shot-level converse). No
trained selector — pure framework, with Qiskit Aer finite-shot demonstrations
plus a pre-registered hardware micro-cell probe.

**Key numbers / findings.** Qualitative but pointed: "Clifford-data regression can
be decision-flat while improving mean-squared error"; "probabilistic error
cancellation can improve accuracy while worsening decision risk through sampling
overhead"; "Decision-aware selection modestly reduces static held-out failure
relative to accuracy-based selection, often by retaining Raw, but the dynamic
success target is not reached." Aer simulation + a hardware micro-cell.

**Overlap verdict.** Not a competitor to any of our five contributions — it's the
strongest available *motivation* that per-use-case QEM selection is a recognized
open problem, and simultaneously a **methodological warning** we should absorb.
Motivation use: it explicitly frames "select QEM methods through residual gap
geometry, not from expectation-value accuracy alone," i.e. the selection question
we operationalize is live theory. Warning use, and we must be honest about it: our
labels select the technique with lowest |error| (accuracy), which is exactly the
metric this paper says can be *decision-irrelevant* — their CDR-decision-flat and
"selection often just retains Raw" results are a caution for our accuracy-argmin
labels and for reading too much into small |error| wins. It also independently
notes CDR can look good on MSE yet do nothing downstream — relevant to our (e) CDR
work and to our CDR-dominance caveats. They build no selector, use no static
feature model, run no sim-to-real transfer of a selector, so (a)-(e) all remain
ours; cite as motivation + as a limitation to disclose (accuracy-based label
choice), not as prior art we're beaten by.

---

## 6. Q-LEAR (arXiv 2404.12892, FSE Companion 2024) — learn-to-mitigate, not selection

**URL:** https://arxiv.org/abs/2404.12892

**What they actually did (from the verbatim abstract + task brief; detailed
methods NOT recoverable from a clean text source — see sourcing note).** A
"practical ML-based approach, called Q-LEAR, with a novel feature set, to mitigate
noise errors in quantum software outputs." It is a **learn-to-mitigate** method: a
model trained on circuit+backend features that corrects noisy outputs directly,
motivated by prior ML-QEM work "only targeting specific noise types or specific
quantum circuits." Evaluated on **eight IBM quantum computers and their matched
noisy simulators**, compared against a state-of-the-art ML-based baseline. It does
NOT select among distinct QEM technique families.

**Key numbers.** "25% average improvement in error mitigation on both real quantum
computers and simulators" vs the ML baseline; 8 IBM machines + simulators.
(A binary-PDF fetch additionally suggested a random-forest regressor, a ~50-circuit
suite, a 70/30 split, and per-machine 15-35% spread — I could NOT verify these from
a reliable text source, so treat them as UNCONFIRMED and do not cite them as fact.)

**Overlap verdict.** Same category as ML-QEM (#3): feature-engineering + ML overlap,
but the task is direct output correction, not technique selection — so it's another
"related ML-QEM feature work to acknowledge," not a scoop of our contribution. It
corrects outputs; we choose a mitigator and then run the real thing. All five
contributions remain open: (a) no technique-family selector or grouped/LOFO/LODO
protocol (it reports average improvement, not held-out family/device generalization
of a *selector*); (b) no controlled scaled-noise axis; (c) no equal-budget raw
control; (d) it spans 8 real machines + sims, which is genuinely relevant to our
sim-to-real story and worth citing as evidence that ML corrections do move between
sim and hardware — but it never tests whether a *selector's choice* transfers,
because it selects nothing; (e) no CDR internals. Differentiator: "Q-LEAR learns a
correction to the output; we learn a decision over which correction method to run."
Honest caveat to ourselves: its custom circuit+backend feature set overlaps ours,
so cite it as the nearest feature-engineering precedent and make sure our feature
novelty claims are modest.

---

## Cross-cutting takeaways for the paper

1. **Nobody does our exact thing.** No accessible work trains an OFFLINE,
   static-feature classifier that selects among QEM technique *families* with honest
   grouped/LOFO/LODO generalization. GSC-QEMit selects intensity online from
   telemetry; Mitiq Calibrator selects a strategy by spending shots; ML-QEM and
   Q-LEAR learn to *mitigate*, not select; Qermit benchmarks by hand; Decision
   Kernels theorizes selection without building one. Delineation holds.

2. **Two claims to soften (skeptic's honesty).** (i) Do NOT say "no tool selects
   across technique families" — the Mitiq Calibrator spans ZNE+PEC empirically;
   say "no tool *predicts* the family from static features without spending shots."
   (ii) Do NOT overstate feature novelty — ML-QEM and Q-LEAR use closely related
   circuit+backend/noise feature sets; ours is standard, the novelty is the label
   (technique choice) and the evaluation, not the features.

3. **Free supporting evidence to cite.** Qermit's "emulation overestimates hardware"
   and Lagos-vs-Casablanca divergence directly support our sim-to-real distribution-
   shift hypothesis (contribution d). Decision Kernels supplies the theoretical case
   that per-use-case selection is an open problem (framing) AND a caution that our
   accuracy-argmin labels may be decision-irrelevant (a limitation to disclose).

4. **Reviewer-risk ranking.** GSC-QEMit (closest in spirit, cite + differentiate on
   axis: intensity-online-telemetry vs family-offline-static) > ML-QEM (shared RF/
   feature machinery, differentiate on mitigate-vs-select) > Mitiq Calibrator (the
   defined-against baseline) > Q-LEAR (another learn-to-mitigate) > Qermit (manual,
   complementary) > Decision Kernels (motivation/limitation, not competition).

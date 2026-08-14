# QEM-Selector — Literature Review & Novelty Map

Written 2026-07-22 by the literature-synthesis pass. Audience: you (final-year
AI/ML student, new to quantum). This file merges six parallel literature searches
(novelty threat scan, benchmark comparison, CDR-regressor landscape, ZNE failure
modes, sim-to-real noise fidelity, and 24 deep-read paper verdicts). Every claim
is cited with a URL. Honest tone throughout — where a searcher found that one of
our "novel" angles is already partly done, it says so in bold.

The purpose of this document is to answer one question before you write the paper:
**what can we actually claim, and what will a hostile reviewer throw at us?**

---

## 1. The landscape in plain words

**Quantum error mitigation (QEM) has no universal winner.** The whole field agrees
on this. Which technique reduces error most depends on the circuit, the device, and
crucially the *shot budget*. The canonical demonstration is UNITED (Bultrini et al.,
Quantum 2023): at 10^5 shots ZNE wins, at 10^6-10^8 vnCDR wins, at 10^10 shots their
UNITED method wins — same circuits, different budget, different winner. Cross-platform
benchmarks (Russo et al., IEEE TQE 2023) and volumetric benchmarks (Qermit, Quantum
2023) show the same story across real IBM/IonQ/Rigetti hardware: ZNE beats CDR on
random circuits, CDR beats ZNE on structured Pauli circuits, and on some devices
mitigation makes things *worse* (improvement factor mu < 1). This "which-wins-when"
structure is exactly the signal our selector tries to learn — but every one of these
papers stops at a human-read plot. Nobody builds a predictor.

**There is a large and fast-growing "ML for QEM" literature — but almost all of it
uses ML to *perform* mitigation, not to *choose* it.** IBM's ML-QEM (Liao et al.,
Nature Machine Intelligence 2024) trains a random forest to predict the mitigated
expectation value directly, mimicking digital ZNE at lower runtime cost. Q-LEAR (FSE
2024), the deep-learning distribution-correction models (Placidi et al. 2026), GEM's
graph neural network (2026), DAEM (npj QI 2025), and Sack & Egger's QAOA network
(PRR 2024) are all variations on the same idea: learn a noisy-to-ideal correction
map. This matters for us in two ways. First, the *machinery* overlaps ours heavily —
they use random forests and circuit+backend features just like us — so a reviewer
will ask "isn't this just ML-QEM?" Our answer must be crisp: **ML-QEM learns to *be*
a cheaper mitigator; we learn *which* mitigator to run, and the actual mitigation is
still done by real Mitiq ZNE/CDR/REM.** Second, our feature engineering is *not* our
novelty — these papers already use similar features. Our novelty is the **task**
(selection) and the **evaluation** (honest held-out generalization + a hardware
transfer test).

**The closest things to an actual QEM *selector* are all adjacent, never a hit.**
Three near-threats: (1) **GSC-QEMit** (arXiv:2604.24551, IJCNN 2026) is an online
contextual bandit that picks mitigation *intensity* (NONE/MODERATE/SEVERE) from
streaming device telemetry — it chooses *how much* mitigation over time, not *which
technique family*, and its "families" are synthetic noise-scaling stand-ins, not real
ZNE/CDR/REM. (2) **Mitiq's Calibrator** picks a strategy by *running* benchmark
circuits on the backend and measuring which won — it spends shots, does no ML, and
mostly tunes ZNE hyperparameters (plus PEC). (3) **Beisel et al.** (Electronics 2022)
selects readout-mitigation methods inside a workflow engine using hand-written rules —
selection exists, but it is rule-based and stays inside the REM family. The perfect
methodological precedent is outside QEM entirely: **MQT Predictor** (Quetschlich et
al., IEEE QSW 2023) trains a supervised classifier to predict the best *compilation*
options from circuit features. That is our exact method template, applied to a
different problem — we should own that analogy openly rather than let a reviewer
frame us as derivative of it.

**Our real-hardware finding — ZNE made things worse on a clean Heron device at 1024
shots — is known and expected physics, not a discovery.** A May-2026 theory paper
(Scavino Alfaro, arXiv:2605.08251) literally derives the finite-shot "help-harm
boundary" where ZNE flips from helpful to harmful, and places the harmful regime at
low shots, shallow circuits, and low noise — our exact regime. The mechanism is that
Richardson extrapolation trades noise bias for variance that grows with the
extrapolation coefficients (Mohammadipour & Li, Quantum 2025; Krebsbach et al.,
PRA 2022): when the device is clean there is little bias to remove, so the added
variance dominates and mitigation hurts. Empirical corroboration on modern hardware
exists too — Köster & Mauerer (arXiv:2607.09360, 2026) show ZNE "improvements" can be
pure artifacts overshooting the ideal by up to 21% on IQM hardware, and IBM's own docs
say ZNE is "not guaranteed to produce an unbiased result" and that their default
resilience level enables only readout mitigation, not ZNE. **So we must present our
ZNE result as an independent confirmation, not a finding — and disclose that we used
off-the-shelf folding-based ZNE at 1024 shots, which is the variant IBM itself has
moved away from on Heron (they now recommend PEA).**

**The sim-to-real framing is well grounded, and our specific claim appears
unpublished.** Stock Qiskit calibration-based noise models deviate 5.4%-72% in
fidelity from real backends (EPJ Quantum Technology 2024); a 2026 digital-twin study
finds fake-backend snapshots agree with real hardware only in a device- and
transpilation-dependent way; learned noise models do not transfer zero-shot across
devices — a 2026 paper shows this *on ibm_marrakesh itself* (our exact device),
recovering only ~35% of the gap with 20 fine-tuning samples and naming CX-gate error
as the dominant mismatch. The Eagle→Heron generational jump (EPLG 1.7e-2 → 6.2e-3,
McKay et al. 2023) cleanly explains why our Falcon-era fake backends are so much
noisier than what we measured on marrakesh (raw error 0.016-0.031). But there is an
honest counterpoint to cite: an ICCAD 2025 study finds calibration-driven compilation
decisions survive stale calibration cycles, so "snapshots are stale" does not
automatically mean "snapshot-trained decisions are worthless." Whether staleness
invalidates *technique selection across a hardware generation* is precisely our open
empirical question — nobody has trained a QEM-technique selector on FakeBackend noise
and evaluated it on live hardware.

**Bottom line for framing.** The core contribution — a learned, feature-driven,
execution-free selector across QEM technique *families*, with honest LOFO/LODO
generalization and a sim-to-real hardware transfer test — survives the search intact.
But three of our five sub-claims need careful wording (see §3), and the CDR
regressor-swap angle (our originally-planned "novel angle") has been substantially
closed by a June-2026 paper and must be reframed.

---

## 2. The most relevant papers (24)

Overlap class legend:
- **DIRECT** = benchmarks/compares multiple QEM techniques (our motivation; no predictor)
- **SELECTOR-ADJACENT** = does some kind of selection/adaptation (closest threats)
- **ML-MITIGATOR** = uses ML to *perform* mitigation (machinery overlap, task differs)
- **CDR-LINE** = about the CDR technique / its regressor (relevant to angle e)
- **ZNE-LIMITS** = why/when ZNE fails (grounds our hardware negative)
- **SIM2REAL** = noise-model fidelity / drift / transfer (grounds contribution d)
- **BACKGROUND** = foundations, surveys, references

| # | Citation | What it did (one line) | Overlap |
|---|----------|------------------------|---------|
| 1 | **GSC-QEMit: Telemetry-Driven Hierarchical Forecast-and-Bandit for Adaptive QEM** — Szachara et al., 2026, IJCNN 2026, arXiv:2604.24551 — https://arxiv.org/abs/2604.24551 | Online contextual bandit picks mitigation *intensity* (NONE/MOD/SEVERE) from streaming telemetry; Aer only, no hardware. | SELECTOR-ADJACENT (closest in spirit) |
| 2 | **Mitiq Calibrator** — Unitary Foundation, 2023–, docs — https://mitiq.readthedocs.io/en/stable/guide/calibrators.html | Picks a strategy by *running* benchmark circuits on the backend (spends shots); mostly ZNE knobs + PEC; no ML. | SELECTOR-ADJACENT (the tool we define against) |
| 3 | **Configurable Readout Error Mitigation in Quantum Workflows** — Beisel et al., 2022, Electronics 11:2983 — https://www.mdpi.com/2079-9292/11/19/2983 | Rule-based selection of REM methods inside a BPMN workflow engine; no learning, within-REM only. | SELECTOR-ADJACENT (rule-based) |
| 4 | **Predicting Good Quantum Circuit Compilation Options (MQT Predictor)** — Quetschlich, Burgholzer, Wille, 2023, IEEE QSW, arXiv:2210.08027 — https://arxiv.org/abs/2210.08027 | Supervised classifier (RF best) predicts best compiler/device from circuit features; ~75% top-1, >95% top-3. | SELECTOR-ADJACENT (our method template, different problem) |
| 5 | **Decision Kernels for QEM: Why Accuracy Gains Need Not Improve Downstream Decisions** — Scavino, 2026, arXiv:2607.02888 — https://arxiv.org/abs/2607.02888 | Theory: QEM should be selected by downstream *decision* quality, not MSE; CDR can be "decision-flat". No selector built. | SELECTOR-ADJACENT (motivation + a limitation for our labels) |
| 6 | **Machine Learning for Practical QEM (ML-QEM)** — Liao et al. (IBM), 2024, Nat. Mach. Intell. 6:1478, arXiv:2309.17368 — https://arxiv.org/abs/2309.17368 | RF/linear/MLP/GNN predict the *mitigated value* (mimic ZNE) up to 100 qubits on real IBM HW; RF best. | ML-MITIGATOR (the must-cite cousin) |
| 7 | **Q-LEAR: ML-Based Error Mitigation for Reliable Quantum Software** — 2024, FSE Companion, arXiv:2404.12892 — https://arxiv.org/abs/2404.12892 | ML with a novel feature set *corrects outputs* across 8 IBM machines + sims; ~25% over a prior ML baseline. | ML-MITIGATOR (nearest feature-engineering precedent) |
| 8 | **Deep Learning Approaches to QEM** — Placidi et al., 2026, arXiv:2601.14226 — https://arxiv.org/abs/2601.14226 | Seq2seq attention corrects noisy output *distributions*; real IBM data ≤5q; cross-device transfer works same-architecture. | ML-MITIGATOR (nearest analogue of our transfer question) |
| 9 | **GEM: Scalable QEM with Physically Informed GNNs** — Wang et al., 2026, arXiv:2604.16815 — https://arxiv.org/abs/2604.16815 | GNN beats CDR/ZNE on 16q random circuits (MAE 0.090 vs 0.095/0.120); zero-shot 10q→16q. | ML-MITIGATOR (nonlinear map beats linear CDR) |
| 10 | **Noise-agnostic QEM with Data-Augmented Neural Models (DAEM)** — M. Liao et al., 2025, npj QI 11:8, arXiv:2311.01727 — https://arxiv.org/abs/2311.01727 | Neural mitigation without noise-free labels; on real 4q HW: DAEM 0.067 vs ZNE 0.259 vs CDR 0.095 MAE. | ML-MITIGATOR (data point: ZNE worst on real HW) |
| 11 | **Testing Platform-Independent QEM on Noisy Quantum Computers** — Russo et al., 2023, IEEE TQE, arXiv:2210.07194 — https://arxiv.org/abs/2210.07194 | ZNE/PEC via Mitiq on IBM/IonQ/Rigetti + sims; improvement mu 1-7x, several mu<1 (worse). | DIRECT (closest benchmark; precedents ZNE<1x) |
| 12 | **Volumetric Benchmarking of Error Mitigation with Qermit** — Cirstoiu et al., 2023, Quantum 7:1059, arXiv:2204.09725 — https://arxiv.org/abs/2204.09725 | Width×depth benchmark of ZNE vs CDR on real IBM devices; winner depends on circuit family; emulation overestimates HW. | DIRECT (which-wins-when + sim-to-real support) |
| 13 | **UNITED: Unifying and Benchmarking SOTA QEM** — Bultrini et al., 2023, Quantum 7:1034, arXiv:2107.13470 — https://arxiv.org/abs/2107.13470 | ZNE/CDR/vnCDR/VD under trapped-ion noise; **shot budget decides the winner**; up to 20x at 10^10 shots. | DIRECT (best support for the selector premise) |
| 14 | **ML-Based QEM for Variational Algorithms** — Korolev, Lakhmanskiy, Rabinovich, 2026, arXiv:2606.02697 — https://arxiv.org/abs/2606.02697 | Benchmarks 6 regressors (Ridge/Lasso/RF/SVM/KNN/MLP/XGBoost) on near-Clifford data for VQE; **Ridge wins most regimes**; sim only, ≤12q. | CDR-LINE (**closes naive angle e**) |
| 15 | **Error Mitigation with Clifford Quantum-Circuit Data (original CDR)** — Czarnik et al., 2021, Quantum 5:592, arXiv:2005.10189 — https://arxiv.org/abs/2005.10189 | The CDR method: near-Clifford training circuits, explicitly a *linear* noisy→ideal fit; ~10x on 16q IBMQ. | CDR-LINE / BACKGROUND |
| 16 | **Improving the Efficiency of Learning-Based Error Mitigation** — Czarnik et al., 2025, Quantum 9:1727, arXiv:2204.07109 — https://arxiv.org/abs/2204.07109 | CDR follow-up: smarter *training-data* selection + symmetries; ~10x cheaper; keeps the linear fit. | CDR-LINE (regressor axis left open) |
| 17 | **Unified Approach to Data-Driven QEM (vnCDR)** — Lowe et al., 2021, PRR 3:033098, arXiv:2011.01157 — https://arxiv.org/abs/2011.01157 | CDR + variable noise levels (unifies ZNE+CDR); 8q Ising 33x over raw; still a (multi-)linear fit. | CDR-LINE |
| 18 | **Benchmarking EM: Artefactual Improvements in ZNE** — Köster & Mauerer, 2026, arXiv:2607.09360 — https://arxiv.org/abs/2607.09360 | On IQM HW, Richardson ZNE collapses to a fixed rescaling → fake improvement overshooting ideal by 21%; proposes garbage-folding control + checklist. | ZNE-LIMITS (corroborates our ZNE-negative) |
| 19 | **Claim against Measurement: Statistical Artefacts in QEM Benchmarks** — Köster & Mauerer, 2026, arXiv:2605.29872 — https://arxiv.org/abs/2605.29872 | 132-config sweep flips ZNE from "significant improvement" to "significant degradation"; 72h drift → effect size varies >3x; only 25% of 81 QEM papers use inferential stats. | ZNE-LIMITS (the hygiene bar we must clear) |
| 20 | **The Finite-Shot Help-Harm Boundary of ZNE** — Scavino Alfaro, 2026, arXiv:2605.08251 — https://arxiv.org/abs/2605.08251 | Derives the MSE crossing where fixed Richardson ZNE flips harmful→helpful; harmful at low shots/shallow/low-noise — our exact regime. | ZNE-LIMITS (theory for our hardware result) |
| 21 | **Direct Analysis of ZNE: Polynomial Methods, Error Bounds** — Mohammadipour & Li, 2025, Quantum 9:1909, arXiv:2502.20673 — https://arxiv.org/abs/2502.20673 | Rigorous bias+variance bounds: extrapolation coefficients grow exponentially in nodes, amplifying shot noise. | ZNE-LIMITS (mechanism) |
| 22 | **Few-Shot Cross-Device Transfer for Quantum Noise Modeling** — 2026, arXiv:2604.24397 — https://arxiv.org/abs/2604.24397 | Noise model ibm_fez→**ibm_marrakesh** (our device): substantial zero-shot loss; K=20 samples recover ~35%; CX error dominant. | SIM2REAL (direct support + a follow-up recipe) |
| 23 | **Scalable Mitigation of Measurement Errors (M3)** — Nation et al. (IBM), 2021, PRX Quantum 2:040326, arXiv:2108.12518 — https://arxiv.org/abs/2108.12518 | Production readout mitigation; readout dominates low-depth error; scales to 42q GHZ. | SIM2REAL / REM reference |
| 24 | **Quantum Error Mitigation (review)** — Cai et al., 2023, Rev. Mod. Phys. 95:045005, arXiv:2210.00921 — https://arxiv.org/abs/2210.00921 | The authoritative field survey; taxonomy of ZNE/PEC/REM/learning-based methods + sampling-overhead limits. | BACKGROUND |

Additional supporting refs (cite as needed, not in the core table): McKay et al.,
"Benchmarking Quantum Processor Performance at Scale" (EPLG, arXiv:2311.05933,
Eagle→Heron); "A methodology to select and adjust quantum noise models" (EPJ QT 2024,
5.4-72% fidelity deviation, https://link.springer.com/article/10.1140/epjqt/s40507-024-00284-4);
"Evaluating Calibration-Based Digital Twins" (arXiv:2603.14607); Proctor et al. drift
(Nat. Commun. 2020, arXiv:1907.13608); Kim et al. "Error mitigation with stabilized
noise" (IBM, arXiv:2407.02467); "Revisiting Noise-adaptive Transpilation" (ICCAD 2025,
arXiv:2507.01195, the honest counterpoint); Krebsbach et al. (PRA 2022,
arXiv:2201.08080, ZNE node placement); IBM "Configure error mitigation" docs
(https://quantum.cloud.ibm.com/docs/en/guides/configure-error-mitigation).

---

## 3. NOVELTY STATEMENT — what we can and cannot claim

### What we CAN claim (defensible wording)

> **We present the first offline-trained, execution-free selector across QEM
> technique *families* (raw / ZNE / CDR / REM).** Given only static circuit and
> backend features — computed with zero quantum executions at decision time — a
> supervised classifier predicts which technique family will give the lowest error
> for a given (circuit, device) pair. We evaluate it with honest grouped
> cross-validation and leave-one-family-out / leave-one-device-out generalization
> tests, add an equal-shot-budget control (`raw_plus`), and report the first
> sim-to-real transfer test of such a *selector* — training on fake-backend
> calibration-snapshot noise and evaluating on live IBM Heron hardware.

The four survivable pillars, in priority order:

1. **(a) The learned technique-family selector — this is the headline. PROTECT IT.**
   No prior work *predicts* the technique family from static pre-execution features
   without spending shots. Nearest three are each differentiable: GSC-QEMit (intensity,
   online, telemetry, no LOFO/LODO), Calibrator (shot-spending, empirical, no model),
   Beisel (rule-based, within-REM). The grouped/LOFO/LODO holdout protocol applied to
   *selection* is itself unclaimed.

2. **(d) Sim-to-real transfer of the *selector* — the strongest empirical novelty.**
   Every hardware-transfer paper transfers a *mitigator* (Placidi cross-IBM, GEM
   zero-shot across qubit counts, Q-LEAR 8 machines). None transfers a *selection
   policy*. Frame our Heron data as **motivating preliminary evidence** (n=3 circuits,
   one device, 1024 shots), not a validated transfer study.

3. **(b) Controlled scaled-noise axis feeding a LODO "new noise environment" number.**
   Novel *in combination* as evaluation design, not as a discovery (noise scaling is
   what ZNE already does). Present as rigor, keep the disclosed caveats.

4. **(c) Equal-budget `raw_plus` control.** Sound and not previously highlighted in a
   selector context, but it is a baseline, not a contribution. Present as rigor. It is
   the same *species* of matched-cost negative control as Köster & Mauerer's
   garbage-folding — cite that connection.

### What we must NOT claim (DO-NOT-CLAIM list — these get us desk-rejected)

These are the adversarial novelty judge's contested claims. Read them as hard rules.

1. **DO NOT say "no tool selects across QEM technique families."** FALSE — Mitiq's
   Calibrator spans ZNE+PEC. Correct wording: *"no method **predicts** the technique
   family from static circuit features **without spending shots**."*

2. **DO NOT say "first to use ML to choose a QEM technique" / "first adaptive ML QEM
   selection."** Too strong — GSC-QEMit chooses *intensity* via a bandit; Beisel
   selects within REM via rules. Correct: *"first **learned, feature-driven** selector
   across distinct technique **families** (ZNE/CDR/REM)."*

3. **DO NOT say "first to apply ML regressors to CDR / to Clifford training data."**
   FALSE — Korolev et al. (arXiv:2606.02697) already benchmark Ridge/Lasso/RF/SVM/KNN/
   MLP/XGBoost on near-Clifford data; and Chen et al. (NIL, arXiv:2512.12578) compared
   linear/Lasso/NN inside their framework. Our CDR regressor-swap (angle e) must be
   reframed to: *"a characterization of **when** a nonlinear CDR regressor swap helps
   vs overfits, across heterogeneous circuit families and on real hardware, integrated
   into the selector"* — and cite Korolev prominently, reproducing their Ridge-wins
   result as a sanity anchor if our data agrees.

4. **DO NOT claim "first / novel ML approach to QEM using circuit+backend features."**
   FALSE — ML-QEM, Q-LEAR, and the deep-learning papers all use such features. Keep
   feature-engineering claims **modest**; our novelty is the **task** (selection), not
   the features/machinery. Cite Q-LEAR as the nearest feature precedent.

5. **DO NOT claim "first hardware-transfer study of an ML-QEM model."** FALSE for
   mitigators (Placidi 2601.14226, GEM 2604.16815, Q-LEAR). Correct: *"first
   sim-to-real transfer test of a technique **selector / selection policy**."*

6. **DO NOT treat lowest-|error| labels as unimpeachable.** Decision Kernels
   (2607.02888) argues accuracy gains need not change downstream decisions, especially
   for CDR. Disclose this as an explicit limitation of our argmin-|error| labels.

7. **DO NOT call the noise-scale axis "physics."** It is a controlled synthetic dial
   with disclosed cap compression (~x1.28/x1.44 realized on Lagos) and a noise-character
   change between x1.0 (composite `from_backend`) and scaled (depolarizing+readout).

8. **DO NOT claim the Heron result "proves" transfer failure.** n=3 circuits, one
   device, 1024 shots — motivating, not conclusive.

### Reframed angle (e): the CDR regressor swap

Your originally-planned "novel angle" (swap CDR's linear fit for sklearn regressors)
is **substantially closed** by Korolev et al. (June 2026) in the VQE setting. What
survives, narrowly: (i) per-circuit Mitiq-style CDR across our 5 *heterogeneous*
circuit families (Korolev is VQE/SK-only); (ii) on **real hardware** (Korolev is
sim-only); (iii) a systematic **when-does-it-overfit** regime study as a function of
training-set size and `fraction_non_clifford` (no paper does this scaling analysis —
and no Gaussian-process/kernel-CDR paper exists at all); (iv) feeding the swapped
regressor as an additional technique the *selector* can recommend. Cite Korolev, GEM,
Chen (NIL), and Zhao (ES/NCE-CDR, arXiv:2511.03556) and phrase precisely. Korolev's
own finding — *regularized linear (Ridge) usually wins, nonlinear only helps at high
noise* — is the result you should expect, so frame this as "characterize when," not
"we introduce it."

---

## 4. Numbers from the literature vs ours

Where a comparison is meaningful. Ours = QEM-Selector measured values (small
simulated run + the 2026-07-22 ibm_marrakesh Heron run). Use this table to sanity-
check that our magnitudes are physically believable — they are.

| Source | Setting | They reported | Ours (comparable) | Read |
|--------|---------|---------------|-------------------|------|
| Russo 2023 (2210.07194) | ZNE/PEC on IBM/IonQ/Rigetti HW | mu 1-7x, several mu<1 (worse) | ZNE worse than raw on all 3 Heron circuits | Our ZNE-negative is **precedented** |
| UNITED 2023 (2107.13470) | sim, 10^5 shots | ZNE best but modest | ZNE weak at 1024 shots | **Consistent** (low-shot regime) |
| UNITED 2023 | sim, 10^10 shots | vnCDR/UNITED ~20x | CDR ~10-17x on Manila sim @11x shots | Same order of magnitude |
| CDR original 2021 (2005.10189) | 16q IBMQ HW | ~10x error reduction | CDR sim ~10x (0.209→0.012, non-Clifford, 11x shots) | **Matches** original claim |
| M3 2021 (2108.12518) | GHZ up to 42q | REM largest gain, low-depth | REM 7x on mirror@Heron (0.027→0.004) | **Consistent** (readout dominates low-depth) |
| FF-ZNE 2026 (2603.13949) | Heron 50q | raw 16.8% → ~6% | raw 1.6-3.1% @5q Heron | Heron is cleaner at small n — **consistent** |
| Köster 2026 (2607.09360) | IQM HW | ZNE overshoot up to 21% | ZNE worse on all 3 circuits | Same failure family |
| McKay 2023 (2311.05933) | EPLG Eagle vs Heron | 1.7e-2 vs 6.2e-3 (Heron r3 2Q ~8.1e-4) | Fake (Falcon) noise ≫ marrakesh raw 0.016-0.031 | **Explains** why fake backends are so much noisier |
| Korolev 2026 (2606.02697) | near-Clifford VQE regressors | Ridge 8x; ZNE 44x at low noise; ML wins at high noise | (predicts our angle-e ablation) | Expect **linear hard to beat** |
| EPJ QT 2024 | stock calibration noise models | 5.4-72% fidelity deviation from real | (motivates sim-to-real) | Snapshots ≠ the device |
| Few-shot transfer 2026 (2604.24397) | noise model fez→marrakesh | K=20 recovers ~35% of gap | (follow-up recipe for our selector) | Cheap re-calibration is plausible |

**Our selector's own numbers (small run, 74 rows, simulated noise):** grouped 5-fold
CV accuracy **0.823 ± 0.088** vs grouped-majority baseline 0.521; macro-F1 0.821;
leave-one-family-out 0.808. Read as "clearly above baseline" — the ±0.09 fold noise
means it is *not* a precise number. Do **not** quote the pipeline's 0.905 (that was
training-set-optimistic; already corrected in the upgraded `model.py`).

---

## 5. GAP LIST — the "kami" checklist (what reviewers will ask that we lack)

Ranked by severity. Each gap has the concrete action that closes it. These are the
things that turn a rejection into an accept.

**SEVERITY 1 — will sink the paper if unaddressed**

1. **The whole quantitative story rests on 74 rows of SIMULATED noise; the research
   sweep is not run yet.** CV 0.82 ± 0.09 is a small, noisy dataset.
   → **ACTION:** run `configs\research.yaml` (1620 units → 540 seed-averaged rows,
   ~5h, resumable), train on `aggregated.csv` with `--label both`, and report LOFO
   (new family) + LODO (new noise environment) as the headline generalization numbers.
   This is next-step 1 in PROJECT_STATUS §5.

2. **The sim-to-real claim (our strongest empirical novelty, d) rests on n=3 circuits,
   one device, 1024 shots.** A reviewer will call this an anecdote.
   → **ACTION:** spend the shipped hardware confirmation run (PROJECT_STATUS §5.2),
   expand to more circuits / a second device / more shots if budget allows, and frame
   the current Heron data explicitly as *motivating preliminary evidence*, not a
   validated transfer study. A negative transfer result is publishable — just scope it.

**SEVERITY 2 — a reviewer will demand these**

3. **Our ZNE used off-the-shelf Mitiq defaults (folding, default nodes, 1024 shots).**
   Krebsbach (2201.08080) and Mohammadipour-Li (2502.20673) show optimized node
   placement tames ZNE variance; IBM moved to PEA on Heron. A reviewer citing these
   will say our ZNE-negative is a strawman.
   → **ACTION:** state plainly that our ZNE is *off-the-shelf folding-based dZNE at
   1024 shots* (not the best-possible ZNE); frame the negative result as
   folding-ZNE@low-shots-specific; optionally add an optimized-node or PEA comparison
   as future work. Cite Scavino Alfaro (2605.08251) as the finite-shot help-harm
   boundary our selector approximates empirically.

4. **No inferential statistics / effect sizes / CIs yet, and no drift study.** Köster
   & Mauerer (2605.29872) found only 25% of QEM papers do this and show config/drift
   flips ZNE conclusions — this is now the expected bar.
   → **ACTION:** report confidence intervals and effect sizes (not point estimates),
   robustness across ZNE knobs, and be explicit about winner's-curse (we pick the
   better of two models on the same CV) and fold noise. Adopt their reporting checklist
   (report whether E(λ) still has signal at λ>1; per-state negative-probability weight
   W_neg; flag any overshoot beyond the physical maximum). Consider adding a
   garbage-folding negative control alongside `raw_plus`.

5. **Our argmin-|error| labels may be decision-irrelevant.** Decision Kernels
   (2607.02888) shows CDR can improve MSE while being "decision-flat."
   → **ACTION:** disclose as an explicit limitation; note the planned significance-aware
   "tie" labels (PROJECT_STATUS §6.5) as the mitigation; optionally add a decision-aware
   label variant.

**SEVERITY 3 — disclose honestly, minor code/framing**

6. **CDR structurally refuses on Clifford-heavy families (ghz_plus, near_clifford), so
   it carries no label signal there, and "CDR refused" is not an input feature.**
   → **ACTION:** either add a `cdr_refused` indicator feature as a paper-ablation
   (changes the frozen FEATURE_NAMES interface — scope it), or acknowledge that the
   model learns CDR-unavailability only implicitly via `clifford_fraction`. The research
   grid already broadens CDR's coverage to all 5 families (1206/1620 units pass).

7. **The noise-scale dial is synthetic with cap compression and a character change at
   x1.0→scaled.**
   → **ACTION:** never call it physics; report realized (not nominal) rates — the report
   §5 already does this; keep it.

8. **Feature novelty is modest — ML-QEM and Q-LEAR use similar circuit+backend features.**
   → **ACTION:** keep feature claims modest; put the novelty weight on the task
   (selection) + evaluation + transfer. Cite Q-LEAR as nearest feature precedent.

9. **The field is moving fast — the three closest papers (GSC-QEMit, Decision Kernels,
   the ZNE-artefacts paper) are all April–July 2026.**
   → **ACTION:** re-run the novelty scan immediately before submission. This space is
   heating up in exactly our direction.

---

## 6. Reading list — 5 papers to read fully, in order

Read these five properly (not just abstracts) before writing. Each shapes a specific
part of the paper.

1. **ML-QEM — Liao et al., Nature Machine Intelligence 2024 (arXiv:2309.17368).**
   *Why first:* this is the paper reviewers will most compare us to — same RF, same
   circuit+backend features. You must be able to say, in one sentence, why we are
   different: they learn to *be* a cheaper ZNE (predict the mitigated value); we learn
   *which* mitigator to run. Read their feature engineering and their App. D noise-drift
   fine-tuning (the nearest prior to our sim-to-real concern).
   https://arxiv.org/abs/2309.17368

2. **GSC-QEMit — Szachara et al., IJCNN 2026 (arXiv:2604.24551).**
   *Why:* the single closest "learned QEM selector." Understand exactly why choosing
   *intensity* online from telemetry is a different problem from choosing a *technique
   family* offline from static features — this is your key differentiation paragraph.
   https://arxiv.org/abs/2604.24551

3. **UNITED — Bultrini et al., Quantum 2023 (arXiv:2107.13470).**
   *Why:* this is the empirical foundation of your entire premise — "no universal
   winner; the best technique depends on circuit, device, and shot budget." It also
   explains why your 1024-shot Heron ZNE failure is expected (ZNE only wins in the
   low-shot regime, and even then modestly).
   https://arxiv.org/abs/2107.13470

4. **ML-Based QEM for Variational Algorithms — Korolev et al. 2026 (arXiv:2606.02697).**
   *Why:* this closes the naive version of your CDR regressor-swap angle. You cannot
   write that section honestly without knowing it. Note their headline (Ridge usually
   wins) — it predicts your own ablation's likely outcome and forces the reframe to
   "characterize when nonlinear helps."
   https://arxiv.org/abs/2606.02697

5. **Claim against Measurement — Köster & Mauerer 2026 (arXiv:2605.29872).**
   *Why:* this sets the statistical-hygiene bar your paper must clear (CIs, effect
   sizes, config-robustness, drift). It also independently validates two of your
   findings — that ZNE conclusions flip with configuration/time, and that a
   calibration snapshot is stale by construction (supporting the sim-to-real story).
   Read its companion ZNE-artefacts paper (arXiv:2607.09360) for the negative-control
   idea if you have time.
   https://arxiv.org/abs/2605.29872

*Companion (read the abstract at least):* MQT Predictor (arXiv:2210.08027) — the
algorithm-selection method template you should openly own the analogy to; and Cai et
al.'s QEM review (arXiv:2210.00921) for the technique taxonomy your classes come from.

---

*This scan was run 2026-07-22. Re-run it before submission (see gap 9) — three of the
most relevant papers are less than four months old.*

# Adversarial Novelty Judge — verdict (2026-07-22)

Role: hostile journal reviewer. Re-checked the deep-readers' overlap verdicts by
fetching the closest papers myself, then ran kill-searches with the phrasings a
reviewer would actually try. Bottom line: the core contribution survives, but three
claims must be reworded or they will get us desk-rejected for over-claiming.

## Contribution components under review (from the brief)
- (a) Learned, feature-driven selector that predicts WHICH technique FAMILY
  (ZNE vs CDR vs REM) from static pre-execution features, no shots spent, with
  grouped / LOFO / LODO holdout honesty.
- (b) Controlled scaled-noise axis as a selector-generalization dial.
- (c) Equal-budget `raw_plus` shot-matched control.
- (d) Sim-to-real transfer of the SELECTOR (fake-backend -> ibm_marrakesh Heron).
- (e) CDR regressor swap (nonlinear regressors inside the mitiq CDR pipeline).

## Papers I re-fetched (verdicts CONFIRMED, not just trusted)
- **GSC-QEMit, arXiv:2604.24551** (Szachara et al., 2026) — REAL paper, fetched.
  Abstract confirms it selects "graded mitigation intensity" (lightweight vs heavy),
  NOT technique families; GHSOM+SVGP+Thompson-sampling bandit on *streaming
  telemetry*; Qiskit Aer only. Deep-reader verdict holds exactly. Overlap is shallow
  (learned policy over a mitigation action) and orthogonal on every axis of ours.
- **Korolev et al., arXiv:2606.02697** (2026) — REAL, fetched. ML-QEM for VQE on the
  Sherrington-Kirkpatrick Hamiltonian, up to 12 qubits, **simulation only**, compared
  vs ZNE baseline, **not a selector**. NOTE: the abstract does not enumerate the "six
  regressor families" the deep-reader cited (that detail is in the body) — but it
  unambiguously IS ML regressors trained on near-Clifford data. That is enough to
  kill the naive "first to try ML regressors on Clifford training data" headline for (e).
- **mitiq Calibrator (docs)** — CONFIRMED: RUNS candidate strategies on the backend
  (spends shots, `get_cost`/`cal.run()`), spans **ZNE hyperparameters + PEC** (>1
  family), empirical/exhaustive, no ML, no feature prediction. The honesty caveat is
  real: it DOES span multiple families, so we cannot say "no tool selects across
  families."
- **S-ZNE, arXiv:2511.07092** (new, surfaced in search) — classical learning surrogate
  that does ZNE on the classical side. A within-ZNE *mitigator*, not a selector. No threat.

## Kill-searches run (reviewer phrasings)
"meta-learning QEM technique selection", "automated selection QEM classifier circuit
features", "QEM portfolio predict best technique without running", "which error
mitigation technique recommend ML ZNE CDR readout", "RL/bandit select QEM technique",
"predict optimal QEM strategy per circuit feature-based 2025/2026".

Nothing found that predicts the technique FAMILY from static features. One search
result explicitly stated the literature "doesn't specifically address using
reinforcement learning or bandit algorithms to adaptively select among these
techniques." New neighbors that surfaced are all mitigators, not selectors:
Adaptive-NN-for-QEM (Springer s42484-024-00234-4), Pauli-weight term selection for
ML-QEM (2606.31195), neighbor-informed learning. IBM patents US 12,198,013
("Calibrating a QEM technique") and US 12,481,908 (runtime ML mitigation) are
calibration/runtime, not static-feature selection — one-line acknowledgements.

## NOVELTY VERDICT (per component)
- **(a) SURVIVES — core novelty.** No prior work PREDICTS the QEM technique family
  from static pre-execution features without spending shots. Nearest three are each
  differentiable: GSC-QEMit (intensity, online telemetry, synthetic stand-ins, no
  LOFO/LODO); Calibrator (shot-spending, empirical, ZNE+PEC, no model); Beisel 2022
  (rule-based, within-REM only). The grouped/LOFO/LODO holdout protocol applied to
  *selection* is itself unclaimed elsewhere. This is the headline; protect it.
- **(b) SURVIVES as method-design rigor, not a discovery.** The controlled scaled-noise
  dial feeding a LODO "new noise environment" number is novel *in combination*, but
  noise scaling itself is old (it is what ZNE does). Frame as evaluation design, and
  keep the disclosed caveats (cap compression to ~x1.28/x1.44 on Lagos; x1.0 composite
  vs scaled depolarizing character change).
- **(c) SURVIVES as a control, not a headline.** `raw_plus` equal-budget baseline is
  sound and not previously highlighted in a selector context, but it is a baseline, not
  a contribution. Present as rigor.
- **(d) SURVIVES — strongest EMPIRICAL novelty.** Sim-to-real transfer of a SELECTOR is
  untouched. Every hardware-transfer paper transfers a MITIGATOR: Deep Learning
  Approaches (2601.14226, cross-IBM-QPU transfer), GEM (2604.16815, zero-shot across
  qubit counts), Q-LEAR (8 machines+sims). None transfer a selection policy. Qermit
  (2204.09725) "emulation overestimates hardware" is independent SUPPORT for the
  distribution-shift hypothesis — harvest it. CAVEAT: our Heron data is n=3 circuits,
  one device, 1024 shots — frame as motivating preliminary evidence, not a validated
  transfer study.
- **(e) NEEDS REFRAMING — naive headline is DONE by Korolev.** arXiv:2606.02697 already
  benchmarks ML regressors as the noisy->ideal map trained on near-Clifford circuits.
  The CDR-extension line otherwise kept linear fits (Perez-Guijarro 2411.16653 =
  circuit construction; Czarnik 2204.07109 = training-data selection), so the regressor
  axis is genuinely open *in the mitiq/CDR line* — but NOT globally. Our surviving
  novelty is narrow and must be stated as such: (i) across our 5 heterogeneous circuit
  families (Korolev is VQE/SK only), (ii) on REAL hardware (Korolev is sim-only), (iii)
  a systematic WHEN-does-it-overfit characterization. Reframe (e) as "characterize when
  a nonlinear CDR regressor swap helps vs overfits," not "we introduce it."

## CONTESTED — claims we must NOT make
1. "No tool selects across QEM technique families." FALSE — mitiq Calibrator spans
   ZNE+PEC. Correct: "no method PREDICTS the technique family from static circuit
   features without spending shots."
2. "First to use ML to choose a QEM technique / adaptive ML QEM selection." Too strong —
   GSC-QEMit chooses mitigation INTENSITY via a bandit; Beisel selects within REM via
   rules. Correct: "first LEARNED, feature-driven selector across distinct technique
   FAMILIES (ZNE/CDR/REM)."
3. "First to apply ML regressors to CDR / to Clifford training data." FALSE — Korolev
   2606.02697. Reframe (e) to heterogeneous-family + hardware + overfitting analysis.
4. "First / novel ML approach to QEM using circuit+backend features." FALSE — ML-QEM
   (2309.17368), Q-LEAR (2404.12892), Deep Learning Approaches (2601.14226) all use
   such features. Our novelty is the TASK (selection), not the features/machinery.
   Keep feature-engineering novelty claims modest; cite Q-LEAR as nearest feature
   precedent.
5. "First hardware-transfer study of an ML-QEM model." FALSE for mitigators (2601.14226,
   2604.16815, Q-LEAR). Correct: "first sim-to-real transfer test of a technique
   SELECTOR / selection policy."
6. Do not treat lowest-|error| labels as unimpeachable — Decision Kernels (2607.02888)
   argues accuracy gains need not change downstream decisions, esp. for CDR. Disclose
   as a limitation.
7. Do not call the noise-scale axis "physics" — it is a controlled synthetic dial with
   disclosed cap compression and noise-character change.
8. Do not claim the Heron result proves transfer failure — n=3, one device; motivating,
   not conclusive.

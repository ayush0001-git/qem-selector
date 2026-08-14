# Literature: SIM-TO-REAL / NOISE MODEL FIDELITY (searcher "sim2real", 2026-07-22)

Task: ground the headline question — does a QEM-technique selector trained on
FakeBackend calibration-snapshot noise transfer to real (Heron-era) hardware?
Our hardware evidence (ibm_marrakesh, 2026-07-22): raw errors 0.016-0.031 (far
cleaner than Falcon-era fake snapshots), ZNE actively hurt at 1024 shots, REM
best on mirror circuits. Method: 10 web searches (arXiv-focused), abstracts of
the top ~10 verified via fetch. Classification: direct-overlap / adjacent /
supporting / background.

## Verdict in one paragraph

The sim-to-real framing is WELL GROUNDED and, importantly, nobody appears to
have published *our specific* claim (a technique-SELECTOR trained on
calibration-snapshot noise failing/succeeding on real hardware). The 2024-2026
noise-model-fidelity literature says: (a) stock Qiskit calibration-based noise
models deviate from real backends by anywhere from ~5% to ~72% in fidelity
terms and agreement is device- and transpilation-dependent (EPJ QT 2024;
arXiv:2603.14607); (b) learned noise models do NOT transfer zero-shot across
devices — a 2026 paper demonstrates this ON ibm_marrakesh itself, with CX-gate
error the dominant mismatch source (arXiv:2604.24397); (c) even on one device,
noise drifts (Proctor 2020) and TLS fluctuations destabilize learned noise
models used by QEM (IBM, arXiv:2407.02467); (d) the Eagle->Heron generational
jump is quantified (EPLG 1.7e-2 -> 6.2e-3; Heron r3 2Q errors into the 8e-4
range), which cleanly explains why Falcon-snapshot FakeManila/Lagos/Jakarta
noise is much heavier than what we measured on marrakesh. One counterpoint to
cite honestly: arXiv:2507.01195 finds calibration-data-driven transpilation
choices matter LESS than assumed (stale-calibration circuits stay usable),
so "snapshots are stale" does not automatically mean "snapshot-trained
decisions are worthless" — the question is empirical, which is our contribution.

## Papers

### Direct-overlap

1. **Evaluating Calibration-Based Digital Twins for IBM Quantum Hardware
   Simulation** — arXiv:2603.14607 (2026-03).
   https://arxiv.org/abs/2603.14607
   Compared FOUR digital-twin variants — calibration-CSV-built, backend-derived
   simulator, backend-derived noise model, and **fake-backend snapshots** —
   against real ibm_brisbane and ibm_sherbrooke, randomized 5q circuits at
   depths 10/20/30, four optimization levels, weighted Jaccard similarity.
   Findings: CSV-built twins often closest to hardware; agreement varies
   strongly with device AND transpilation settings; twins "cannot be assumed
   transferable across systems without validation". This is the closest
   published study to our fake-backend-fidelity premise. It does NOT do QEM or
   selection — our angle stays open.

2. **Few-Shot Cross-Device Transfer for Quantum Noise Modeling on Real
   Hardware** — arXiv:2604.24397 (2026-04, submitted to IEEE QCE 2026).
   https://arxiv.org/abs/2604.24397
   Residual NN noise model trained on ibm_fez (Heron r2), transferred to
   **ibm_marrakesh — our exact device**. Zero-shot transfer degrades
   substantially; 20 fine-tuning samples recover ~35% of the zero-shot vs
   in-domain KL gap (1.67 -> 1.19). Ablation: **CX/2q gate error is the
   primary cause of cross-device mismatch**, then readout. Strongest published
   evidence that learned noise models are device-specific — direct support for
   our distribution-shift hypothesis, and a template for a "few-shot selector
   re-calibration" follow-up experiment.

3. **A methodology to select and adjust quantum noise models through
   emulators: benchmarking against real backends** — EPJ Quantum Technology
   (2024). https://link.springer.com/article/10.1140/epjqt/s40507-024-00284-4
   Benchmarked emulator noise models against real IBM backends: **stock Qiskit
   calibration-based models deviated 5.4%-72.0% in fidelity from real
   devices** (Qaptiva emulator 0.7%-14.0%); after model adjustment they reach
   0.686% deviation on ibm_perth. Quotable headline numbers for "calibration
   snapshots are not the device".

4. **Machine learning for practical quantum error mitigation** — Liao, Wang,
   Sitdikov, Salcedo, Seif, Minev (IBM), Nature Machine Intelligence 6,
   1478-1486 (2024); arXiv:2309.17368 (2023).
   https://arxiv.org/abs/2309.17368
   THE ML-QEM paper. RF / linear regression / MLP / GNN predict mitigated
   expectation values; trained partly on simulated (near-Clifford) data,
   deployed on real IBM hardware up to 100 qubits; digital ZNE as the
   reference; "mimicry" mode reproduces expensive mitigation cheaply.
   Overlap caution: it REPLACES mitigation with ML — it does not SELECT among
   techniques from static features, and does not study sim->real selector
   transfer as a question. Must-cite + must-delineate.

5. **Revisiting Noise-adaptive Transpilation in Quantum Computing: How Much
   Impact Does it Have?** — Huo, Wei, Kverne, Akewar, Bhimani, Patel,
   ICCAD 2025; arXiv:2507.01195. https://arxiv.org/abs/2507.01195
   Five 127q IBM devices, 16 algorithms. Findings: circuits compiled with OLD
   calibration data can be reused across calibration cycles without
   significant fidelity loss; noise-aware mapping concentrates load on "good"
   qubits and increases outcome variability. The honest counterpoint: some
   calibration-driven decisions are less brittle than assumed. Frame our
   result against it — technique CHOICE under a generation shift (Falcon
   snapshot -> Heron reality) is a bigger distribution shift than
   cycle-to-cycle staleness on one device.

### Adjacent

6. **Volumetric Benchmarking of Quantum Computing Noise Models** — Weber,
   Borras, Jansen, Kruecker, Riebisch, arXiv:2306.08427 (2023).
   https://arxiv.org/abs/2306.08427
   Systematic framework for scoring noise models against hardware
   (volumetric-benchmark style, model fitted on training circuits, compared
   with literature models). Gives us the vocabulary "noise model benchmarking"
   for related-work; no QEM, no ML selection.

7. **Simulation and Benchmarking of Real Quantum Hardware** — Piskor,
   Schoendorf, Bauer, Smith, Ayral, Pogorzalek, Auer, Papic, arXiv:2508.04483
   (2025). https://arxiv.org/abs/2508.04483
   Builds a richer-than-calibration noise model for a 20q superconducting
   device; claims improved prediction accuracy over existing literature
   models. Evidence that plain calibration-parameter models underfit real
   noise (crosstalk, leakage etc. missing).

8. **Modeling Noisy Quantum Circuits Using Experimental Characterization** —
   Dahlhauser & Humble, Phys. Rev. A 103, 042603 (2021); arXiv:2001.08653.
   https://arxiv.org/abs/2001.08653
   Composite noise models built from bootstrapped sub-circuit experiments
   (GHZ, Bernstein-Vazirani on IBM transmons), accuracy via total variation
   distance; explicitly motivated by "fluctuations in the underlying noise
   sources and other nonreproducible behaviors". Early, well-cited anchor for
   "calibration parameters alone underdetermine device behavior".

9. **QuantumNAT: Quantum Noise-Aware Training with Noise Injection,
   Quantization and Normalization** — Wang et al., DAC 2022;
   arXiv:2110.11331. https://arxiv.org/abs/2110.11331
   The QML mirror of our problem: QNNs trained noise-free lose up to ~60%
   accuracy on real IBMQ devices; injecting realistic noise models during
   training closes much of the gap; robustness is device-specific. Good
   related-work paragraph: "sim-to-real gap is documented in QML; we document
   it for QEM technique selection."

10. **Error mitigation with stabilized noise in superconducting quantum
    processors** — Kim, Govia, Dane et al. (IBM), arXiv:2407.02467 (2024).
    https://arxiv.org/abs/2407.02467
    Qubit-TLS interaction fluctuations destabilize the learned noise models
    that model-based QEM (PEC/PEA-style) depends on -> incorrect observable
    estimation; tuning qubit-TLS interactions stabilizes noise and makes
    mitigation reliable. IBM itself saying: noise-model staleness breaks QEM.
    Strong physical motivation for our headline.

### Supporting

11. **Detecting and tracking drift in quantum information processors** —
    Proctor et al., Nature Communications 11, 5396 (2020); arXiv:1907.13608.
    https://arxiv.org/abs/1907.13608
    Canonical drift paper: spectral time-series analysis showing
    static-error-model assumptions fail; drift detection/localization
    demonstrated on real qubits. Cite for "any snapshot is stale by
    construction".

12. **Benchmarking Quantum Processor Performance at Scale** — McKay, Hincks,
    Pritchett, Carroll, Govia, Merkel (IBM), arXiv:2311.05933 (2023).
    https://arxiv.org/abs/2311.05933
    Defines layer fidelity + EPLG. Measured: 127q Eagle ibm_sherbrooke EPLG
    1.7e-2 vs 133q Heron ibm_montecarlo 6.2e-3 (~2.7x better); 80q-layer
    fidelity 0.26 vs 0.61. THE citation for the Eagle->Heron generational
    improvement behind "marrakesh is much cleaner than our fake snapshots".

13. **Learning the noise fingerprint of quantum devices** — Martina et al.,
    arXiv:2109.11405 (2021; later Quantum Machine Intelligence / software in
    SoftwareX 2022, arXiv:2202.04581). https://arxiv.org/abs/2109.11405
    SVMs classify which IBM device produced a measurement time-series with
    >99% accuracy, and distinguish time periods on the SAME device. Evidence
    that noise is a device-specific, time-varying fingerprint — i.e., exactly
    the thing a frozen snapshot cannot capture.

14. **DGR: Tackling Drifted and Correlated Noise in Quantum Error Correction
    via Decoding Graph Re-weighting** — arXiv:2311.16214 (2023).
    https://arxiv.org/abs/2311.16214
    QEC-side response to drift: decoders using stale error rates degrade;
    re-weighting from recent syndrome statistics recovers accuracy. (Search
    summaries attribute large short-timescale gate-error drift figures to
    this line of work — verify exact numbers in the PDF before quoting.)
    Shows "adapt-to-current-noise" is an active theme in the neighboring QEC
    field.

### Background

15. **IBM Quantum Heron r2 launch (156q, 5,000 2q-gate circuits)** — IBM
    Newsroom, 2024-09-26.
    https://newsroom.ibm.com/2024-09-26-ibm-expands-quantum-data-center-in-poughkeepsie,-new-york-to-advance-algorithm-discovery-globally
    Official generational specs for the family ibm_marrakesh belongs to
    (Heron r2). Secondary spec roundups: Heron r3 ibm_pittsburgh best 2Q
    error 8.14e-4 ("Early IBM quantum computers: architectural analysis and
    performance benchmarks", J. Supercomputing, 2026,
    https://link.springer.com/article/10.1007/s11227-026-08386-9).

## Gaps we can claim (for the paper's related-work section)

- No published work trains a QEM-TECHNIQUE SELECTOR on calibration-snapshot
  (FakeBackend) noise and evaluates it on live hardware. Closest neighbors:
  digital-twin fidelity studies (1), noise-model transfer (2), ML-QEM value
  prediction (4). Our LODO -> real-hardware evaluation chain appears novel.
- The "ZNE hurts at low shots on clean Heron hardware" observation is
  consistent with the drift/noise-character literature but not published as a
  selector-relevant failure mode; worth reporting as a concrete negative.
- Few-shot selector re-calibration (paper 2's K=20 recipe applied to our
  selector) is an obvious, unclaimed follow-up.

## Search log (for reproducibility)

Queries run (WebSearch, 2026-07-22): fake backend noise model accuracy real
IBM; arxiv noise model fidelity real device mismatch; ML QEM trained
simulation deployed hardware; noise drift IBM calibration fluctuation; IBM
Heron vs Eagle EPLG; Proctor drift tracking; sim-to-real RL quantum control;
temporal stability QEM ZNE; Dahlhauser Humble composite noise model;
site:arxiv.org "fake backend" discrepancy; Martina noise fingerprint;
noise-aware training QML distribution shift; ibm_marrakesh Heron r2 EPLG.
Abstracts fetched and verified: 2603.14607, 2604.24397, 2306.08427,
2309.17368, 2311.05933, 2407.02467, 2508.04483, 2507.01195, 2404.03501
(rejected — fake-device-only QAOA study, no real-hardware comparison).

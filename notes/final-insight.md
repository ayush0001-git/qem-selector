# final-insight — insight-analyst pass notes (2026-07-22)

Owned outputs, all written this pass:
- `docs\ANALYSIS.md` — full tables + findings for the 6 research questions.
- `results\research\figs\winner_share_vs_scale.png`
- `results\research\figs\error_vs_scale.png`
- `results\research\figs\zne_win_region.png`
- this file.

Integrity: best_technique == argmin(abs_error) on 1620/1620 rows; errors.log
(415 cdr + 156 rem) exactly matches the NaN pattern in results.csv.

## Headline findings (each verified, numbers in ANALYSIS.md)

1. CDR share grows with noise: 59.3 -> 61.7 -> 65.7% of rows at x1.0/1.5/2.0
   (79.6 -> 83.0 -> 88.3% among rows it accepts); REM pays for it
   (33.7 -> 25.4%). CDR's own error grows only 0.084 -> 0.096 while raw grows
   0.366 -> 0.475 — that robustness is the mechanism.
2. raw/raw_plus do NOT vanish with noise — they grow 1.9 -> 4.3%, but 49/49 of
   those wins are on Lagos and mostly menu artifacts (REM/CDR refused on most
   of them). On Manila/Jakarta, do-nothing wins 0/1080 rows.
3. ZNE's 78 wins: only 30 beat a full menu; those concentrate at moderate noise
   x depth>=8 (Jakarta x1.0 11 wins; 28/30 at d8/d16). The Lagos@x2.0 "surge"
   (20 wins) is 100% REM-refusal artifact. Direction matches the help-harm
   theory (worse-than-raw 28.5% at d4 vs ~15% at d8/16; only +0.006 mean
   improvement on readout-dominated Lagos, worse on 41.1% rows there). Fixed
   4096 shots => this is a boundary preview, not the (noise x shots) overlay.
4. raw_plus (11x shots) is a coin flip vs raw (better on 49.7% of rows; paired
   mean -0.0003 +/- 0.0138) => raw's error is bias, not variance; the
   equal-budget control cannot rescue raw, which legitimizes mitigation's wins.
5. Error magnitudes (pooled mean): raw 0.4205, raw_plus 0.4202, zne 0.3728,
   rem 0.1963, cdr 0.0893. Reduction factors on own valid rows: cdr 4.3x,
   rem 2.0x, zne 1.1x.
6. CDR's 62.2% overall share is DEFLATED, not inflated, by refusals: among the
   1205 rows it accepts it wins 83.7%; its 415 refused rows are won mostly by
   rem (352). Refusals are structural (ghz_plus 225 + near_clifford 190;
   backend-independent, 46/env). REM refusals (156) are all Lagos,
   8/46/102 by scale.
7. Seed-flip rate (label noise): 20.9% best_technique / 21.6% cost-aware;
   only 58.0% of groups unanimous. Worst family ghz_plus 32.4%, best mirror
   5.2%; dominant confusion cdr<->rem. Median error gap on flips 0.063 (not
   ties). 0/540 aggregate winners rest on partial seed coverage. This is the
   quantitative case for the trainer's merge-back seed-averaged labels.
8. Hardware bridge (n=3, Heron): sim raw error 8.7-15.3x worse than hardware
   on identical circuits; winner agreement 2/3 (both mirrors -> rem); the miss
   (layered_random: hw raw, sim rem) is exactly the low-noise regime the sim
   grid never reaches; ZNE lost 3/3 on hardware.

## Warnings for the paper-writer

- Never quote CDR trends on Lagos without the 0.45 readout-cap caveat
  (realized x1.28/x1.44).
- ZNE feature signature (high clifford_fraction, high readout) is partly a
  refusal-menu signature — say so, it is actually the reason the selector can
  learn it from static features.
- The 78-zne-wins characterization uses per-seed rows; aggregated zne wins are
  25/540 and tell the same story.
- Cost-aware raw share ~50% on Lagos@x2.0: mitigation stops paying for its
  shots on ultra-noisy readout devices — a nice honest sentence.

Scripts used (session scratchpad, logic summarized in ANALYSIS.md appendix):
analysis.py + followup.py, run with the project venv python.

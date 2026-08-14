# review-v2-science — Senior science review of the V2 upgrades (2026-07-23)

Scope: does `zne_fr` faithfully match Scavino's analyzed variant; is the
(noise scale, backend error) -> eps mapping defensible; does the x0.25-x2.0
dial + shots axis trace the boundary; do `cdr_ridge`/`cdr_rf` keep CDR's
guards; would Angle 3 as now buildable convince a hostile referee.
Method: read `mitigation.py` / `boundary.py` / `backends.py` /
`experiment.py` / `features.py`, `configs/boundary*.yaml`,
`results/boundary_smoke/*`, `notes/spike-boundary.md`, `notes/B1..B8`,
tests; then ran verification code with the project venv (scripts in the
session scratchpad; key numbers reproduced below are from those runs).

## What checks out (verified by execution, not just reading)

- `richardson_coefficients((1,3))` = (1.5, -0.5); `variance_k_q()` = 4.0;
  `_variance_penalty` gives K = 4*nu (q=0), 5*nu (q=1) — exactly the spike's
  extraction of Scavino's (1,3)/uniform-split rule.
- `zne_fr` protocol mechanics are faithful: fixed a-priori coefficients
  (never refit), equal-split budget (2 executor builds at base_shots//2,
  verified by monkeypatch — passed executor never invoked), multiplier 1 is
  truthful, deterministic `fold_global`, raw comparator spends the full B.
  Cost model `SHOT_MULTIPLIER_V2` matches actual executions.
- `cdr_ridge`/`cdr_rf` guard parity is real: on live circuits, ghz_plus n2d4
  refuses CLIFFORD and near_clifford n2d4 refuses DEGENERATE identically for
  all three cdr variants; guards run before any noisy call; refusal
  conditions/texts match `_apply_cdr`. (Caveat: parity holds only while
  `CDR_SKLEARN_NUM_TRAINING_CIRCUITS == CDR_NUM_TRAINING_CIRCUITS`; the
  Angle-2 N-sweep will intentionally break it between cells.)
- Shots axis machinery works end to end (boundary_smoke: base_shots column,
  per-budget labels, feature_version 2 with log2_shots, per-budget
  aggregation), and the raw-error dial is monotone down to x0.25.

## FINDING 1 (HIGH) — the analytic boundary is computed for an amplification
## channel `zne_fr` does not implement; on this grid the theory curve is
## quantitatively wrong, sign-flipping on readout-heavy points

Scavino's ΔMSE assumes noise amplification eps -> lambda*eps of ALL the
noise entering the bias slope alpha (paper amplifies by scaling the noise
parameter). Our `zne_fr` amplifies by `fold_global` — which amplifies gate
noise only, never readout. But `estimate_params` derives alpha by a secant
from (0, mu0) to the noisy probe, so D_p = alpha^2 includes the readout
bias that folding cannot touch. Measured (mirror n2 d4, B=4096, 30 seeds,
200k-shot asymptotics):

| backend | raw bias | zne_fr asympt. bias | theory ΔMSE | empirical ΔMSE |
|---|---|---|---|---|
| FakeLagosV2 | -0.516 | -0.514 (0.3% removed) | **+0.258 (HELP)** | **-0.003 (harm)** |
| FakeManilaV2@x2.0 | -0.248 | -0.219 | +0.062 | +0.0136 (4.6x less) |
| layered n2d4, Lagos | +0.379 | +0.371 | +0.139 | +0.0013 (107x less) |

The Manila decomposition is exact: gate (foldable) bias -0.029 (12%),
readout bias -0.219 (88%); zne_fr's residual bias equals the readout part,
and (g+r)^2 - r^2 = +0.0136 reproduces the empirical ΔMSE to 4 decimals.
So the implementation is internally consistent — it is the THEORY SIDE
(D_p from the total-bias secant) that describes an estimator we do not run.
An overlay drawn from these params is not "external ground truth"; a referee
who probes any Lagos point falsifies the curve immediately.

Fixes (pick one, disclose either way):
(a) make sim-side `zne_fr` amplify via the @x dial (level j executes on
    `<base>@x{lambda_j * scale}`) — this IS the paper's amplification; then
    the secant alpha is the right slope. Caps must be checked per node
    (Manila clean through x2.0*3=x6.0; Lagos q2 readout is cap-saturated at
    every scale — drop Lagos from the overlay or disclose node compression);
(b) keep folding but put the non-foldable bias in the theory: probe the
    folded circuit too (alpha_fold from (E3-E1)/(2 eps)) and use
    ΔMSE = (g+r)^2 - r^2 - K_q eps^q / B. Note this is no longer Scavino's
    2-term closed form — the overlay would compare against a corrected curve;
(c) run the overlay on readout-free (depolarizing-only) synthetic noise,
    where folding and eps-scaling coincide — cleanest "theory validation
    cell", then treat real-model grids as the empirical extension.
Also add the missing validation test: nothing in tests/ compares
`delta_mse` to zne_fr's measured MSE difference (all boundary tests are
formula-internal or mocked) — a slow test on a depolarizing-only model
would have caught this.

## FINDING 2 (HIGH) — the selector side of the overlay is degenerate:
## full-menu argmax labels produce an (almost) empty ZNE region

Theory's "help" means zne_fr beats RAW (pairwise, equal budget). The
overlay's "selector chooses ZNE" means zne_fr/zne is argmax over the WHOLE
menu — it must also beat cdr (which wins 84% of its accepted rows) and rem.
In `results/boundary_smoke`: zne_fr wins the menu on 0/24 rows, the trained
bundle's classes are ['cdr','cdr_ridge','raw','raw_plus','rem'] — the model
CANNOT emit a ZNE label — so zne_vote_share = 0.0 at all 15 grid points,
IoU = 0.0, and the reported 86.7% "agreement" is just the theory's harm
share. Yet the pairwise signal exists in the same CSV: zne_fr beats raw on
11/24 rows, with the right pattern (loses at the clean end at 256 shots,
wins at higher noise). The full boundary.yaml grid will reproduce this
degeneracy (in the 1620-row research run, legacy zne won 4.8% of menu
labels; zne_fr additionally competes against cdr on mirror circuits, where
cdr is accepted 36/36 and wins by 10-100x).

Fix is configuration-level, not code: train the Angle-3 selector on a
{raw, zne_fr}-only technique list (then best_technique IS the pairwise
label the theory describes), or derive a pairwise zne_fr-vs-raw label
column from the full-menu run. Keep the full-menu selector as a separate
"in-context refusal" figure if wanted — but the boundary overlay must be
pairwise, or a referee will call the agreement number vacuous.
Related (LOW): `DEFAULT_ZNE_LABELS` counts legacy 'zne' (3x budget, random
folding, 3-node refit) as "chooses ZNE" — a technique on a different
boundary; report section 9 discloses this, but the default invites an
apples-to-oranges overlay for V1 bundles.

## FINDING 3 (HIGH) — `cdr_ridge` at fixed alpha=1.0 is over-regularized to
## the point of inverting the Angle-2 anchor

`Ridge(alpha=1.0)` on ONE unstandardized feature and ~10 training points
shrinks the slope by Sxx/(Sxx+1); measured Sxx = 0.64 on layered n2d4 →
slope 1.088 -> 0.424 (x0.39). Result on that circuit: cdr_ridge error
0.288 vs cdr 0.0069 — and vs RAW 0.079, i.e. "Ridge-CDR" is 3.6x WORSE
than doing nothing. Smoke means: cdr_ridge 0.1418 vs cdr 0.0211. The B1
note flagged this (capture circuit 0.776 vs 0.166) and the cdr-nl spike's
recommended regressor was `RidgeCV(alphas=logspace(-6,3,19))`, which picks
near-zero alpha and reproduces linear CDR (= the Korolev anchor). As
shipped, the Angle-2 experiment would "find" that regularized-linear loses
badly — the opposite of the Korolev result the paper claims to anchor on,
and an artifact of penalty scale, not physics. Fix: RidgeCV (deterministic
LOO, seedable) or standardize the feature and keep a small fixed alpha; the
constant is `CDR_RIDGE_ALPHA` in mitigation.py. Until fixed, cdr_ridge rows
also leak into boundary.yaml runs (it is in that config's technique list),
adding a fake "cdr_ridge" class to the selector (it won a smoke row).

## FINDING 4 (MEDIUM) — eps = avg_2q_error is not a sufficient statistic on
## this grid: identical eps, opposite empirical regimes

The two devices carry very different non-foldable fractions: readout/2q
ratio ~3.7 (Manila) vs ~13.9 (Lagos). Concretely, Manila@x1.5
(eps=0.01493) and Lagos@x1.0 (eps=0.01463) are the same point on the
overlay's x-axis, but empirically zne_fr helps on one (+0.0136) and not the
other (-0.003). A single boundary curve in the (eps, B) plane cannot
represent the fold-based zne_fr across both devices — this compounds
Finding 1 and argues for per-device overlays (or fix 1a/1c). Additional
eps-axis caveats, all measurable: (i) Lagos max readout is cap-saturated
(0.4638 plain -> 0.45 at x1.5/x2.0; non-monotone), so its dial changes
noise composition, not just magnitude; (ii) the x1.0 point runs the
composite from_backend channels while every other dial point is synthetic
depolarizing+readout — a noise-character discontinuity in the middle of the
axis (PROJECT_STATUS 4.10; there is currently no way to force the synthetic
model at scale 1.0, which would make the axis homogeneous); (iii)
`_estimate_params_for_backend` averages D_p and K_q across circuits and
takes the MODAL q — with the shipped 3-family suite it mixes q=1 (mirror)
and q=0 (variational) circuits whose K_q multiply different powers of eps;
averaging coefficients across observable classes is dimensionally
incoherent (the paper fits per family — the overlay default should
partition the suite by q, or grid_spec docs should require one family);
(iv) `estimate_params` uses a single 8192-shot probe at fixed seed 0:
measured D_p spread across probe seeds at Manila@x0.25 is 143-243 (+/-26%)
— an error bar the overlay reports nowhere, largest exactly in the
Heron-like regime the paper targets.

## FINDING 5 (MEDIUM) — shipped documentation states the opposite of what
## zne_fr does on sim

`configs/boundary.yaml` ("zne_fr's @x noise amplification matches the
paper's parameter-scaling on sim but not hardware gate-folding") and
report section 9 caveat 3 ("the noise amplification the theory assumes ...
matches our @x dial more closely than gate folding does — on real hardware
only folding exists") both imply the sim-side zne_fr amplifies via the @x
dial. It does not: `ZNE_FR_FOLD_METHOD='global'` folds gates on sim AND
hardware (mitigation.py `_apply_zne_fr`). Whoever wrote the config/report
text assumed spike option (a); B1 implemented option (b). If Finding 1 is
resolved by switching to dial amplification the text becomes true; until
then these sentences would propagate a false methods claim into the paper.

## FINDING 6 (MEDIUM) — dial + shots coverage traces the boundary only
## partially, and mostly where the theory is least trustworthy

Realized eps span of boundary.yaml's 10 backends: 0.00249 -> 0.02927 (12x);
budgets [256, 1024, 4096] move the q=0 crossing eps* ~ B^(-1/2) by 4x — so
in principle the crossing sweeps through the dial. Verified at B=4096: the
theory crossing sits between Manila x1.0 (ΔMSE -0.0016) and x1.5 (+0.0087).
But at B=256 the entire Manila dial is theory-harm (smoke: no help point);
the help side at low budgets is reachable only via Lagos eps values — and
Lagos is exactly where Finding 1/4 invalidate the curve (readout-dominated,
cap-saturated). Also the plan's 16384-shot budget (INTERFACES V2, spike §6)
was dropped from boundary.yaml, which cuts the high-B side where the
harm region shrinks below the dial floor. Consequences: as configured, the
overlay's trustworthy region (Manila-like, mid/high B) contains the
crossing at only ~one budget. Cheap wins: re-add 16384 (measured cost of
shots is nearly flat — config's own bench notes 0.966 budget factor), and/or
add a third clean device (Jakarta) rather than leaning on Lagos.

## Also noted (LOW)

- `raw` (executor at B, seed s) and zne_fr's level-1 executor (B/2, same
  seed, same circuit) share the seed — their shot noise is not independent,
  which slightly correlates paired raw-vs-zne_fr errors in results.csv.
  Both estimators stay unbiased; paired tests get a (conservative or not)
  covariance term. Worth one sentence in methods.
- Angle 2 is not yet "buildable" end to end: INTERFACES' planned
  `cdr_regressor.yaml` / `cdr_regressor_smoke.yaml` do not exist in
  configs/, `cdr_rf` has never run in an experiment sweep (verified live
  here + unit tests only), and the N x fraction_non_clifford map needs a
  driver that varies `CDR_SKLEARN_NUM_TRAINING_CIRCUITS` — not written.
- Smoke bundle trains fine, but `metrics_significant.json` /
  abstain machinery were not audited here (out of scope).

## Verdict on the review questions

1. **zne_fr vs Scavino:** protocol-faithful (nodes, fixed coefficients,
   uniform split, equal total budget, deterministic amplification) — EXCEPT
   the amplification channel itself (folding vs eps-scaling), which on
   realistic noise models with readout is not a nuance but the dominant
   term (Finding 1). Overlay as-is: apples-to-oranges.
2. **eps mapping:** locally self-consistent (secant alpha makes the bias
   term exact at each grid point) but not defensible as a single-axis
   boundary across devices (Finding 4), and mis-specified for the folding
   estimator (Finding 1).
3. **Dial + shots sufficiency:** span yes, placement partial — the
   crossing sits in the trustworthy part of the grid at only ~one budget;
   re-add 16384 and prefer clean devices (Finding 6).
4. **cdr guards:** preserved, verified live (Clifford + degenerate refusals
   identical across cdr/cdr_ridge/cdr_rf). Ridge itself is mis-tuned
   (Finding 3).
5. **Hostile referee:** would not survive today. The three blockers are
   Findings 1-3; all have concrete, mostly config-level fixes (dial-based
   or readout-free amplification cell; pairwise zne_fr-vs-raw selector;
   RidgeCV). With those, the Angle-3 design — including the genuinely nice
   non-circularity argument and the existing pairwise signal (11/24 smoke
   rows) — is salvageable without new architecture.

# B8 — report.py + recommend.py (V2)

Implemented VERBATIM against INTERFACES §V2. Files: `src/qemsel/report.py`,
`src/qemsel/recommend.py`, `scripts/recommend.py`, `scripts/make_report.py`,
`tests/test_report_v2.py`.

**report.py**: removed the stub guard; `generate_report` gains additive
§8 "Statistical hygiene" (win-share bootstrap CIs per label column,
paired permutation tests, Cliff's delta, Koester pass/flag table — gating
FLAGs + overall verdict in **bold**) and §9 boundary overlay (figure
validated to sit inside out_dir else ValueError; agreement/IoU/shares +
mandatory caveats: realized-eps, zne_fr alignment, sim-only). Consumes
stats/boundary OUTPUT dicts only (no import of those modules). 7 techniques
render via existing dynamic `_detect_techniques`.

**recommend.py**: V1 bundle (no `feature_version`) => exact 3-key return,
base_shots ignored, 2-arg extract_features call preserved. V2 bundle =>
version+base_shots features, abstain when max proba < threshold
(`technique='abstain'`), adds abstained/abstain_threshold/feature_version.
CLI `--shots`; abstain path prints "No confident recommendation" + exit 2.

**Verify**: capture-first golden — V1 report SHA256 byte-identical
before/after (`81c15fe6…`, 5437 B). 27 new tests + full suite green.

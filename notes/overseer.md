# Overseer notes (2026-07-21)

Read all of PROJECT_STATE.md and every notes/*.md, plus README, configs,
results/small (metrics.json, report.md) and both verifier findings.

Deliverable written: `PROJECT_STATUS.md` (repo root) — student-facing status:
(1) what was built, (2) exact run commands, (3) honest small-run snapshot
(74 rows; winners rem 38/cdr 35/zne 1/raw 0; independent grouped CV
0.823 +/- 0.088 vs 0.521 baseline, LOFO 0.808; the pipeline's 0.905 is
training-set and must not be quoted), (4) limitations incl. all-simulated
noise + IBM token location (`configs\hardware.yaml`, ibm_token — token alone
does nothing until RealHardwareBackend is implemented), (5) next 5 steps
(full.yaml, hardware, CDR regressor ablation, arXiv check, QOSF/mentor),
(6) 14 specific unresolved items compiled from the logs.

No code or shared state was modified; PROJECT_STATE.md untouched.

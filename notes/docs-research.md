# docs-research — research-pass documentation update (2026-07-21/22)

Task: bring README.md, PROJECT_STATUS.md and docs/LEARNING_GUIDE.md in line
with the landed research-run upgrades (noise scaling, raw_plus, seed-averaged
labels, source-level screening, LOFO/LOBO/LODO, REM damping floor). Owned
files touched: those three + this note. Nothing else edited.

## Verification done BEFORE writing (numbers checked against code/configs)

- `pytest tests -q` (slow included) run fresh during this pass:
  **424 passed, 0 failed** (330 s, 3 benign mitiq warnings) — the "424 tests"
  claims in README-adjacent docs are measured, not copied.
- `scripts/train_model.py` argparse: `--data` and `--results` are ALIASES for
  the same dest; `--label {best_technique, best_technique_cost_aware, both}`.
  Both spellings in the docs are therefore valid.
- `scripts/make_report.py`: `--data --metrics --out`, all required. Matches
  every documented command.
- `scripts/run_experiment.py`: default `--config` is `configs/tiny.yaml`
  (README claim kept).
- `mitigation.py`: TECHNIQUES = [raw, raw_plus, zne, cdr, rem];
  SHOT_MULTIPLIER raw_plus = 11 (derived max); REM_MIN_DAMPING = 0.02.
- `model.py`: metrics keys lofo/lobo/lodo present; dropped_classes wired.
- `configs/research.yaml`: 1620 units (180 circuits x 9 environments),
  4096 shots, ~5 h final estimate (5.4 s/unit basis x2 margin) — quoted as an
  ESTIMATE in all docs. `configs/research_smoke.yaml`: 45 units (~90-102 s
  measured => "~2 min").
- `results/research_smoke/` artifacts inspected directly: aggregated.csv =
  15 rows x 27 cols (seed-mean feat_*, <tech>_n_seeds, winners from means);
  report.md has 7 sections, winner_vs_noise.png, realized-noise table
  (Lagos readout 0.204 -> 0.260 at x1.5), LOFO/LOBO/LODO sections, both-labels
  table. Smoke numbers quoted in docs (raw_plus 0.5604 vs raw 0.5615; 28.9%
  per-seed/aggregated label disagreement; Lagos realized ~x1.28/x1.44;
  monotonicity probe 0.60/0.78/0.91) cross-checked against notes/fixer.md,
  notes/tester-research.md and the research.yaml header.
- `results/research` does NOT exist yet — all docs present the sweep as
  next-step / estimate, never as a result.

## README.md changes

1. Header technique list: added `raw_plus` (equal-budget control).
2. Install: "skip this line if .venv already exists" on the venv step
   (PROJECT_STATUS §6.12 nit).
3. Quickstart config table: tiny "~4 min" -> "~1 min" (stale, §6.12); added
   research_smoke (45 units, ~2 min) and research (1620 units, ~5 h estimate)
   rows; full.yaml marked legacy.
4. NEW section "The research run (`configs\research.yaml`)" before the
   hardware section: contents (1620 units, source-balanced 36/family), the
   noise-scale dimension with BOTH honest caveats (cap compression on Lagos —
   realized ~x1.28/x1.44 vs nominal x1.5/x2.0; x1.0 vs scaled noise-character
   difference), raw_plus rationale + smoke result, seed-averaged aggregated.csv
   (540 rows), runtime estimate framed as estimate, the exact 3-command
   pipeline (train on aggregated.csv with --label both), smoke dry-run pointer,
   and which numbers to quote (LOFO/LODO headlines, LOBO = interpolation).
5. Roadmap step 3 rewritten: research.yaml supersedes the old "full run" plan;
   Sherbrooke explicitly deferred (6-8x cost per unit).
6. Project-layout configs row updated (research + research_smoke, full = legacy).

## PROJECT_STATUS.md changes

1. Intro: notes the research-upgrade pass landed, sweep NOT yet run.
2. §1 refreshed: backends (noise-scale suffix), mitigation (5 techniques,
   raw_plus, REM damping floor), experiment (source-level screen +
   aggregated.csv with coverage rule), model (dropped_classes,
   LOFO/LOBO/LODO, --label both), report (7 sections, up to 5 plots), tests
   300 -> 424.
3. §2: pytest count 424; step-2 note re aggregated.csv/--label both; scale-up
   paragraph points at research.yaml (+smoke), full.yaml relabeled legacy.
4. §3: added post-upgrade check bullet — retraining on the same 74-row CSV now
   reports honest CV 0.822 / LOFO 0.824 / LOBO 0.797 (matches the old
   independent hand-check; 0.905 can no longer occur).
5. NEW §3a "Research-run upgrades (landed, sweep NOT yet run)": 8-item summary
   with measured evidence and pointers to notes/. Numbered §3a on purpose —
   §4/§5/§6 references from other files (LEARNING_GUIDE §4.8, notes) stay valid.
6. §4 limitations refreshed in place (no renumbering): 3 (still-74-rows until
   sweep), 4 (CDR coverage now 74% of research units, all 5 families),
   5 (backend-ID -> continuous axis after sweep; quote LODO), 6 (raw_plus
   exists; cost-aware LABEL still sqrt proxy), 7 (skew fixed at source),
   9 (angle-blind features <-> aggregated-label semantics), NEW 10 (noise-dial
   approximation caveats).
7. §5 step 1: "run full.yaml" replaced by the research-sweep 3-command block +
   headline guidance (LOFO/LODO; per-seed vs aggregated ablation).
8. §6 statuses refreshed: 1, 3, 4, 6, 8, 9 RESOLVED (with §3a pointers);
   12 mostly fixed (mojibake remains); 2, 5, 7, 10, 13, 14 unchanged/open.

## docs/LEARNING_GUIDE.md changes (short additions per task)

1. §3: new "Noise scaling (`@x<scale>` backend names)" paragraph — what the
   dial does, why (continuous noise axis), and the honest approximation caveat
   (synthetic depolarizing+readout rebuild, character change at the first
   scaling step, cap compression, quote realized rates).
2. §4: heading now "three techniques, two baselines"; new "### raw_plus"
   subsection — why equal-budget baselines matter (bias vs variance: extra
   shots shrink the error bar around the WRONG number).
3. §7: label-noise caveat expanded to explain seed-averaged labels (sqrt(n)
   noise reduction, ~29% smoke disagreement = the removed label noise, and the
   angle-blind-features subtlety: the aggregated label is per circuit-KIND);
   new "How generalization is scored" block defining LOFO / LOBO
   (interpolation, do-not-oversell) / LODO; model.py table row updated.

## Honest-tone checks

- Research-run duration always "~5 h (estimate)" — never presented as measured.
- raw_plus described as barely beating raw AND why that is the point.
- LOBO consistently labeled interpolation; LOFO/LODO the headline claims.
- Small-run §3 numbers left intact (they are the verified results); skew and
  74-row caveats kept, marked "fixed for future runs" only where true.

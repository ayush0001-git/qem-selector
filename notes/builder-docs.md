# Notes — builder-docs (2026-07-21)

## Delivered
- `README.md` — overview, honest motivation, install (venv + pinned reqs +
  `pip install -e .`), quickstart against `configs\experiment.yaml`, tiny-config
  YAML example (schema matches `generate_suite` + `run_experiment` stub
  docstrings exactly), full workflow commands, real-IBM-hardware section,
  layout table, roadmap-to-paper (tiny -> small -> full -> hardware -> CDR
  regressor ablation as the novel angle -> arXiv/mitiq-Calibrator lit check).
- `docs/LEARNING_GUIDE.md` — quantum-for-ML-students guide; each module mapped
  to its concept; CDR framed as supervised regression with classically
  computable labels; facts cross-checked against spike notes.
- `.gitignore` — required entries plus `*.egg-info/`, `.pytest_cache/`, and
  `configs/hardware.yaml`.

## Integrator must know
1. **`.gitignore` ignores `configs/hardware.yaml`** (token safety; README says
   "never commit"). Since builder-experiment ships that file as a placeholder,
   consider committing a `configs/hardware.yaml.example` instead, or delete the
   gitignore line if you prefer committing the null placeholder.
2. README documents CLI flags verbatim from INTERFACES.md for
   `run_experiment.py` / `train_model.py`. Flags for `make_report.py` /
   `recommend.py` were NOT specified by the architect — README shows plausible
   invocations with an explicit "run --help for exact flags" caveat. If
   builder-recommend used different flags, update README's "Full workflow"
   section (only place they appear).
3. README's quickstart assumes `configs\experiment.yaml` exists
   (builder-experiment deliverable).
4. No test file: builder-docs owns only docs; per task, testsPass=true after
   proofreading commands against INTERFACES.md and stub docstrings.

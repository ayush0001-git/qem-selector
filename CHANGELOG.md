# Changelog

All notable changes to the `qem-selector` project are documented in this file.

## [0.1.0] - 2026-08-19

### Added
- **High-level public APIs:** `MitigatedExecutor` and convenience wrapper `run` in `qemsel.api`.
- **Angle 4 (GNN Selector):** DAG graph circuit representation in `qemsel.features.convert_circuit_to_graph` and a demonstration training pipeline in `scripts/train_gnn_selector.py`.
- **Interactive sweeps:** Standalone CLI tool `scripts/sweep_lambda.py` to swept shot-cost multipliers and format LaTeX/Markdown comparison tables.
- **Dual OS Quickstart:** Added macOS / Linux installation and runtime instructions to the `README.md`.
- **CI Pipeline:** Setup automated test runs in `.github/workflows/ci.yml` supporting multi-version Python targets (3.12 and 3.13).
- **Jupyter Demo:** Added `demo.ipynb` illustrating executor operations, error mitigation fallbacks, and circuit graph DAGs.
- **Testing specs:** Added regression guidelines and fixture anchors in `docs/TESTING_GUIDE.md`.

### Changed
- **Lighter Library Code:** Cleaned up copy-pasted/vibe-coding file duplicates (consolidated test files, removed dead comments citing `INTERFACES.md`).
- **Standardized Logging:** Refactored direct `print` statements in library modules (`api.py`, `model.py`, `experiment.py`) to run through the standard library `logging` framework.
- **Safer Defaults:** Switched physical QPU simulation tests to safe mode (`hardware_confirmed: false`) to safeguard QPU runtime allocation.
- **Robust Path Resolving:** Scripts now calculate reference paths dynamically relative to execution location.

### Fixed
- **Thread Safety:** Added thread locks (`threading.Lock()`) around global backend info cache reads/writes to prevent race conditions during parallel sweeps.
- **Python Optimizations (-O):** Replaced keyword `assert` guards with robust `RuntimeError` checks in features and circuit pipelines so order-invariance checks remain active under optimized Python execution.

"""V2 tests for qemsel.experiment — the shots-as-list axis (builder B4).

Covers ``_normalize_shots`` and the list-mode behavior of ``run_experiment``:
the ``base_shots`` cross-product dimension, per-budget resume key, per-budget
executors / <tech>_shots / winner labels, the once-per-(circuit,backend)
low-signal screen, the list-only ``s{base_shots}`` errors.log field,
feature_version 2 wiring, and the V2 aggregated.csv grouping.

Backward-compat is pinned two ways:
  * a FAST fake-based scalar test asserting the V1 schema (no base_shots
    column) is untouched, and
  * a SLOW real-simulation regression: a fresh ``configs/tiny.yaml`` run is
    byte-identical to the stored reference ``results/tiny/results.csv``
    (capture-first: the reference was produced by the pre-V2 code).

Like tests/test_experiment.py, the heavy collaborators are monkeypatched with
fast deterministic fakes so the fake-based tests need no quantum simulation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from qiskit import QuantumCircuit

from qemsel.circuits import CircuitSpec
from qemsel.experiment import (
    BASE_SHOTS_COLUMN,
    COST_AWARE_COLUMN,
    _normalize_shots,
    run_experiment,
)
from qemsel.features import FEATURE_NAMES, FEATURE_NAMES_BY_VERSION, FEATURE_NAMES_V2

# A technique set mixing V1 and V2 names — exercises TECHNIQUES_V2 validation.
LIST_TECHNIQUES = ["raw", "zne", "zne_fr", "cdr_ridge"]
MULT = {"raw": 1, "raw_plus": 11, "zne": 3, "zne_fr": 1, "cdr": 11, "cdr_ridge": 11, "rem": 3}
IDEAL_VALUE = 0.5
#: plain winner = zne (err 0.02); cost-aware winner = zne (0.02*sqrt3 ~ 0.035
#: beats raw 0.30, zne_fr 0.15, cdr_ridge 0.10*sqrt11 ~ 0.332).
ERR = {"raw": 0.30, "zne": 0.02, "zne_fr": 0.15, "cdr_ridge": 0.10, "cdr": 0.10, "rem": 0.20}


# ---------------------------------------------------------------------------
# _normalize_shots — the sole shots validator/normalizer
# ---------------------------------------------------------------------------


class TestNormalizeShots:
    def test_scalar_int_is_single_budget_scalar_mode(self):
        assert _normalize_shots(2048) == ([2048], False)

    def test_list_is_list_mode_order_preserved(self):
        assert _normalize_shots([256, 1024, 4096]) == ([256, 1024, 4096], True)
        # order is the innermost-loop order — must NOT be sorted
        assert _normalize_shots([4096, 256, 1024]) == ([4096, 256, 1024], True)

    def test_single_element_list_is_still_list_mode(self):
        # a length-1 LIST opts into list mode (base_shots column), unlike a
        # bare scalar — the type, not the length, decides.
        assert _normalize_shots([256]) == ([256], True)

    def test_tuple_accepted_as_list(self):
        assert _normalize_shots((256, 1024)) == ([256, 1024], True)

    @pytest.mark.parametrize("bad", [0, -1, -4096])
    def test_non_positive_scalar_raises(self, bad):
        with pytest.raises(ValueError, match="shots"):
            _normalize_shots(bad)

    def test_bool_scalar_raises(self):
        # bool is an int subclass — must be rejected, not treated as 1/0.
        with pytest.raises(ValueError, match="shots"):
            _normalize_shots(True)

    def test_float_scalar_raises(self):
        with pytest.raises(ValueError, match="shots"):
            _normalize_shots(2048.0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _normalize_shots([])

    def test_duplicate_entries_raise(self):
        with pytest.raises(ValueError, match="distinct"):
            _normalize_shots([256, 256, 1024])

    @pytest.mark.parametrize("bad_list", [[256, 0], [256, -1], [256, True], [256, 3.5], [256, "x"]])
    def test_bad_list_entry_raises(self, bad_list):
        with pytest.raises(ValueError, match="shots"):
            _normalize_shots(bad_list)

    def test_string_raises(self):
        with pytest.raises(ValueError, match="shots"):
            _normalize_shots("2048")


# ---------------------------------------------------------------------------
# Fake collaborator stack (deterministic, no simulation)
# ---------------------------------------------------------------------------


def _fake_generate_suite(config: dict) -> list[tuple[QuantumCircuit, CircuitSpec]]:
    out = []
    for family in config["families"]:
        for n in config["n_qubits"]:
            for d in config["depths"]:
                for s in config["seeds"]:
                    qc = QuantumCircuit(n)
                    qc.h(0)
                    out.append((qc, CircuitSpec(family, n, d, s)))
    return out


def _fake_features(
    circuit: QuantumCircuit,
    backend_name: str,
    *,
    version: int = 1,
    base_shots: int | float | None = None,
) -> dict[str, float]:
    """Version-aware fake. For version 2, log2_shots echoes base_shots so a
    test can prove the unit budget reached extract_features."""
    names = FEATURE_NAMES_BY_VERSION[version]
    feats = {name: float(i) for i, name in enumerate(names)}
    feats["n_qubits"] = float(circuit.num_qubits)
    if version == 2:
        assert base_shots is not None and base_shots > 0
        feats["log2_shots"] = math.log2(base_shots)
    return feats


@pytest.fixture()
def list_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Patch the heavy stack; return call logs for apply_technique/make_executor."""
    apply_calls: list[tuple] = []
    made: list[tuple] = []

    def fake_apply(name, circuit, pauli, executor, backend_name, shots, seed):
        apply_calls.append((name, backend_name, shots, seed))
        return IDEAL_VALUE + ERR[name]

    def fake_make(backend_name, shots, seed):
        made.append((backend_name, shots, seed))
        return lambda c, p: IDEAL_VALUE

    monkeypatch.setattr("qemsel.circuits.generate_suite", _fake_generate_suite)
    monkeypatch.setattr("qemsel.ideal.ideal_expectation", lambda c, p: IDEAL_VALUE)
    monkeypatch.setattr("qemsel.features.extract_features", _fake_features)
    monkeypatch.setattr("qemsel.backends.make_executor", fake_make)
    monkeypatch.setattr("qemsel.mitigation.apply_technique", fake_apply)
    monkeypatch.setattr(
        "qemsel.mitigation.shots_consumed", lambda name, base: base * MULT[name]
    )
    return {"apply": apply_calls, "made": made}


def _list_config(
    shots: Any,
    *,
    seeds: list[int] | None = None,
    feature_version: int = 2,
    techniques: list[str] | None = None,
    **overrides: Any,
) -> dict:
    config = {
        "circuits": {
            "families": ["mirror_circuit"],
            "n_qubits": [2],
            "depths": [4],
            "seeds": seeds if seeds is not None else [0],
        },
        "backends": ["FakeManilaV2"],
        "shots": shots,
        "pauli": "auto",
        "techniques": techniques if techniques is not None else list(LIST_TECHNIQUES),
        "feature_version": feature_version,
    }
    config.update(overrides)
    return config


def _expected_list_columns(techniques: list[str], feature_names: list[str]) -> list[str]:
    cols = ["circuit_id", "family", "n_qubits", "depth", "seed", "backend", "pauli"]
    cols += [BASE_SHOTS_COLUMN, "ideal"]
    cols += [f"feat_{n}" for n in feature_names]
    for t in techniques:
        cols += [f"{t}_value", f"{t}_abs_error", f"{t}_shots"]
    return cols + ["best_technique", COST_AWARE_COLUMN]


# ---------------------------------------------------------------------------
# List-mode schema + cross-product
# ---------------------------------------------------------------------------


class TestListModeSchema:
    def test_base_shots_column_between_pauli_and_ideal(self, list_stack, out_dir: Path):
        df = run_experiment(_list_config([256, 1024, 4096]), out_dir)
        cols = list(df.columns)
        assert cols == _expected_list_columns(LIST_TECHNIQUES, FEATURE_NAMES_V2)
        # exactly between 'pauli' and 'ideal'
        assert cols[cols.index("pauli") + 1] == BASE_SHOTS_COLUMN
        assert cols[cols.index(BASE_SHOTS_COLUMN) + 1] == "ideal"

    def test_cross_product_units_and_budget_values(self, list_stack, out_dir: Path):
        # 2 circuits (seeds) x 1 backend x 3 budgets = 6 units
        df = run_experiment(_list_config([256, 1024, 4096], seeds=[0, 1]), out_dir)
        assert len(df) == 6
        assert sorted(df[BASE_SHOTS_COLUMN].unique().tolist()) == [256, 1024, 4096]
        # every (seed, budget) pair appears once
        pairs = set(zip(df["seed"], df[BASE_SHOTS_COLUMN]))
        assert pairs == {(s, b) for s in (0, 1) for b in (256, 1024, 4096)}

    def test_budgets_are_innermost_loop(self, list_stack, out_dir: Path):
        # loop order: circuit (outer) x backend x budget (innermost)
        run_experiment(
            _list_config([256, 1024], seeds=[0, 1], backends=["FakeManilaV2"]),
            out_dir,
        )
        df = pd.read_csv(out_dir / "results.csv")
        # rows appended in order -> budgets cycle fastest within a circuit
        got = list(zip(df["seed"], df[BASE_SHOTS_COLUMN]))
        assert got == [(0, 256), (0, 1024), (1, 256), (1, 1024)]

    def test_per_unit_executor_built_at_unit_budget(self, list_stack, out_dir: Path):
        run_experiment(_list_config([256, 1024, 4096]), out_dir)
        made_shots = sorted(shots for _, shots, _ in list_stack["made"])
        assert made_shots == [256, 1024, 4096]

    def test_tech_shots_use_unit_budget(self, list_stack, out_dir: Path):
        df = run_experiment(_list_config([256, 4096]), out_dir)
        for _, row in df.iterrows():
            base = row[BASE_SHOTS_COLUMN]
            for tech in LIST_TECHNIQUES:
                assert row[f"{tech}_shots"] == base * MULT[tech]

    def test_apply_technique_receives_unit_budget(self, list_stack, out_dir: Path):
        run_experiment(_list_config([256, 4096]), out_dir)
        shots_seen = sorted({shots for _, _, shots, _ in list_stack["apply"]})
        assert shots_seen == [256, 4096]

    def test_winners_use_per_budget_base(self, list_stack, out_dir: Path):
        df = run_experiment(_list_config([256, 4096]), out_dir)
        # deterministic winners independent of budget with constant ERR
        assert (df["best_technique"] == "zne").all()
        assert (df[COST_AWARE_COLUMN] == "zne").all()

    def test_feature_version_2_columns_and_log2_shots(self, list_stack, out_dir: Path):
        df = run_experiment(_list_config([256, 4096]), out_dir)
        # v2 feature columns present
        for name in FEATURE_NAMES_V2:
            assert f"feat_{name}" in df.columns
        # log2_shots echoes the unit budget (proves base_shots reached features)
        for _, row in df.iterrows():
            assert row["feat_log2_shots"] == pytest.approx(math.log2(row[BASE_SHOTS_COLUMN]))


# ---------------------------------------------------------------------------
# Resume / crash-safety in list mode
# ---------------------------------------------------------------------------


class TestListModeResume:
    def test_resume_by_budget_only_computes_missing(self, list_stack, out_dir: Path):
        run_experiment(_list_config([256]), out_dir)
        n_after_first = len(list_stack["apply"])
        assert n_after_first == 4  # 1 unit x 4 techniques
        df = run_experiment(_list_config([256, 1024]), out_dir)
        # only the 1024 unit is new (+4 apply calls); 256 skipped
        assert len(list_stack["apply"]) == 8
        assert len(df) == 2
        assert sorted(df[BASE_SHOTS_COLUMN]) == [256, 1024]
        on_disk = pd.read_csv(out_dir / "results.csv")
        assert not on_disk.duplicated(
            subset=["circuit_id", "backend", BASE_SHOTS_COLUMN]
        ).any()

    def test_completed_list_run_fully_skips(self, list_stack, out_dir: Path):
        cfg = _list_config([256, 1024], seeds=[0, 1])
        df1 = run_experiment(cfg, out_dir)
        n = len(list_stack["apply"])
        df2 = run_experiment(cfg, out_dir)
        assert len(list_stack["apply"]) == n  # nothing recomputed
        assert len(df2) == len(df1) == 4

    def test_torn_tail_repaired_and_recomputed(self, list_stack, out_dir: Path):
        run_experiment(_list_config([256, 1024]), out_dir)
        csv_path = out_dir / "results.csv"
        lines = csv_path.read_bytes().splitlines(keepends=True)
        torn = lines[-1].rstrip(b"\r\n")
        torn = torn[: len(torn) // 2]
        csv_path.write_bytes(b"".join(lines[:-1]) + torn)
        df = run_experiment(_list_config([256, 1024]), out_dir)
        assert len(df) == 2
        on_disk = pd.read_csv(csv_path)  # must not raise ParserError
        assert len(on_disk) == 2
        assert not on_disk.duplicated(
            subset=["circuit_id", "backend", BASE_SHOTS_COLUMN]
        ).any()

    def test_old_scalar_csv_plus_list_config_raises(self, list_stack, out_dir: Path):
        # A scalar (V1-schema, NO base_shots column) results.csv resumed with a
        # LIST config must fail the column-equality check, never silently mix.
        run_experiment(_list_config(256, feature_version=2), out_dir)  # scalar
        scalar_cols = list(pd.read_csv(out_dir / "results.csv").columns)
        assert BASE_SHOTS_COLUMN not in scalar_cols  # only diff vs the list schema
        with pytest.raises(ValueError, match="columns"):
            run_experiment(_list_config([256, 1024], feature_version=2), out_dir)


# ---------------------------------------------------------------------------
# Low-signal screen, errors.log, aggregated — list-mode specifics
# ---------------------------------------------------------------------------


class TestListModeLogsAndScreen:
    def test_low_signal_screen_once_per_circuit_backend(
        self, list_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # ideal 0.5 < 0.6 -> every (circuit, backend) screened out, ONE log
        # line per pair regardless of how many budgets it has.
        cfg = _list_config([256, 1024, 4096], seeds=[0, 1], min_abs_ideal=0.6)
        df = run_experiment(cfg, out_dir)
        assert len(df) == 0
        assert len(list_stack["apply"]) == 0  # no noisy work on any budget
        log = (out_dir / "skipped_low_signal.log").read_text(encoding="utf-8")
        # 2 circuits x 1 backend = 2 lines, NOT 2 x 3 budgets = 6
        assert len([ln for ln in log.splitlines() if ln.strip()]) == 2

    def test_errors_log_gains_budget_field_in_list_mode(
        self, list_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def flaky(name, circuit, pauli, executor, backend, shots, seed):
            if name == "zne":
                raise RuntimeError("zne boom")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", flaky)
        run_experiment(_list_config([256, 1024]), out_dir)
        log = (out_dir / "errors.log").read_text(encoding="utf-8")
        # list-mode prefix: circuit_id,backend,s{base_shots},technique: ...
        assert "mirror_circuit_q2_d4_s0,FakeManilaV2,s256,zne:" in log
        assert "mirror_circuit_q2_d4_s0,FakeManilaV2,s1024,zne:" in log
        assert "zne boom" in log

    def test_failed_technique_nan_triple_in_list_mode(
        self, list_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def flaky(name, circuit, pauli, executor, backend, shots, seed):
            if name == "cdr_ridge":
                raise RuntimeError("refused")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", flaky)
        df = run_experiment(_list_config([256, 1024]), out_dir)
        assert df["cdr_ridge_value"].isna().all()
        assert df["cdr_ridge_abs_error"].isna().all()
        assert df["cdr_ridge_shots"].isna().all()
        assert (df["best_technique"] == "zne").all()  # winner excludes failed


class TestListModeAggregated:
    def test_aggregated_grouped_by_base_shots(self, list_stack, out_dir: Path):
        # 2 seeds x 2 budgets -> 2 aggregate rows (one per budget), seeds
        # averaged WITHIN a budget, never across budgets.
        run_experiment(_list_config([256, 1024], seeds=[0, 1]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert BASE_SHOTS_COLUMN in agg.columns
        # base_shots sits right after backend (V2 key order)
        cols = list(agg.columns)
        assert cols[cols.index("backend") + 1] == BASE_SHOTS_COLUMN
        assert len(agg) == 2
        assert sorted(agg[BASE_SHOTS_COLUMN]) == [256, 1024]
        assert (agg["n_seeds"] == 2).all()
        for _, row in agg.iterrows():
            for tech in LIST_TECHNIQUES:
                assert row[f"{tech}_mean_abs_error"] == pytest.approx(ERR[tech])
            assert row["best_technique"] == "zne"

    def test_aggregated_has_v2_feature_columns(self, list_stack, out_dir: Path):
        run_experiment(_list_config([256, 4096], seeds=[0, 1]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        for name in FEATURE_NAMES_V2:
            assert f"feat_{name}" in agg.columns
        # log2_shots aggregate equals the group's (constant) budget log2
        for _, row in agg.iterrows():
            assert row["feat_log2_shots"] == pytest.approx(math.log2(row[BASE_SHOTS_COLUMN]))


# ---------------------------------------------------------------------------
# feature_version validation
# ---------------------------------------------------------------------------


class TestFeatureVersion:
    def test_unknown_feature_version_raises(self, list_stack, out_dir: Path):
        with pytest.raises(ValueError, match="feature_version"):
            run_experiment(_list_config([256], feature_version=3), out_dir)

    def test_feature_version_bool_raises(self, list_stack, out_dir: Path):
        with pytest.raises(ValueError, match="feature_version"):
            run_experiment(_list_config([256], feature_version=True), out_dir)

    def test_v2_technique_names_accepted(self, list_stack, out_dir: Path):
        # zne_fr / cdr_ridge are TECHNIQUES_V2 names — must validate.
        df = run_experiment(
            _list_config([256], techniques=["raw", "zne_fr", "cdr_ridge"]), out_dir
        )
        assert "zne_fr_value" in df.columns and "cdr_ridge_value" in df.columns


# ---------------------------------------------------------------------------
# Backward compat: scalar mode is byte-identical to V1
# ---------------------------------------------------------------------------


class TestScalarBackwardCompat:
    def test_scalar_v1_schema_has_no_base_shots_column(self, list_stack, out_dir: Path):
        # scalar shots + feature_version 1 (default V1) -> V1 results schema.
        cfg = _list_config(128, feature_version=1, techniques=["raw", "zne", "cdr", "rem"])
        df = run_experiment(cfg, out_dir)
        assert BASE_SHOTS_COLUMN not in df.columns
        v1_cols = (
            ["circuit_id", "family", "n_qubits", "depth", "seed", "backend", "pauli", "ideal"]
            + [f"feat_{n}" for n in FEATURE_NAMES]
            + [c for t in ("raw", "zne", "cdr", "rem") for c in (f"{t}_value", f"{t}_abs_error", f"{t}_shots")]
            + ["best_technique", COST_AWARE_COLUMN]
        )
        assert list(df.columns) == v1_cols

    def test_scalar_aggregated_has_no_base_shots_column(self, list_stack, out_dir: Path):
        cfg = _list_config(128, feature_version=1, techniques=["raw", "zne"])
        run_experiment(cfg, out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert BASE_SHOTS_COLUMN not in agg.columns
        assert list(agg.columns[:4]) == ["family", "n_qubits", "depth", "backend"]


REFERENCE_TINY = Path(__file__).resolve().parents[1] / "results" / "tiny" / "results.csv"


@pytest.mark.slow
@pytest.mark.skipif(
    not REFERENCE_TINY.exists(),
    reason="stored reference results/tiny/results.csv not present",
)
def test_fresh_tiny_run_byte_identical_to_reference(tmp_path: Path):
    """Capture-first regression: a fresh scalar configs/tiny.yaml run (real
    simulation) must reproduce the stored reference byte-for-byte — the shots
    axis must not perturb the V1 scalar path."""
    import yaml

    from qemsel.experiment import run_experiment as run

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "tiny.yaml").read_text(encoding="utf-8"))
    run(config, tmp_path)
    fresh = (tmp_path / "results.csv").read_bytes()
    assert fresh == REFERENCE_TINY.read_bytes()

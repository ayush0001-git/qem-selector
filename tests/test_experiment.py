"""Unit tests for qemsel.experiment.run_experiment.

All heavy dependencies (circuit generation, noisy executors, mitigation,
features, ideal values) are monkeypatched with fast deterministic fakes, so
these tests run standalone in seconds with NO quantum simulation — even while
the other qemsel modules are still stubs.

run_experiment accesses collaborators as module attributes
(``_mitigation.apply_technique`` etc.), so patching ``qemsel.<module>.<fn>``
is picked up at call time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit

from qemsel.circuits import CircuitSpec
from qemsel.experiment import COST_AWARE_COLUMN, run_experiment

TECHNIQUES = ["raw", "zne", "cdr", "rem"]
#: The full default list (mitigation.TECHNIQUES) incl. the raw_plus
#: equal-budget baseline; most tests here pass the explicit 4-technique
#: list above to stay independent of mitigation's defaults.
DEFAULT_TECHNIQUES = ["raw", "raw_plus", "zne", "cdr", "rem"]
MULT = {"raw": 1, "raw_plus": 11, "zne": 3, "cdr": 11, "rem": 3}
FEATURE_NAMES = [
    "n_qubits",
    "depth",
    "n_1q_gates",
    "n_2q_gates",
    "n_cnot",
    "n_non_clifford",
    "clifford_fraction",
    "depth_per_qubit",
    "backend_avg_2q_error",
    "backend_avg_readout_error",
]

IDEAL_VALUE = 0.5
#: Deterministic per-technique error offsets. Plain argmin winner: zne.
#: Cost-aware scores err*sqrt(mult): raw 0.30, raw_plus ~0.83, zne ~0.035,
#: cdr ~0.33, rem ~0.35 -> cost-aware winner also zne.
ERR = {"raw": 0.30, "raw_plus": 0.25, "zne": 0.02, "cdr": 0.10, "rem": 0.20}


def _fake_generate_suite(config: dict) -> list[tuple[QuantumCircuit, CircuitSpec]]:
    """Deterministic stand-in for circuits.generate_suite (same nesting order)."""
    out = []
    for family in config["families"]:
        for n in config["n_qubits"]:
            for d in config["depths"]:
                for s in config["seeds"]:
                    qc = QuantumCircuit(n)
                    qc.h(0)
                    out.append((qc, CircuitSpec(family, n, d, s)))
    return out


def _fake_features(circuit: QuantumCircuit, backend_name: str) -> dict[str, float]:
    """Deterministic stand-in for features.extract_features."""
    feats = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    feats["n_qubits"] = float(circuit.num_qubits)
    return feats


def _make_config(seeds: list[int] | None = None, **overrides: Any) -> dict:
    """A 4-circuit x 1-backend config (4 units) unless overridden."""
    config = {
        "circuits": {
            "families": ["mirror_circuit"],
            "n_qubits": [2],
            "depths": [4],
            "seeds": seeds if seeds is not None else [0, 1, 2, 3],
        },
        "backends": ["FakeManilaV2"],
        "shots": 128,
        "pauli": "auto",
        "techniques": list(TECHNIQUES),
    }
    config.update(overrides)
    return config


@pytest.fixture()
def patched_stack(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Patch all heavy collaborators; returns the apply_technique call log."""
    calls: list[tuple] = []

    def fake_apply(name, circuit, pauli, executor, backend_name, shots, seed):
        calls.append((name, backend_name, shots, seed))
        return IDEAL_VALUE + ERR[name]

    monkeypatch.setattr("qemsel.circuits.generate_suite", _fake_generate_suite)
    monkeypatch.setattr("qemsel.ideal.ideal_expectation", lambda c, p: IDEAL_VALUE)
    monkeypatch.setattr("qemsel.features.extract_features", _fake_features)
    monkeypatch.setattr(
        "qemsel.backends.make_executor",
        lambda backend_name, shots, seed: (lambda c, p: IDEAL_VALUE),
    )
    monkeypatch.setattr("qemsel.mitigation.apply_technique", fake_apply)
    monkeypatch.setattr(
        "qemsel.mitigation.shots_consumed", lambda name, base: base * MULT[name]
    )
    return calls


def _expected_columns(techniques: list[str]) -> list[str]:
    cols = ["circuit_id", "family", "n_qubits", "depth", "seed", "backend", "pauli", "ideal"]
    cols += [f"feat_{n}" for n in FEATURE_NAMES]
    for t in techniques:
        cols += [f"{t}_value", f"{t}_abs_error", f"{t}_shots"]
    return cols + ["best_technique", COST_AWARE_COLUMN]


class TestWellFormedRun:
    def test_four_circuit_run_shape_and_columns(self, patched_stack, out_dir: Path):
        df = run_experiment(_make_config(), out_dir)
        assert len(df) == 4  # 4 circuits x 1 backend
        assert list(df.columns) == _expected_columns(TECHNIQUES)
        # 4 techniques x 4 units
        assert len(patched_stack) == 16

    def test_row_contents(self, patched_stack, out_dir: Path):
        df = run_experiment(_make_config(), out_dir)
        row = df.iloc[0]
        assert row["circuit_id"] == "mirror_circuit_q2_d4_s0"
        assert row["family"] == "mirror_circuit"
        assert row["n_qubits"] == 2
        assert row["depth"] == 4
        assert row["seed"] == 0
        assert row["backend"] == "FakeManilaV2"
        assert row["pauli"] == "ZZ"  # auto => 'Z' * n_qubits
        assert row["ideal"] == pytest.approx(IDEAL_VALUE)
        for tech in TECHNIQUES:
            assert row[f"{tech}_value"] == pytest.approx(IDEAL_VALUE + ERR[tech])
            assert row[f"{tech}_abs_error"] == pytest.approx(ERR[tech])
            assert row[f"{tech}_shots"] == 128 * MULT[tech]
        assert row["feat_n_qubits"] == 2.0

    def test_csv_matches_returned_dataframe(self, patched_stack, out_dir: Path):
        df = run_experiment(_make_config(), out_dir)
        on_disk = pd.read_csv(out_dir / "results.csv")
        assert list(on_disk.columns) == list(df.columns)
        assert len(on_disk) == len(df)
        assert list(on_disk["circuit_id"]) == list(df["circuit_id"])

    def test_run_meta_sidecar(self, patched_stack, out_dir: Path):
        config = _make_config()
        run_experiment(config, out_dir)
        meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["config"]["shots"] == 128
        assert meta["config"]["backends"] == ["FakeManilaV2"]
        for pkg in ("qiskit", "qiskit-aer", "mitiq", "numpy", "scikit-learn", "pandas"):
            assert pkg in meta["versions"]
        assert "python_version" in meta
        assert "qemsel_version" in meta
        assert "timestamp" in meta


class TestWinnerColumns:
    def test_winner_matches_recomputed_argmin(self, patched_stack, out_dir: Path):
        df = run_experiment(_make_config(), out_dir)
        for _, row in df.iterrows():
            errs = {t: row[f"{t}_abs_error"] for t in TECHNIQUES}
            valid = {t: e for t, e in errs.items() if not math.isnan(e)}
            expected = min(valid, key=valid.get)
            assert row["best_technique"] == expected

    def test_cost_aware_matches_recomputed_argmin(self, patched_stack, out_dir: Path):
        base_shots = 128
        df = run_experiment(_make_config(), out_dir)
        for _, row in df.iterrows():
            scores = {
                t: row[f"{t}_abs_error"]
                * math.sqrt(row[f"{t}_shots"] / base_shots)
                for t in TECHNIQUES
                if not math.isnan(row[f"{t}_abs_error"])
            }
            assert row[COST_AWARE_COLUMN] == min(scores, key=scores.get)

    def test_cost_aware_can_differ_from_plain_winner(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # cdr has the smallest raw error (0.02) but pays sqrt(11) ~ 3.32x
        # cost penalty: 0.02*3.32 ~ 0.066 > raw's 0.05 -> raw wins cost-aware.
        errs = {"raw": 0.05, "zne": 0.50, "cdr": 0.02, "rem": 0.60}
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique",
            lambda name, *a, **k: IDEAL_VALUE + errs[name],
        )
        df = run_experiment(_make_config(seeds=[0]), out_dir)
        assert df.iloc[0]["best_technique"] == "cdr"
        assert df.iloc[0][COST_AWARE_COLUMN] == "raw"


class TestFailureIsolation:
    def test_one_technique_failing_yields_nan_and_log(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def flaky_apply(name, circuit, pauli, executor, backend_name, shots, seed):
            if name == "cdr":
                raise RuntimeError("cdr exploded")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", flaky_apply)
        df = run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert len(df) == 2
        assert df["cdr_value"].isna().all()
        assert df["cdr_abs_error"].isna().all()
        assert df["cdr_shots"].isna().all()
        # other techniques unaffected; winner excludes the failed one
        assert (df["best_technique"] == "zne").all()
        log = (out_dir / "errors.log").read_text(encoding="utf-8")
        assert "mirror_circuit_q2_d4_s0,FakeManilaV2,cdr:" in log
        assert "cdr exploded" in log

    def test_all_techniques_failing_gives_empty_winner(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def always_fail(name, *a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr("qemsel.mitigation.apply_technique", always_fail)
        df = run_experiment(_make_config(seeds=[0]), out_dir)
        assert len(df) == 1
        assert df.iloc[0]["best_technique"] == ""
        assert df.iloc[0][COST_AWARE_COLUMN] == ""
        for tech in TECHNIQUES:
            assert math.isnan(df.iloc[0][f"{tech}_value"])

    def test_empty_winner_survives_csv_resume_roundtrip(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        run_experiment(_make_config(seeds=[0]), out_dir)
        # second (resuming) call re-reads the CSV; '' must not become NaN
        df = run_experiment(_make_config(seeds=[0]), out_dir)
        assert df.iloc[0]["best_technique"] == ""
        assert df.iloc[0][COST_AWARE_COLUMN] == ""


class TestRestartSkip:
    def test_completed_units_are_skipped(self, patched_stack, out_dir: Path):
        config = _make_config()
        df1 = run_experiment(config, out_dir)
        n_calls_first = len(patched_stack)
        assert n_calls_first == 16
        df2 = run_experiment(config, out_dir)
        # no new apply_technique calls, same rows returned
        assert len(patched_stack) == n_calls_first
        assert len(df2) == len(df1) == 4
        assert list(df2["circuit_id"]) == list(df1["circuit_id"])

    def test_partial_resume_only_computes_missing_units(
        self, patched_stack, out_dir: Path
    ):
        run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert len(patched_stack) == 8  # 2 units x 4 techniques
        df = run_experiment(_make_config(seeds=[0, 1, 2, 3]), out_dir)
        # only the 2 NEW units were computed (8 more calls), 4 rows total
        assert len(patched_stack) == 16
        assert len(df) == 4
        assert sorted(df["seed"]) == [0, 1, 2, 3]
        # CSV holds all rows exactly once
        on_disk = pd.read_csv(out_dir / "results.csv")
        assert len(on_disk) == 4
        assert not on_disk.duplicated(subset=["circuit_id", "backend"]).any()

    def test_mismatched_existing_columns_raise(self, patched_stack, out_dir: Path):
        (out_dir / "results.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="columns"):
            run_experiment(_make_config(), out_dir)

    def test_torn_tail_is_repaired_and_unit_recomputed(
        self, patched_stack, out_dir: Path
    ):
        """Crash mid-append leaves a partial final line without a newline.
        Pre-fix this either poisoned done_pairs (permanently NaN row) or made
        the next append glue onto the partial line, after which every
        pd.read_csv died with a ParserError. The repair must truncate the
        torn tail and recompute that one unit."""
        run_experiment(_make_config(seeds=[0, 1]), out_dir)
        csv_path = out_dir / "results.csv"
        raw = csv_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        torn_last = lines[-1].rstrip(b"\r\n")
        torn_last = torn_last[: len(torn_last) // 2]  # half a row, no newline
        csv_path.write_bytes(b"".join(lines[:-1]) + torn_last)

        df = run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert len(df) == 2
        on_disk = pd.read_csv(csv_path)  # must not raise ParserError
        assert len(on_disk) == 2
        assert not on_disk.duplicated(subset=["circuit_id", "backend"]).any()
        # the recomputed unit carries real values, not NaN-filled stubs
        assert not on_disk["raw_value"].isna().any()

    def test_append_never_glues_onto_torn_tail(
        self, patched_stack, out_dir: Path
    ):
        """_append_row itself must write a newline before appending to a
        file whose last byte is not a newline (belt and braces)."""
        from qemsel.experiment import _append_row, _result_columns

        run_experiment(_make_config(seeds=[0]), out_dir)
        csv_path = out_dir / "results.csv"
        raw = csv_path.read_bytes().rstrip(b"\r\n")
        csv_path.write_bytes(raw)  # strip the trailing newline
        columns = _result_columns(TECHNIQUES)
        row = {c: "x" for c in columns}
        _append_row(csv_path, row, columns)
        on_disk = pd.read_csv(csv_path)  # parses -> no glued line
        assert len(on_disk) == 2


class TestConfigValidation:
    @pytest.mark.parametrize("missing", ["circuits", "backends", "shots"])
    def test_missing_required_key_raises(self, patched_stack, out_dir, missing):
        config = _make_config()
        del config[missing]
        with pytest.raises(ValueError, match=missing):
            run_experiment(config, out_dir)

    def test_unknown_backend_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="backend"):
            run_experiment(_make_config(backends=["FakeNopeV2"]), out_dir)

    def test_unknown_technique_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="technique"):
            run_experiment(_make_config(techniques=["raw", "pec"]), out_dir)

    def test_bad_shots_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="shots"):
            run_experiment(_make_config(shots=0), out_dir)

    def test_explicit_pauli_length_mismatch_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="pauli"):
            run_experiment(_make_config(pauli="ZZZ"), out_dir)  # circuits are 2q

    def test_explicit_pauli_is_used(self, patched_stack, out_dir: Path):
        df = run_experiment(_make_config(seeds=[0], pauli="ZI"), out_dir)
        assert df.iloc[0]["pauli"] == "ZI"

    def test_techniques_default_to_all(self, patched_stack, out_dir: Path):
        # Default = mitigation.TECHNIQUES, which includes the raw_plus
        # equal-budget baseline.
        config = _make_config(seeds=[0])
        del config["techniques"]
        df = run_experiment(config, out_dir)
        assert list(df.columns) == _expected_columns(DEFAULT_TECHNIQUES)

    def test_per_family_pauli_dict_is_resolved(self, patched_stack, out_dir: Path):
        df = run_experiment(
            _make_config(
                seeds=[0],
                pauli={"mirror_circuit": "X", "default": "auto"},
            ),
            out_dir,
        )
        # single-character spec repeats to the circuit width
        assert df.iloc[0]["pauli"] == "XX"

    def test_pauli_dict_unknown_family_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="unknown family"):
            run_experiment(_make_config(pauli={"not_a_family": "X"}), out_dir)

    def test_circuit_wider_than_backend_raises(self, patched_stack, out_dir: Path):
        config = _make_config()
        config["circuits"] = dict(config["circuits"], n_qubits=[6])
        # FakeManilaV2 has 5 qubits; 6-qubit circuits would be under-noised.
        with pytest.raises(ValueError, match="only 5 qubits"):
            run_experiment(config, out_dir)

    def test_bad_min_abs_ideal_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="min_abs_ideal"):
            run_experiment(_make_config(min_abs_ideal=1.5), out_dir)


class TestLowSignalScreening:
    def test_low_signal_units_skipped_and_logged(
        self, patched_stack, out_dir: Path
    ):
        # patched ideal is 0.5 for every unit -> a 0.6 threshold skips all.
        df = run_experiment(_make_config(min_abs_ideal=0.6), out_dir)
        assert len(df) == 0
        assert len(patched_stack) == 0  # no noisy work spent on lottery rows
        log = (out_dir / "skipped_low_signal.log").read_text(encoding="utf-8")
        assert "mirror_circuit_q2_d4_s0" in log
        assert "min_abs_ideal" in log

    def test_high_signal_units_survive_screening(
        self, patched_stack, out_dir: Path
    ):
        # ideal 0.5 >= 0.25 -> nothing skipped, no log file left behind.
        df = run_experiment(
            _make_config(seeds=[0], min_abs_ideal=0.25), out_dir
        )
        assert len(df) == 1
        assert not (out_dir / "skipped_low_signal.log").exists()


class TestScaledBackendNames:
    """Backend names may carry an '@x<scale>' noise-scale suffix (grammar
    of backends.parse_backend_name); validation checks the BASE name (and
    device width) while the full name flows into the 'backend' column as a
    distinct noise environment. Validation must use the SAME grammar as
    make_executor so a validated config cannot die mid-run on a bad name."""

    def test_scaled_name_accepted_and_recorded(self, patched_stack, out_dir: Path):
        df = run_experiment(
            _make_config(seeds=[0], backends=["FakeManilaV2@x2.0"]), out_dir
        )
        assert list(df["backend"]) == ["FakeManilaV2@x2.0"]

    def test_two_scales_are_distinct_units(self, patched_stack, out_dir: Path):
        df = run_experiment(
            _make_config(
                seeds=[0], backends=["FakeLagosV2", "FakeLagosV2@x1.5"]
            ),
            out_dir,
        )
        assert sorted(df["backend"]) == ["FakeLagosV2", "FakeLagosV2@x1.5"]

    def test_unknown_base_name_raises(self, patched_stack, out_dir: Path):
        with pytest.raises(ValueError, match="backend"):
            run_experiment(_make_config(backends=["FakeNopeV2@x2.0"]), out_dir)

    @pytest.mark.parametrize(
        "name",
        [
            "FakeManilaV2@2.0",     # missing the 'x'
            "FakeManilaV2@xzebra",  # non-numeric scale
            "FakeManilaV2@x0",      # zero
            "FakeManilaV2@x-1.5",   # negative
        ],
    )
    def test_bad_scale_suffix_fails_validation(
        self, patched_stack, out_dir: Path, name
    ):
        with pytest.raises(ValueError):
            run_experiment(_make_config(backends=[name]), out_dir)

    def test_width_check_uses_base_device(self, patched_stack, out_dir: Path):
        config = _make_config(backends=["FakeManilaV2@x2.0"])
        config["circuits"] = dict(config["circuits"], n_qubits=[6])
        with pytest.raises(ValueError, match="only 5 qubits"):
            run_experiment(config, out_dir)


class TestAggregatedCsv:
    """aggregated.csv: seed-averaged errors and winners (less label noise).

    Fixer pass 2026-07-21: the file also carries seed-mean feat_* columns
    (so qemsel.model can train on it directly — previously the
    seed-averaged labels existed but were unconsumable) and per-technique
    seed-coverage counts, and winners are restricted to maximum-coverage
    techniques (a 1-of-3-seed mean must not outrank 3-seed means).
    """

    AGG_COLS = (
        ["family", "n_qubits", "depth", "backend", "n_seeds"]
        + [f"feat_{n}" for n in FEATURE_NAMES]
        + [c for t in TECHNIQUES for c in (f"{t}_mean_abs_error", f"{t}_n_seeds")]
        + ["best_technique", COST_AWARE_COLUMN]
    )

    def test_groups_across_seeds_and_means(self, patched_stack, out_dir: Path):
        run_experiment(_make_config(), out_dir)  # 4 seeds x 1 backend
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert list(agg.columns) == self.AGG_COLS
        assert len(agg) == 1
        row = agg.iloc[0]
        assert row["family"] == "mirror_circuit"
        assert row["n_qubits"] == 2 and row["depth"] == 4
        assert row["backend"] == "FakeManilaV2"
        assert row["n_seeds"] == 4
        for tech in TECHNIQUES:
            # constant per-tech error -> mean equals it exactly
            assert row[f"{tech}_mean_abs_error"] == pytest.approx(ERR[tech])
            assert row[f"{tech}_n_seeds"] == 4  # full coverage everywhere
        assert row["best_technique"] == "zne"
        assert row[COST_AWARE_COLUMN] == "zne"

    def test_feat_columns_are_seed_means(self, patched_stack, out_dir: Path):
        """aggregated.csv carries feat_* columns equal to the per-seed
        means, making it directly trainable by qemsel.model."""
        df = run_experiment(_make_config(), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        row = agg.iloc[0]
        for name in FEATURE_NAMES:
            col = f"feat_{name}"
            assert row[col] == pytest.approx(float(df[col].mean()))

    def test_winner_recomputed_from_means_not_per_seed_votes(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Per-seed winners flip (seed 0 -> zne, seeds 1,2 -> cdr) but the
        # MEAN error makes cdr the aggregate winner: zne (0.02+0.4+0.4)/3
        # = 0.273 vs cdr 0.10. The aggregate label must come from means.
        def seed_dependent(name, circuit, pauli, executor, backend, shots, seed):
            errs = {
                "raw": 0.50,
                "zne": 0.02 if seed == 0 else 0.40,
                "cdr": 0.10,
                "rem": 0.60,
            }
            return IDEAL_VALUE + errs[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", seed_dependent)
        df = run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        assert set(df["best_technique"]) == {"zne", "cdr"}  # per-seed flips
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert len(agg) == 1
        assert agg.iloc[0]["best_technique"] == "cdr"
        assert agg.iloc[0]["zne_mean_abs_error"] == pytest.approx(0.82 / 3)

    def test_partial_nan_mean_skips_failed_seeds(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def flaky(name, circuit, pauli, executor, backend, shots, seed):
            if name == "cdr" and seed == 0:
                raise RuntimeError("cdr refused on seed 0")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", flaky)
        run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        # mean over the 2 non-NaN seeds only; coverage recorded honestly
        assert agg.iloc[0]["cdr_mean_abs_error"] == pytest.approx(ERR["cdr"])
        assert agg.iloc[0]["cdr_n_seeds"] == 2
        assert agg.iloc[0]["n_seeds"] == 3

    def test_all_nan_technique_excluded_from_winner(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def no_zne(name, circuit, pauli, executor, backend, shots, seed):
            if name == "zne":
                raise RuntimeError("zne always fails")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", no_zne)
        run_experiment(_make_config(seeds=[0, 1]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        row = agg.iloc[0]
        assert math.isnan(row["zne_mean_abs_error"])
        assert row["zne_n_seeds"] == 0
        assert row["best_technique"] == "cdr"  # next-best mean (0.10)

    def test_partial_coverage_technique_cannot_win(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Coverage rule (fixer 2026-07-21): cdr has the smallest MEAN but
        only 2 of 3 seeds — a partial-coverage mean competing against
        full-coverage means is the exact artifact observed live (cdr
        'winning' ghz_plus aggregates from its single non-refused seed),
        so cdr must NOT win; the best full-coverage technique (zne) must."""

        def cdr_partial(name, circuit, pauli, executor, backend, shots, seed):
            if name == "cdr":
                if seed == 0:
                    raise RuntimeError("cdr refused on seed 0")
                return IDEAL_VALUE + 0.001  # best mean, partial coverage
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", cdr_partial)
        run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        row = agg.iloc[0]
        assert row["cdr_mean_abs_error"] == pytest.approx(0.001)
        assert row["cdr_n_seeds"] == 2  # recorded, but ineligible
        assert row["best_technique"] == "zne"
        assert row[COST_AWARE_COLUMN] == "zne"

    def test_winner_falls_back_to_max_coverage_when_no_full_group(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When EVERY technique misses a seed (all-NaN row on seed 0), the
        maximum coverage is 2/3 and techniques at that coverage compete
        normally — the group is not thrown away."""

        def seed0_dies(name, circuit, pauli, executor, backend, shots, seed):
            if seed == 0:
                raise RuntimeError("everything fails on seed 0")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", seed0_dies)
        run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        row = agg.iloc[0]
        assert row["n_seeds"] == 3
        for tech in TECHNIQUES:
            assert row[f"{tech}_n_seeds"] == 2
        assert row["best_technique"] == "zne"

    def test_aggregated_csv_is_trainable(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The seed-averaged-label WIRING regression (fixer 2026-07-21):
        qemsel.model.train_and_eval must accept aggregated.csv as-is.
        Before the fix the file had no feat_* columns and training raised
        ValueError, so every headline number silently came from the noisier
        per-seed labels."""
        from qemsel.model import train_and_eval

        def backend_dependent(name, circuit, pauli, executor, backend, shots, seed):
            # zne wins on Manila, cdr on Lagos -> 2 label classes
            errs = dict(ERR)
            if backend.startswith("FakeLagosV2"):
                errs["cdr"], errs["zne"] = 0.005, 0.35
            return IDEAL_VALUE + errs[name]

        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique", backend_dependent
        )
        config = _make_config(seeds=[0, 1, 2])
        config["backends"] = ["FakeManilaV2", "FakeLagosV2"]
        run_experiment(config, out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert sorted(agg["best_technique"]) == ["cdr", "zne"]

        model_dir = out_dir / "model_from_agg"
        metrics = train_and_eval(agg, model_dir)  # must NOT raise
        assert metrics["n_samples"] == 2
        assert sorted(metrics["labels"]) == ["cdr", "zne"]
        assert (model_dir / "model.joblib").exists()

    def test_one_row_per_configuration(self, patched_stack, out_dir: Path):
        config = _make_config(seeds=[0, 1])
        config["circuits"] = dict(config["circuits"], depths=[4, 8])
        run_experiment(config, out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert len(agg) == 2  # one aggregate row per depth
        assert sorted(agg["depth"]) == [4, 8]
        assert (agg["n_seeds"] == 2).all()

    def test_rewritten_on_resume_with_more_seeds(
        self, patched_stack, out_dir: Path
    ):
        run_experiment(_make_config(seeds=[0, 1]), out_dir)
        agg1 = pd.read_csv(out_dir / "aggregated.csv")
        assert agg1.iloc[0]["n_seeds"] == 2
        run_experiment(_make_config(seeds=[0, 1, 2, 3]), out_dir)
        agg2 = pd.read_csv(out_dir / "aggregated.csv")
        assert len(agg2) == 1
        assert agg2.iloc[0]["n_seeds"] == 4

    def test_empty_run_writes_header_only(self, patched_stack, out_dir: Path):
        # min_abs_ideal 0.6 > patched ideal 0.5 screens every unit out.
        run_experiment(_make_config(min_abs_ideal=0.6), out_dir)
        agg = pd.read_csv(out_dir / "aggregated.csv")
        assert list(agg.columns) == self.AGG_COLS
        assert len(agg) == 0


class TestHardwareBudgetAbort:
    """HardwareBudgetExceededError anywhere in a technique's exception
    chain must abort the WHOLE sweep cleanly: completed rows preserved, the
    incomplete unit dropped for resume, no exception propagated, no
    further units attempted."""

    @staticmethod
    def _budget_error():
        from qemsel.hardware import HardwareBudgetExceededError

        return HardwareBudgetExceededError("qpu_seconds_cap exhausted")

    def _abort_on(self, target_seed: int, target_tech: str, calls: list):
        outer = self

        def fake_apply(name, circuit, pauli, executor, backend, shots, seed):
            calls.append((name, seed))
            if seed == target_seed and name == target_tech:
                raise outer._budget_error()
            return IDEAL_VALUE + ERR[name]

        return fake_apply

    def test_abort_preserves_completed_and_drops_partial(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list = []
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique", self._abort_on(1, "zne", calls)
        )
        df = run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        # unit seed=0 completed; unit seed=1 aborted mid-way; seed=2 never ran
        assert list(df["seed"]) == [0]
        on_disk = pd.read_csv(out_dir / "results.csv")
        assert len(on_disk) == 1
        assert calls == [
            ("raw", 0), ("zne", 0), ("cdr", 0), ("rem", 0),
            ("raw", 1), ("zne", 1),
        ]
        log = (out_dir / "errors.log").read_text(encoding="utf-8")
        assert "SWEEP ABORTED" in log
        assert "mirror_circuit_q2_d4_s1,FakeManilaV2,zne" in log

    def test_budget_error_wrapped_in_mitigation_error_still_aborts(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # mitigation.apply_technique wraps internal failures in
        # MitigationError with the original as __cause__ — the abort must
        # see through the wrapper.
        from qemsel.mitigation import MitigationError

        def fake_apply(name, circuit, pauli, executor, backend, shots, seed):
            if seed == 1:
                raise MitigationError(
                    name, "budget gone"
                ) from self._budget_error()
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", fake_apply)
        df = run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        assert list(df["seed"]) == [0]
        assert "SWEEP ABORTED" in (out_dir / "errors.log").read_text(
            encoding="utf-8"
        )

    def test_ordinary_failures_still_do_not_abort(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Regression guard: a plain MitigationError (no budget cause) must
        # keep the NaN-and-continue contract.
        from qemsel.mitigation import MitigationError

        def fake_apply(name, circuit, pauli, executor, backend, shots, seed):
            if name == "zne":
                raise MitigationError("zne", "ordinary failure")
            return IDEAL_VALUE + ERR[name]

        monkeypatch.setattr("qemsel.mitigation.apply_technique", fake_apply)
        df = run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert len(df) == 2
        assert df["zne_value"].isna().all()

    def test_resume_after_abort_completes_the_sweep(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list = []
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique", self._abort_on(1, "raw", calls)
        )
        run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        # budget restored: plain fake now succeeds everywhere
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique",
            lambda name, *a, **k: IDEAL_VALUE + ERR[name],
        )
        df = run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        assert sorted(df["seed"]) == [0, 1, 2]
        on_disk = pd.read_csv(out_dir / "results.csv")
        assert len(on_disk) == 3
        assert not on_disk.duplicated(subset=["circuit_id", "backend"]).any()


class TestExecutorClose:
    """Executors exposing close() (real hardware's shared Batch) must be
    closed once per unit, in a finally — also on failure and budget abort."""

    @pytest.fixture()
    def closing_executors(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        record = {"made": 0, "closed": 0}

        def make(backend_name, shots, seed):
            record["made"] += 1

            def _exec(circuit, pauli):
                return IDEAL_VALUE

            def _close():
                record["closed"] += 1

            _exec.close = _close
            return _exec

        monkeypatch.setattr("qemsel.backends.make_executor", make)
        return record

    def test_closed_once_per_unit(
        self, patched_stack, closing_executors, out_dir: Path
    ):
        run_experiment(_make_config(seeds=[0, 1, 2]), out_dir)
        assert closing_executors["made"] == 3
        assert closing_executors["closed"] == 3

    def test_closed_even_when_all_techniques_fail(
        self, patched_stack, closing_executors, out_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "qemsel.mitigation.apply_technique",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        run_experiment(_make_config(seeds=[0]), out_dir)
        assert closing_executors["closed"] == 1

    def test_closed_on_budget_abort(
        self, patched_stack, closing_executors, out_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from qemsel.hardware import HardwareBudgetExceededError

        def fake_apply(name, *a, **k):
            raise HardwareBudgetExceededError("cap")

        monkeypatch.setattr("qemsel.mitigation.apply_technique", fake_apply)
        run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert closing_executors["made"] == 1  # aborted on the first unit
        assert closing_executors["closed"] == 1

    def test_close_failure_is_swallowed(
        self, patched_stack, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def make(backend_name, shots, seed):
            def _exec(circuit, pauli):
                return IDEAL_VALUE

            def _close():
                raise RuntimeError("close exploded")

            _exec.close = _close
            return _exec

        monkeypatch.setattr("qemsel.backends.make_executor", make)
        df = run_experiment(_make_config(seeds=[0, 1]), out_dir)
        assert len(df) == 2  # run unharmed

    def test_executor_without_close_is_fine(
        self, patched_stack, out_dir: Path
    ):
        # patched_stack's make_executor returns a bare lambda (no close).
        df = run_experiment(_make_config(seeds=[0]), out_dir)
        assert len(df) == 1


class TestCliScript:
    def test_script_main_runs_config(
        self, patched_stack, out_dir: Path, tmp_path: Path
    ):
        import importlib.util
        import yaml

        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
        )
        spec = importlib.util.spec_from_file_location("run_experiment_cli", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(_make_config(seeds=[0])), encoding="utf-8"
        )
        rc = mod.main(["--config", str(config_path), "--out", str(out_dir)])
        assert rc == 0
        assert (out_dir / "results.csv").exists()

    def test_script_missing_config_errors(self, tmp_path: Path):
        import importlib.util

        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
        )
        spec = importlib.util.spec_from_file_location("run_experiment_cli2", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main(["--config", str(tmp_path / "nope.yaml")])
        assert rc == 2

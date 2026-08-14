"""Tests for qemsel.recommend + qemsel.report (builder-recommend).

Standalone by design: uses conftest fixtures (tiny_results_df, tiny_circuit,
out_dir) and a locally trained sklearn bundle. qemsel.features.extract_features
is monkeypatched for recommend() happy-path tests so nothing here depends on
other builders' unimplemented modules.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import qemsel.recommend as recommend_mod
from qemsel.recommend import recommend
from qemsel.report import generate_report

_FEATURE_NAMES = [
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
_FEAT_COLS = ["feat_" + n for n in _FEATURE_NAMES]
_EXPECTED_PNGS = [
    "error_by_technique.png",
    "win_rate.png",
    "confusion_matrix.png",
    "feature_importances.png",
]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fake_extract_features(circuit, backend_name):  # noqa: ANN001 - test stub
    """Stand-in for qemsel.features.extract_features (still a stub in
    parallel builds); returns a deterministic full feature dict."""
    return {name: float(i + 1) for i, name in enumerate(_FEATURE_NAMES)}


@pytest.fixture()
def fake_metrics(tiny_results_df: pd.DataFrame) -> dict:
    """Metrics dict matching the exact train_and_eval return schema."""
    labels = sorted(tiny_results_df["best_technique"].unique())
    k = len(labels)
    cm = (np.eye(k, dtype=int) * 3 + 1).tolist()
    rng = np.random.default_rng(0)
    raw_imp = rng.uniform(0.01, 1.0, size=len(_FEAT_COLS))
    importances = {
        c: float(v / raw_imp.sum()) for c, v in zip(_FEAT_COLS, raw_imp)
    }
    return {
        "best_model_name": "RandomForestClassifier",
        "accuracy": 0.75,
        "macro_f1": 0.72,
        "baseline_accuracy": 0.25,
        "labels": [str(label) for label in labels],
        "confusion_matrix": cm,
        "feature_importances": importances,
        "per_model": {
            "RandomForestClassifier": {"accuracy": 0.75, "macro_f1": 0.72},
            "GradientBoostingClassifier": {"accuracy": 0.69, "macro_f1": 0.66},
        },
        "n_samples": int(len(tiny_results_df)),
        "cv_folds": 4,
    }


@pytest.fixture()
def rich_metrics(fake_metrics: dict) -> dict:
    """Metrics dict with ALL research-pass keys (2026-07-21): grouped-CV
    provenance, dropped classes, per-fold stds, LOFO and LOBO blocks."""
    m = dict(fake_metrics)
    m["best_model_name"] = "random_forest"
    m["per_model"] = {
        "random_forest": {"accuracy": 0.75, "accuracy_std": 0.06, "macro_f1": 0.72},
        "gradient_boosting": {"accuracy": 0.69, "accuracy_std": 0.09, "macro_f1": 0.66},
        "dummy_majority": {"accuracy": 0.4, "accuracy_std": 0.02, "macro_f1": 0.2},
    }
    m["feature_importances_note"] = "held-out (mean over CV folds)"
    m["cv_grouping"] = "stratified_group"
    m["cv_n_samples"] = 15
    m["dropped_classes"] = ["zne"]
    m["label_column"] = "best_technique"
    m["lofo"] = {
        "accuracy": 0.7,
        "macro_f1": 0.6,
        "per_family_accuracy": {"ghz_plus": 0.8, "layered_random": 0.6},
        "per_family_macro_f1": {"ghz_plus": 0.7, "layered_random": 0.5},
        "n_families": 2,
    }
    m["lobo"] = {
        "accuracy": 0.65,
        "macro_f1": 0.55,
        "per_backend_accuracy": {"FakeManilaV2": 0.7, "FakeLagosV2@x2.0": 0.6},
        "per_backend_macro_f1": {"FakeManilaV2": 0.62, "FakeLagosV2@x2.0": 0.48},
        "n_backends": 2,
    }
    m["lodo"] = {
        "accuracy": 0.6,
        "macro_f1": 0.5,
        "per_device_accuracy": {"FakeManilaV2": 0.66, "FakeLagosV2": 0.54},
        "per_device_macro_f1": {"FakeManilaV2": 0.58, "FakeLagosV2": 0.42},
        "n_devices": 2,
    }
    return m


@pytest.fixture()
def cost_aware_metrics_fixture(rich_metrics: dict) -> dict:
    """A second metrics dict as trained on best_technique_cost_aware."""
    m = json.loads(json.dumps(rich_metrics))  # deep copy
    m["label_column"] = "best_technique_cost_aware"
    m["accuracy"] = 0.68
    m["macro_f1"] = 0.64
    m["labels"] = ["cdr", "raw", "rem", "zne"]
    m["dropped_classes"] = []
    return m


def _multi_scale_df(tiny_results_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the tiny df with noise-scaled copies of itself so the
    backend column carries '@x<scale>' suffixes at 3 distinct scales."""
    frames = [tiny_results_df]
    for scale, err_mult in (("1.5", 1.5), ("2.0", 2.5)):
        scaled = tiny_results_df.copy()
        scaled["backend"] = scaled["backend"].astype(str) + f"@x{scale}"
        for tech in ["raw", "zne", "cdr", "rem"]:
            scaled[f"{tech}_abs_error"] = scaled[f"{tech}_abs_error"] * err_mult
        frames.append(scaled)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def model_bundle_path(tiny_results_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Real trained sklearn model saved as the exact bundle shape."""
    x = tiny_results_df[_FEAT_COLS]
    y = tiny_results_df["best_technique"]
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    bundle = {
        "model": clf,
        "feature_names": _FEAT_COLS,
        "classes": sorted(y.unique()),
        "model_name": "RandomForestClassifier",
        "qemsel_version": "0.1.0",
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path


def _load_script(name: str):
    """Import a scripts/<name>.py file as a throwaway module."""
    script_path = _PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_qemsel_script_{name}", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# report.generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_returns_report_md_path(self, tiny_results_df, fake_metrics, out_dir):
        path = generate_report(tiny_results_df, fake_metrics, out_dir)
        assert path == out_dir / "report.md"
        assert path.exists()

    def test_report_contains_expected_section_headers(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        path = generate_report(tiny_results_df, fake_metrics, out_dir)
        text = path.read_text(encoding="utf-8")
        for header in [
            "## 1. Overview",
            "## 2. Technique comparison",
            "## 3. Cost-normalized view",
            "## 4. Win rates",
            "## 5. Noise-scale sweep",
            "## 6. Model evaluation",
            "## 7. Reproducibility",
        ]:
            assert header in text, f"missing section header: {header}"

    def test_report_mentions_key_content(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        # winner-count tables per family and backend
        assert "Winner counts per circuit family" in text
        assert "Winner counts per backend" in text
        # model metrics vs baseline
        assert "RandomForestClassifier" in text
        assert "0.25" in text  # baseline accuracy
        # all four figures referenced by relative filename
        for png in _EXPECTED_PNGS:
            assert png in text
        # every technique appears
        for tech in ["raw", "zne", "cdr", "rem"]:
            assert tech in text

    def test_pngs_created_and_nontrivial(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        generate_report(tiny_results_df, fake_metrics, out_dir)
        for png in _EXPECTED_PNGS:
            png_path = out_dir / png
            assert png_path.exists(), f"missing figure: {png}"
            assert png_path.stat().st_size > 1024, f"figure too small: {png}"

    def test_creates_missing_out_dir(self, tiny_results_df, fake_metrics, tmp_path):
        nested = tmp_path / "a" / "b" / "report_out"
        path = generate_report(tiny_results_df, fake_metrics, nested)
        assert path.exists()

    def test_handles_nan_abs_errors(self, tiny_results_df, fake_metrics, out_dir):
        df = tiny_results_df.copy()
        df.loc[df.index[:4], "cdr_abs_error"] = np.nan
        path = generate_report(df, fake_metrics, out_dir)
        assert path.exists()
        assert "NaN" in path.read_text(encoding="utf-8")

    def test_raises_on_missing_df_columns(self, tiny_results_df, fake_metrics, out_dir):
        bad = tiny_results_df.drop(columns=["best_technique"])
        with pytest.raises(ValueError, match="best_technique"):
            generate_report(bad, fake_metrics, out_dir)

    def test_raises_on_no_technique_columns(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        bad = tiny_results_df[
            [c for c in tiny_results_df.columns if not c.endswith("_abs_error")]
        ]
        with pytest.raises(ValueError, match="abs_error"):
            generate_report(bad, fake_metrics, out_dir)

    def test_raises_on_missing_metrics_keys(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        bad = dict(fake_metrics)
        del bad["confusion_matrix"]
        with pytest.raises(ValueError, match="confusion_matrix"):
            generate_report(tiny_results_df, bad, out_dir)

    def test_raises_on_empty_df(self, tiny_results_df, fake_metrics, out_dir):
        with pytest.raises(ValueError):
            generate_report(tiny_results_df.iloc[0:0], fake_metrics, out_dir)


# ---------------------------------------------------------------------------
# report: research-pass features (noise sweep, LOFO/LOBO, dual labels, ...)
# ---------------------------------------------------------------------------


class TestReportResearchPass:
    # ---- noise-scale sweep (the money plot) -------------------------------

    def test_winner_vs_noise_png_written_for_multi_scale(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        df = _multi_scale_df(tiny_results_df)
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        png = out_dir / "winner_vs_noise.png"
        assert png.exists()
        assert png.stat().st_size > 1024
        assert "winner_vs_noise.png" in text
        # all three parsed scales appear in the sweep tables
        for scale_label in ["x1", "x1.5", "x2"]:
            assert scale_label in text, f"missing scale {scale_label}"
        assert "Win rate per technique vs noise scale" in text
        assert "Mean abs error per technique vs noise scale" in text

    def test_no_noise_png_for_single_scale(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert not (out_dir / "winner_vs_noise.png").exists()
        assert "## 5. Noise-scale sweep" in text  # section always present
        assert "not applicable" in text

    def test_noise_scale_parsing_helper(self):
        from qemsel.report import _parse_backend

        assert _parse_backend("FakeManilaV2") == ("FakeManilaV2", 1.0)
        assert _parse_backend("FakeManilaV2@x1.5") == ("FakeManilaV2", 1.5)
        assert _parse_backend("FakeSherbrooke@x3") == ("FakeSherbrooke", 3.0)
        # malformed suffixes are NOT parsed as scales
        assert _parse_backend("Fake@xTwo") == ("Fake@xTwo", 1.0)

    # ---- LOFO / LOBO / dropped classes / accuracy ± std -------------------

    def test_lofo_and_lobo_tables_rendered(
        self, tiny_results_df, rich_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, rich_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "leave-one-family-out" in text
        assert "leave-one-backend-out" in text
        assert "Held-out family" in text
        assert "Held-out backend" in text
        # per-group rows with accuracy AND macro-F1
        assert "FakeLagosV2@x2.0" in text
        assert "ghz_plus" in text
        # LODO is the new-device headline; LOBO is relabeled interpolation
        # (fixer 2026-07-21: scale-sibling folds are NOT new environments)
        assert "leave-one-device-out" in text
        assert "Held-out device" in text
        assert "Noise-level interpolation (leave-one-backend-out)" in text
        assert (
            "Generalization to a NEW noise environment (leave-one-device-out)"
            in text
        )

    def test_legacy_metrics_without_lodo_still_renders(
        self, tiny_results_df, rich_metrics, out_dir
    ):
        m = dict(rich_metrics)
        del m["lodo"]
        text = generate_report(tiny_results_df, m, out_dir).read_text(
            encoding="utf-8"
        )
        assert "leave-one-backend-out" in text
        assert "Held-out device" not in text

    def test_dropped_classes_note_rendered(
        self, tiny_results_df, rich_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, rich_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "classes dropped from CV:** zne" in text
        assert "15 in CV after class drops" in text

    def test_accuracy_plus_minus_std_rendered(
        self, tiny_results_df, rich_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, rich_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "0.75 ± 0.06" in text  # best model accuracy ± fold std

    # ---- both winner labels side by side ----------------------------------

    def test_cost_aware_side_by_side_via_kwarg(
        self, tiny_results_df, rich_metrics, cost_aware_metrics_fixture, out_dir
    ):
        text = generate_report(
            tiny_results_df,
            rich_metrics,
            out_dir,
            cost_aware_metrics=cost_aware_metrics_fixture,
        ).read_text(encoding="utf-8")
        assert "Both winner labels side by side" in text
        assert "best_technique_cost_aware" in text
        assert "model_cost_aware.joblib" in text

    def test_cost_aware_side_by_side_via_embedded_key(
        self, tiny_results_df, rich_metrics, cost_aware_metrics_fixture, out_dir
    ):
        # train_and_eval_all embeds the second metrics dict into
        # metrics.json under 'cost_aware'; the report must auto-detect it.
        combined = dict(rich_metrics)
        combined["cost_aware"] = cost_aware_metrics_fixture
        text = generate_report(tiny_results_df, combined, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Both winner labels side by side" in text

    def test_no_side_by_side_without_cost_metrics(
        self, tiny_results_df, rich_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, rich_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Both winner labels side by side" not in text

    # ---- schema compatibility ---------------------------------------------

    def test_raw_plus_technique_detected(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        df = tiny_results_df.copy()
        df["raw_plus_value"] = df["raw_value"]
        df["raw_plus_abs_error"] = df["raw_abs_error"] * 0.8
        df["raw_plus_shots"] = 11 * 1024
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "raw_plus" in text
        assert "equal-budget baseline" in text

    def test_aggregated_schema_renders(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        # aggregated.csv: n_seeds column present, no seed column
        df = tiny_results_df.drop(columns=["seed"]).copy()
        df["n_seeds"] = 3
        path = generate_report(df, fake_metrics, out_dir)
        text = path.read_text(encoding="utf-8")
        assert "SEED-AVERAGED" in text
        for png in _EXPECTED_PNGS:
            assert (out_dir / png).exists()

    def test_true_aggregated_schema_with_mean_columns_renders(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        """The REAL aggregated.csv stores '<tech>_mean_abs_error' (no
        '<tech>_abs_error', no value/shots columns). Detection must map
        those to the plain technique names — before the fixer pass
        2026-07-21 'cdr_mean' etc. were detected as techniques and every
        winner row counted as invalid."""
        techs = ["raw", "zne", "cdr", "rem"]
        df = tiny_results_df.drop(
            columns=["seed"]
            + [f"{t}_value" for t in techs]
            + [f"{t}_shots" for t in techs]
        ).rename(columns={f"{t}_abs_error": f"{t}_mean_abs_error" for t in techs})
        df["n_seeds"] = 3
        for t in techs:
            df[f"{t}_n_seeds"] = 3
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "raw_mean" not in text  # no phantom '<tech>_mean' techniques
        assert "Rows counted: 16 of 16" in text  # winners are valid again
        assert "SEED-AVERAGED" in text

    # ---- realized noise levels + device composition (fixer 2026-07-21) ----

    def test_realized_noise_table_and_caveats_for_multi_scale(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        df = _multi_scale_df(tiny_results_df)
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Per-scale device composition and realized noise levels" in text
        assert "Nominal vs realized scale" in text
        assert "Scale 1.0 differs in KIND" in text
        # every device contributes every scale here -> no confounding warning
        assert "unequal device composition" not in text

    def test_unequal_scale_composition_warning(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        df = _multi_scale_df(tiny_results_df)
        df = df[df["backend"] != "FakeLagosV2@x2.0"]  # x2.0 loses a device
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "unequal device composition" in text

    # ---- circuit-selection conditioning disclosure (fixer 2026-07-21) -----

    def test_conditioning_disclosure_from_run_config(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        run_config = {
            "min_abs_ideal": 0.25,
            "circuits": {"min_abs_ideal": 0.25},
        }
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, run_config=run_config
        ).read_text(encoding="utf-8")
        assert "Circuit-selection conditioning" in text
        assert "|<Z^n>| >= 0.25" in text

    def test_no_conditioning_disclosure_by_default(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Circuit-selection conditioning" not in text

    def test_conditioning_disclosure_from_bumped_seeds(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        """Even without the config, bumped rejection-sampling seeds
        (seed + k * SUB_SEED_STRIDE) prove the suite was conditioned."""
        from qemsel.circuits import SUB_SEED_STRIDE

        df = tiny_results_df.copy()
        df.loc[df.index[0], "seed"] = SUB_SEED_STRIDE + 1
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Circuit-selection conditioning" in text

    def test_legacy_metrics_dict_still_renders(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        # fake_metrics deliberately lacks every research-pass key (lofo,
        # lobo, lodo, dropped_classes, cv_n_samples, cv_grouping,
        # accuracy_std): pre-upgrade metrics.json files must keep working.
        for key in ["lofo", "lobo", "lodo", "dropped_classes", "cv_n_samples"]:
            assert key not in fake_metrics
        path = generate_report(tiny_results_df, fake_metrics, out_dir)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "## 6. Model evaluation" in text


# ---------------------------------------------------------------------------
# recommend.recommend
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_missing_model_file_raises_file_not_found(self, tmp_path, tiny_circuit):
        missing = tmp_path / "does_not_exist.joblib"
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            recommend(missing, tiny_circuit, "FakeManilaV2")

    def test_malformed_bundle_raises_value_error(self, tmp_path, tiny_circuit):
        bad_path = tmp_path / "bad.joblib"
        joblib.dump({"not_a_model": 1}, bad_path)
        with pytest.raises(ValueError, match="missing keys"):
            recommend(bad_path, tiny_circuit, "FakeManilaV2")

    def test_non_dict_bundle_raises_value_error(self, tmp_path, tiny_circuit):
        bad_path = tmp_path / "bare.joblib"
        joblib.dump([1, 2, 3], bad_path)
        with pytest.raises(ValueError, match="expected a dict"):
            recommend(bad_path, tiny_circuit, "FakeManilaV2")

    def test_happy_path_returns_exact_keys(
        self, monkeypatch, model_bundle_path, tiny_circuit
    ):
        monkeypatch.setattr(
            recommend_mod, "extract_features", _fake_extract_features
        )
        result = recommend(model_bundle_path, tiny_circuit, "FakeManilaV2")
        assert set(result.keys()) == {"technique", "probabilities", "features"}
        assert result["technique"] in {"raw", "zne", "cdr", "rem"}
        probs = result["probabilities"]
        assert set(probs.keys()) == {"raw", "zne", "cdr", "rem"}
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(isinstance(v, float) for v in probs.values())
        # argmax class is the recommended technique
        assert result["technique"] == max(probs, key=probs.get)
        # features are the extracted dict, all floats
        assert result["features"] == _fake_extract_features(None, None)

    def test_deterministic(self, monkeypatch, model_bundle_path, tiny_circuit):
        monkeypatch.setattr(
            recommend_mod, "extract_features", _fake_extract_features
        )
        r1 = recommend(model_bundle_path, tiny_circuit, "FakeManilaV2")
        r2 = recommend(model_bundle_path, tiny_circuit, "FakeManilaV2")
        assert r1 == r2

    def test_feature_mismatch_raises_value_error(
        self, monkeypatch, model_bundle_path, tiny_circuit
    ):
        def _partial_features(circuit, backend_name):
            feats = _fake_extract_features(circuit, backend_name)
            del feats["n_cnot"]
            return feats

        monkeypatch.setattr(recommend_mod, "extract_features", _partial_features)
        with pytest.raises(ValueError, match="feature mismatch"):
            recommend(model_bundle_path, tiny_circuit, "FakeManilaV2")


# ---------------------------------------------------------------------------
# CLI scripts
# ---------------------------------------------------------------------------


class TestScripts:
    def test_make_report_cli(self, tiny_results_df, fake_metrics, tmp_path, capsys):
        data_path = tmp_path / "results.csv"
        metrics_path = tmp_path / "metrics.json"
        out = tmp_path / "reportdir"
        tiny_results_df.to_csv(data_path, index=False)
        metrics_path.write_text(json.dumps(fake_metrics), encoding="utf-8")
        # run_meta.json sidecar (as run_experiment writes it): the CLI must
        # pass its config through so the report discloses the min_abs_ideal
        # circuit-selection conditioning (fixer 2026-07-21).
        (tmp_path / "run_meta.json").write_text(
            json.dumps({"config": {"min_abs_ideal": 0.25}}), encoding="utf-8"
        )

        make_report = _load_script("make_report")
        rc = make_report.main(
            [
                "--data",
                str(data_path),
                "--metrics",
                str(metrics_path),
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        assert (out / "report.md").exists()
        for png in _EXPECTED_PNGS:
            assert (out / png).exists()
        assert "report written" in capsys.readouterr().out
        report_text = (out / "report.md").read_text(encoding="utf-8")
        assert "Circuit-selection conditioning" in report_text

    def test_make_report_cli_missing_input_exits(self, tmp_path):
        make_report = _load_script("make_report")
        with pytest.raises(SystemExit):
            make_report.main(
                [
                    "--data",
                    str(tmp_path / "nope.csv"),
                    "--metrics",
                    str(tmp_path / "nope.json"),
                    "--out",
                    str(tmp_path / "out"),
                ]
            )

    def test_recommend_cli_qasm(
        self, monkeypatch, model_bundle_path, tmp_path, capsys
    ):
        monkeypatch.setattr(
            recommend_mod, "extract_features", _fake_extract_features
        )
        qasm_path = tmp_path / "bell.qasm"
        qasm_path.write_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q -> c;\n",
            encoding="utf-8",
        )
        recommend_script = _load_script("recommend")
        rc = recommend_script.main(
            [
                "--model",
                str(model_bundle_path),
                "--backend",
                "FakeManilaV2",
                "--qasm",
                str(qasm_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Recommended technique" in out
        assert '"probabilities"' in out

    def test_recommend_cli_requires_source(self, model_bundle_path):
        recommend_script = _load_script("recommend")
        with pytest.raises(SystemExit):
            recommend_script.main(
                ["--model", str(model_bundle_path), "--backend", "FakeManilaV2"]
            )

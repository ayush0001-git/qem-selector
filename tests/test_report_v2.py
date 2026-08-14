"""Tests for the V2 additions in qemsel.report + qemsel.recommend (builder B8).

Covers, without depending on other builders' unlanded modules:
* report.py section 8 "Statistical hygiene" rendered from a compute_stats
  dict fixture (win-share CIs for both label columns, permutation tests,
  Cliff's delta, Koester pass/flag table incl. bold FAIL);
* report.py section 9 "ZNE help-harm boundary overlay" rendered from an
  overlay dict fixture, including the plot-path-inside-out_dir guard;
* the byte-identical-V1 golden guarantee (both new kwargs default None);
* the dynamic 7-technique rendering;
* recommend.py V1-bundle regression (exact 3-key return, base_shots
  ignored) and the V2-bundle path (feature_version echo, abstain).

qemsel.stats / qemsel.boundary are NOT imported here — report consumes only
their OUTPUT dicts, which are built as literals below.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import qemsel.recommend as recommend_mod
from qemsel.features import FEATURE_NAMES, FEATURE_NAMES_V2
from qemsel.recommend import recommend
from qemsel.report import generate_report

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FEAT_COLS_V1 = ["feat_" + n for n in FEATURE_NAMES]
_FEAT_COLS_V2 = ["feat_" + n for n in FEATURE_NAMES_V2]
_NEW_TECHNIQUES = ["zne_fr", "cdr_ridge", "cdr_rf"]
_ALL_SEVEN = ["raw", "zne", "cdr", "rem", "zne_fr", "cdr_ridge", "cdr_rf"]


# ---------------------------------------------------------------------------
# local fixtures (this file is standalone; conftest supplies tiny_results_df,
# tiny_circuit, out_dir)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_metrics(tiny_results_df: pd.DataFrame) -> dict:
    """Minimal metrics dict matching the train_and_eval return schema."""
    labels = sorted(tiny_results_df["best_technique"].unique())
    k = len(labels)
    cm = (np.eye(k, dtype=int) * 3 + 1).tolist()
    importances = {c: 1.0 / len(_FEAT_COLS_V1) for c in _FEAT_COLS_V1}
    return {
        "best_model_name": "RandomForestClassifier",
        "accuracy": 0.75,
        "macro_f1": 0.72,
        "baseline_accuracy": 0.25,
        "labels": [str(x) for x in labels],
        "confusion_matrix": cm,
        "feature_importances": importances,
        "per_model": {
            "RandomForestClassifier": {"accuracy": 0.75, "macro_f1": 0.72},
        },
        "n_samples": int(len(tiny_results_df)),
        "cv_folds": 4,
    }


def _boot(estimate: float, lo: float, hi: float, tech: str, n: int = 16) -> dict:
    return {
        "estimate": estimate,
        "lo": lo,
        "hi": hi,
        "ci": 0.95,
        "n": n,
        "n_dropped_nan": 0,
        "n_boot": 1000,
        "seed": 0,
        "technique": tech,
    }


def _perm(mean_diff: float, p_value: float) -> dict:
    return {
        "mean_diff": mean_diff,
        "p_value": p_value,
        "n_pairs": 16,
        "n_dropped_nan": 0,
        "n_perm": 1000,
        "alternative": "two-sided",
        "seed": 0,
    }


@pytest.fixture()
def stats_results() -> dict:
    """A compute_stats.py stats.json dict (schema in qemsel.stats docstring)."""
    return {
        "win_share_ci": {
            "best_technique": {
                "raw": _boot(0.25, 0.10, 0.44, "raw"),
                "zne": _boot(0.25, 0.10, 0.44, "zne"),
                "cdr": _boot(0.25, 0.10, 0.44, "cdr"),
                "rem": _boot(0.25, 0.10, 0.44, "rem"),
            },
            "best_technique_cost_aware": {
                "raw": _boot(0.50, 0.30, 0.70, "raw"),
                "zne": _boot(0.20, 0.05, 0.40, "zne"),
            },
        },
        "paired_tests": {
            "raw_plus_vs_raw": _perm(-0.020, 0.031),
            "top2_cdr_vs_rem": _perm(0.005, 0.410),
        },
        "effect_sizes": {
            "raw_plus_vs_raw": -0.30,
            "top2_cdr_vs_rem": 0.05,
        },
        "checklist": {
            "schema": "per_seed",
            "n_rows": 16,
            "techniques": ["raw", "zne", "cdr", "rem"],
            "checks": {
                "overshoot_beyond_physical_max": {
                    "raw": 0, "zne": 1, "cdr": 0, "rem": 0
                },
                "error_beyond_physical_max": {
                    "raw": 0, "zne": 0, "cdr": 0, "rem": 0
                },
                "nan_rate": {"raw": 0.0, "zne": 0.0, "cdr": 0.0, "rem": 0.0},
                "label_argmin_consistent": {"n_checked": 16, "n_mismatch": 0},
                "winner_margin_below_k_sigma": {
                    "k_sigma": 2.0, "n_flagged": 2, "fraction": 0.125
                },
                "partial_coverage_winners": None,
            },
            "passed": True,
        },
        "n_rows": 16,
        "data_path": "results/run1/results.csv",
    }


def _boundary_overlay(out_dir: Path, *, plot_path: str | None = None) -> dict:
    """Overlay dict as returned by boundary.overlay_selector_vs_theory. The
    figure is written INSIDE out_dir unless a plot_path override is given."""
    if plot_path is None:
        png = out_dir / "boundary_overlay.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 256)
        plot_path = str(png)
    return {
        "agreement_pct": 82.5,
        "iou_help": 0.61,
        "n_points": 15,
        "selector_help_share": 0.40,
        "theory_help_share": 0.53,
        "eps_feature": "avg_2q_error",
        "zne_labels": ["zne", "zne_fr"],
        "plot_path": plot_path,
        "grid": [
            {
                "backend": "FakeManilaV2",
                "eps": 0.011,
                "shots": 1024,
                "selector_zne": False,
                "zne_vote_share": 0.2,
                "theory_regime": "harm",
                "delta_mse": -1e-4,
            }
        ],
    }


def _seven_tech_df(tiny_results_df: pd.DataFrame) -> pd.DataFrame:
    """tiny_results_df extended with the three additive V2 technique columns
    so the report must render all 7 techniques dynamically."""
    df = tiny_results_df.copy()
    rng = np.random.default_rng(7)
    for tech in _NEW_TECHNIQUES:
        mult = {"zne_fr": 1, "cdr_ridge": 11, "cdr_rf": 11}[tech]
        errs = rng.uniform(0.02, 0.25, size=len(df))
        df[f"{tech}_value"] = df["ideal"].to_numpy() + errs
        df[f"{tech}_abs_error"] = errs
        df[f"{tech}_shots"] = 1024 * mult
    return df


def _load_script(name: str):
    script_path = _PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_qemsel_script_{name}", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    """Deterministic predict_proba stub — returns a fixed probability row
    regardless of X, so abstain/argmax behaviour is fully controllable."""

    def __init__(self, classes: list[str], proba_row: list[float]):
        assert len(classes) == len(proba_row)
        self.classes_ = list(classes)
        self._row = np.asarray(proba_row, dtype=float)

    def predict_proba(self, X):  # noqa: N803 - sklearn convention
        n = len(X)
        return np.tile(self._row, (n, 1))


def _dump_bundle(path: Path, bundle: dict) -> Path:
    joblib.dump(bundle, path)
    return path


# ---------------------------------------------------------------------------
# report.py — byte-identical V1 golden guarantee
# ---------------------------------------------------------------------------


class TestReportV1ByteIdentical:
    def test_none_kwargs_byte_identical_to_positional_v1(
        self, tiny_results_df, fake_metrics, tmp_path
    ):
        """The new keyword-only args, defaulted/None, must not perturb a
        single byte of the V1 report (captured golden: the positional call
        is the pre-V2 signature)."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        p_v1 = generate_report(tiny_results_df, fake_metrics, a)
        p_none = generate_report(
            tiny_results_df,
            fake_metrics,
            b,
            stats_results=None,
            boundary_overlay=None,
        )
        assert p_v1.read_bytes() == p_none.read_bytes()

    def test_v1_report_has_no_section_8_or_9(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        text = generate_report(tiny_results_df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "## 8." not in text
        assert "## 9." not in text
        # ...and still ends at reproducibility.
        assert "## 7. Reproducibility" in text


# ---------------------------------------------------------------------------
# report.py — dynamic 7-technique rendering
# ---------------------------------------------------------------------------


class TestSevenTechniques:
    def test_all_seven_techniques_rendered(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        df = _seven_tech_df(tiny_results_df)
        text = generate_report(df, fake_metrics, out_dir).read_text(
            encoding="utf-8"
        )
        assert "Techniques compared (7)" in text
        for tech in _ALL_SEVEN:
            assert tech in text, f"technique missing from report: {tech}"
        # canonical ordering: zne_fr renders right after zne, cdr_ridge/cdr_rf
        # after cdr, all before rem, in the overview technique list.
        overview_line = next(
            ln for ln in text.splitlines() if "Techniques compared (7)" in ln
        )
        assert overview_line.index("zne_fr") < overview_line.index("cdr")
        assert overview_line.index("cdr_rf") < overview_line.index("rem")


# ---------------------------------------------------------------------------
# report.py — section 8 statistical hygiene
# ---------------------------------------------------------------------------


class TestSectionStats:
    def test_section_8_present_and_headers(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=stats_results
        ).read_text(encoding="utf-8")
        assert "## 8. Statistical hygiene" in text
        assert "Win-share bootstrap confidence intervals" in text
        assert "Paired permutation tests" in text
        assert "Effect sizes (Cliff's delta)" in text
        assert "Koester-Mauerer statistical checklist" in text

    def test_both_label_columns_rendered(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=stats_results
        ).read_text(encoding="utf-8")
        assert "`best_technique`" in text
        assert "`best_technique_cost_aware`" in text
        # a bootstrap CI bound appears
        assert "[0.1, 0.44]" in text

    def test_permutation_and_effect_rows(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=stats_results
        ).read_text(encoding="utf-8")
        # raw_plus_vs_raw humanized + p-value + top-2 humanized
        assert "raw_plus vs raw" in text
        assert "top-2: cdr vs rem" in text
        assert "0.031" in text  # a permutation p-value
        assert "-0.3" in text  # a Cliff's delta

    def test_checklist_pass_verdict(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=stats_results
        ).read_text(encoding="utf-8")
        assert "Overall verdict: **PASS**" in text
        assert "label_argmin_consistent" in text
        assert "winner_margin_below_k_sigma" in text

    def test_checklist_fail_is_bold_with_counts(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        failing = json.loads(json.dumps(stats_results))  # deep copy
        failing["checklist"]["passed"] = False
        failing["checklist"]["checks"]["label_argmin_consistent"] = {
            "n_checked": 16,
            "n_mismatch": 3,
        }
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=failing
        ).read_text(encoding="utf-8")
        assert "Overall verdict: **FAIL**" in text
        assert "**FLAG**" in text
        assert "n_mismatch = 3" in text

    def test_partial_coverage_flag_for_aggregated(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        agg = json.loads(json.dumps(stats_results))
        agg["checklist"]["schema"] = "aggregated"
        agg["checklist"]["passed"] = False
        agg["checklist"]["checks"]["partial_coverage_winners"] = 4
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=agg
        ).read_text(encoding="utf-8")
        assert "partial_coverage_winners" in text
        # the non-zero coverage count is flagged
        line = next(
            ln
            for ln in text.splitlines()
            if "partial_coverage_winners" in ln and "|" in ln
        )
        assert "**FLAG**" in line

    def test_partial_stats_dict_degrades_gracefully(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        """A dict missing several optional blocks must still render, not
        KeyError."""
        minimal = {"checklist": {"schema": "per_seed", "passed": True, "checks": {}}}
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, stats_results=minimal
        ).read_text(encoding="utf-8")
        assert "## 8. Statistical hygiene" in text


# ---------------------------------------------------------------------------
# report.py — section 9 boundary overlay
# ---------------------------------------------------------------------------


class TestSectionBoundary:
    def test_section_9_present_with_figure_and_metrics(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        overlay = _boundary_overlay(out_dir)
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, boundary_overlay=overlay
        ).read_text(encoding="utf-8")
        assert "## 9. ZNE help-harm boundary overlay" in text
        # figure embedded by out_dir-relative filename
        assert (
            "![Selector ZNE-refusal region vs analytic boundary]"
            "(boundary_overlay.png)" in text
        )
        # headline numbers
        assert "82.5%" in text
        assert "0.61" in text
        assert "arXiv:2605.08251" in text

    def test_mandatory_caveats_present(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        overlay = _boundary_overlay(out_dir)
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, boundary_overlay=overlay
        ).read_text(encoding="utf-8")
        assert "Realized, not nominal" in text  # realized-eps axis caveat
        assert "zne_fr" in text  # variant-alignment caveat
        assert "Simulation only" in text  # sim-only caveat

    def test_plot_path_outside_out_dir_raises(
        self, tiny_results_df, fake_metrics, out_dir, tmp_path
    ):
        stray = tmp_path / "elsewhere" / "boundary_overlay.png"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        overlay = _boundary_overlay(out_dir, plot_path=str(stray))
        with pytest.raises(ValueError, match="inside the report out_dir"):
            generate_report(
                tiny_results_df, fake_metrics, out_dir, boundary_overlay=overlay
            )

    def test_plot_path_missing_file_raises(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        overlay = _boundary_overlay(
            out_dir, plot_path=str(out_dir / "boundary_overlay.png")
        )
        # remove the file that _boundary_overlay would otherwise have created
        (out_dir / "boundary_overlay.png").unlink(missing_ok=True)
        with pytest.raises(ValueError, match="does not exist"):
            generate_report(
                tiny_results_df, fake_metrics, out_dir, boundary_overlay=overlay
            )

    def test_relative_plot_path_inside_out_dir_ok(
        self, tiny_results_df, fake_metrics, out_dir
    ):
        # a bare filename is interpreted relative to out_dir
        (out_dir / "boundary_overlay.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"0" * 64
        )
        overlay = _boundary_overlay(out_dir, plot_path="boundary_overlay.png")
        text = generate_report(
            tiny_results_df, fake_metrics, out_dir, boundary_overlay=overlay
        ).read_text(encoding="utf-8")
        assert "## 9. ZNE help-harm boundary overlay" in text

    def test_both_sections_together(
        self, tiny_results_df, fake_metrics, out_dir, stats_results
    ):
        overlay = _boundary_overlay(out_dir)
        text = generate_report(
            tiny_results_df,
            fake_metrics,
            out_dir,
            stats_results=stats_results,
            boundary_overlay=overlay,
        ).read_text(encoding="utf-8")
        assert "## 8. Statistical hygiene" in text
        assert "## 9. ZNE help-harm boundary overlay" in text
        # section order: 8 before 9
        assert text.index("## 8.") < text.index("## 9.")


# ---------------------------------------------------------------------------
# recommend.py — V1 bundle regression (byte-for-byte the old 3-key return)
# ---------------------------------------------------------------------------


class TestRecommendV1Regression:
    def _v1_bundle_path(self, tmp_path: Path) -> Path:
        bundle = {
            "model": _FakeModel(["cdr", "raw", "rem", "zne"], [0.6, 0.2, 0.1, 0.1]),
            "feature_names": _FEAT_COLS_V1,
            "classes": ["cdr", "raw", "rem", "zne"],
            "model_name": "fake",
            "qemsel_version": "0.1.0",
        }
        return _dump_bundle(tmp_path / "v1.joblib", bundle)

    def test_v1_bundle_returns_exact_three_keys(self, tmp_path, tiny_circuit):
        path = self._v1_bundle_path(tmp_path)
        result = recommend(path, tiny_circuit, "FakeManilaV2")
        assert set(result.keys()) == {"technique", "probabilities", "features"}
        assert result["technique"] == "cdr"  # argmax of the fixed proba

    def test_v1_bundle_ignores_base_shots(self, tmp_path, tiny_circuit):
        path = self._v1_bundle_path(tmp_path)
        r_no = recommend(path, tiny_circuit, "FakeManilaV2")
        r_shots = recommend(path, tiny_circuit, "FakeManilaV2", base_shots=4096)
        # base_shots must be IGNORED for a V1 bundle: identical 3-key return.
        assert set(r_shots.keys()) == {"technique", "probabilities", "features"}
        assert r_no == r_shots


# ---------------------------------------------------------------------------
# recommend.py — V2 bundle path (feature_version echo + abstain)
# ---------------------------------------------------------------------------


def _fake_extract_v2(circuit, backend_name, *, version=1, base_shots=None):
    names = FEATURE_NAMES if version == 1 else FEATURE_NAMES_V2
    return {n: float(i + 1) for i, n in enumerate(names)}


class TestRecommendV2:
    def _v2_bundle(
        self,
        tmp_path: Path,
        *,
        feature_version: int,
        abstain_threshold=None,
        proba_row=(0.6, 0.2, 0.1, 0.1),
    ) -> Path:
        classes = ["cdr", "raw", "rem", "zne"]
        feat_cols = _FEAT_COLS_V1 if feature_version == 1 else _FEAT_COLS_V2
        bundle = {
            "model": _FakeModel(classes, list(proba_row)),
            "feature_names": feat_cols,
            "classes": classes,
            "model_name": "fake",
            "label_column": "best_technique",
            "qemsel_version": "0.1.0",
            "feature_version": feature_version,
            "calibrated": False,
            "abstain_threshold": abstain_threshold,
        }
        return _dump_bundle(tmp_path / f"v2_fv{feature_version}.joblib", bundle)

    def test_v2_fv1_returns_v2_keys(self, tmp_path, tiny_circuit):
        # feature_version 1 uses the real extract_features (v1 path works).
        path = self._v2_bundle(tmp_path, feature_version=1)
        result = recommend(path, tiny_circuit, "FakeManilaV2")
        assert set(result.keys()) == {
            "technique",
            "probabilities",
            "features",
            "abstained",
            "abstain_threshold",
            "feature_version",
        }
        assert result["feature_version"] == 1
        assert result["abstained"] is False
        assert result["abstain_threshold"] is None
        assert result["technique"] == "cdr"

    def test_v2_fv2_requires_base_shots(self, tmp_path, tiny_circuit):
        path = self._v2_bundle(tmp_path, feature_version=2)
        with pytest.raises(ValueError, match="base_shots"):
            recommend(path, tiny_circuit, "FakeManilaV2")

    def test_v2_fv2_with_base_shots(self, monkeypatch, tmp_path, tiny_circuit):
        monkeypatch.setattr(recommend_mod, "extract_features", _fake_extract_v2)
        path = self._v2_bundle(tmp_path, feature_version=2)
        result = recommend(path, tiny_circuit, "FakeManilaV2", base_shots=4096)
        assert result["feature_version"] == 2
        assert result["technique"] == "cdr"
        # v2 feature dict was actually used
        assert set(result["features"]) == set(FEATURE_NAMES_V2)

    def test_v2_abstain_when_below_threshold(self, tmp_path, tiny_circuit):
        path = self._v2_bundle(
            tmp_path,
            feature_version=1,
            abstain_threshold=0.8,
            proba_row=(0.6, 0.2, 0.1, 0.1),
        )
        result = recommend(path, tiny_circuit, "FakeManilaV2")
        assert result["technique"] == "abstain"
        assert result["abstained"] is True
        assert result["abstain_threshold"] == 0.8
        # the underlying class probabilities are still reported for the caller
        assert set(result["probabilities"]) == {"cdr", "raw", "rem", "zne"}

    def test_v2_no_abstain_when_above_threshold(self, tmp_path, tiny_circuit):
        path = self._v2_bundle(
            tmp_path,
            feature_version=1,
            abstain_threshold=0.5,
            proba_row=(0.6, 0.2, 0.1, 0.1),
        )
        result = recommend(path, tiny_circuit, "FakeManilaV2")
        assert result["technique"] == "cdr"
        assert result["abstained"] is False


# ---------------------------------------------------------------------------
# CLI scripts
# ---------------------------------------------------------------------------


class TestScriptsV2:
    def test_recommend_cli_shots_flag_passthrough(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr(recommend_mod, "extract_features", _fake_extract_v2)
        classes = ["cdr", "raw", "rem", "zne"]
        bundle = {
            "model": _FakeModel(classes, [0.6, 0.2, 0.1, 0.1]),
            "feature_names": _FEAT_COLS_V2,
            "classes": classes,
            "model_name": "fake",
            "qemsel_version": "0.1.0",
            "feature_version": 2,
            "abstain_threshold": None,
        }
        path = _dump_bundle(tmp_path / "v2.joblib", bundle)
        recommend_script = _load_script("recommend")
        rc = recommend_script.main(
            [
                "--model",
                str(path),
                "--backend",
                "FakeManilaV2",
                "--demo",
                "ghz_plus",
                "--qubits",
                "3",
                "--shots",
                "4096",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Recommended technique" in out
        assert '"feature_version": 2' in out

    def test_recommend_cli_abstain_exit_code(self, tmp_path, capsys):
        # feature_version 1 => real features, no monkeypatch needed; a high
        # abstain threshold forces the abstain output path + exit code 2.
        classes = ["cdr", "raw", "rem", "zne"]
        bundle = {
            "model": _FakeModel(classes, [0.6, 0.2, 0.1, 0.1]),
            "feature_names": _FEAT_COLS_V1,
            "classes": classes,
            "model_name": "fake",
            "qemsel_version": "0.1.0",
            "feature_version": 1,
            "abstain_threshold": 0.9,
        }
        path = _dump_bundle(tmp_path / "v2_abstain.joblib", bundle)
        recommend_script = _load_script("recommend")
        rc = recommend_script.main(
            [
                "--model",
                str(path),
                "--backend",
                "FakeManilaV2",
                "--demo",
                "ghz_plus",
                "--qubits",
                "3",
            ]
        )
        assert rc == 2
        out = capsys.readouterr().out
        assert "No confident recommendation" in out
        assert '"technique": "abstain"' in out

    def test_make_report_cli_with_stats_and_boundary(
        self, tiny_results_df, fake_metrics, tmp_path, capsys, stats_results
    ):
        data_path = tmp_path / "results.csv"
        metrics_path = tmp_path / "metrics.json"
        stats_path = tmp_path / "stats.json"
        boundary_path = tmp_path / "boundary.json"
        out = tmp_path / "reportdir"
        out.mkdir(parents=True, exist_ok=True)

        tiny_results_df.to_csv(data_path, index=False)
        metrics_path.write_text(json.dumps(fake_metrics), encoding="utf-8")
        stats_path.write_text(json.dumps(stats_results), encoding="utf-8")
        overlay = _boundary_overlay(out)  # writes boundary_overlay.png into out
        boundary_path.write_text(json.dumps(overlay), encoding="utf-8")

        make_report = _load_script("make_report")
        rc = make_report.main(
            [
                "--data",
                str(data_path),
                "--metrics",
                str(metrics_path),
                "--out",
                str(out),
                "--stats-json",
                str(stats_path),
                "--boundary-json",
                str(boundary_path),
            ]
        )
        assert rc == 0
        report_text = (out / "report.md").read_text(encoding="utf-8")
        assert "## 8. Statistical hygiene" in report_text
        assert "## 9. ZNE help-harm boundary overlay" in report_text
        assert "report written" in capsys.readouterr().out

    def test_make_report_cli_missing_stats_file_exits(
        self, tiny_results_df, fake_metrics, tmp_path
    ):
        data_path = tmp_path / "results.csv"
        metrics_path = tmp_path / "metrics.json"
        tiny_results_df.to_csv(data_path, index=False)
        metrics_path.write_text(json.dumps(fake_metrics), encoding="utf-8")
        make_report = _load_script("make_report")
        with pytest.raises(SystemExit):
            make_report.main(
                [
                    "--data",
                    str(data_path),
                    "--metrics",
                    str(metrics_path),
                    "--out",
                    str(tmp_path / "out"),
                    "--stats-json",
                    str(tmp_path / "nope_stats.json"),
                ]
            )

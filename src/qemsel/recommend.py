"""Recommend the best QEM technique for a new circuit using the trained model."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from qemsel.features import extract_features


class RecommendationResult(TypedDict, total=False):
    technique: str
    probabilities: dict[str, float]
    features: dict[str, float]
    abstained: bool
    abstain_threshold: float | None
    feature_version: int

#: Keys the joblib bundle written by ``qemsel.model.train_and_eval`` must have
#: for a recommendation to be possible ('model_name'/'qemsel_version' are
#: informational and therefore optional here).
_REQUIRED_BUNDLE_KEYS: tuple[str, ...] = ("model", "feature_names", "classes")


def _strip_feat_prefix(name: str) -> str:
    """Map a model feature column name (e.g. 'feat_depth') to the plain
    feature name used by ``extract_features`` (e.g. 'depth')."""
    return name[len("feat_"):] if name.startswith("feat_") else name


def _load_bundle(model_path: Path) -> dict:
    """Load and validate the model bundle dict from ``model_path``.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the loaded object is not the expected bundle dict.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"model file not found: {model_path} — run scripts/train_model.py "
            "first to produce model.joblib"
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(
            f"malformed model bundle in {model_path}: expected a dict with keys "
            f"{list(_REQUIRED_BUNDLE_KEYS)}, got {type(bundle).__name__}"
        )
    missing = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"malformed model bundle in {model_path}: missing keys {missing} "
            f"(has {sorted(bundle.keys())})"
        )
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError(
            f"malformed model bundle in {model_path}: 'model' object of type "
            f"{type(bundle['model']).__name__} has no predict_proba method"
        )
    return bundle


def recommend(
    model_path: Path,
    circuit: QuantumCircuit,
    backend_name: str,
    *,
    base_shots: int | None = None,
) -> RecommendationResult:
    """Predict the best QEM technique for one circuit on one backend.

    Args:
        model_path: path to the ``model.joblib`` BUNDLE saved by
            ``qemsel.model.train_and_eval`` (a dict with keys 'model',
            'feature_names', 'classes', 'model_name', 'qemsel_version';
            V2 bundles additionally carry 'feature_version', 'calibrated',
            'abstain_threshold').
        circuit: QuantumCircuit WITHOUT final measurements.
        backend_name: one of ``qemsel.backends.BACKENDS``.
        base_shots: (V2, keyword-only; builder-recommend / B8 implements)
            the planned shot budget. REQUIRED (ValueError if None) when the
            bundle's 'feature_version' is 2 (its features include
            log2_shots); IGNORED for V1 bundles. CLI: ``scripts/
            recommend.py --shots``.

    Behaviour contract:
    * Features come from ``qemsel.features.extract_features(circuit,
      backend_name)`` and are arranged in the bundle's 'feature_names'
      order (strip the 'feat_' prefix when mapping) — NEVER a hardcoded
      order. V2 (B8): pass ``version=bundle.get('feature_version', 1)``
      and ``base_shots=base_shots`` through to extract_features.
    * Probabilities from ``model.predict_proba``; the recommended technique
      is the argmax class. A 'calibrated' bundle needs no special handling
      (CalibratedClassifierCV exposes predict_proba).
    * V2 ABSTAIN (B8): when the bundle carries a non-None
      'abstain_threshold' and ``max(probabilities) < threshold``, the
      returned 'technique' is the literal string 'abstain' (never a class
      name) — the caller should fall back to a safe default ('raw') or a
      human.

    Returns:
        For a V1 bundle (no 'feature_version' key): dict with EXACTLY the
        three V1 keys — byte-identical behavior:
            'technique'     (str, e.g. 'zne')
            'probabilities' (dict[str, float], class label -> probability,
                             one entry per class in the bundle, sums to ~1)
            'features'      (dict[str, float], the extracted feature dict
                             actually fed to the model, for transparency)
        For a V2 bundle (has 'feature_version'), the same three keys PLUS:
            'abstained'         (bool)
            'abstain_threshold' (float | None, echoed from the bundle)
            'feature_version'   (int, echoed)

    Raises:
        FileNotFoundError: if model_path does not exist.
        ValueError: if the bundle is malformed, backend_name unknown, or a
            feature_version-2 bundle is queried without ``base_shots``.
    """
    bundle = _load_bundle(Path(model_path))
    model = bundle["model"]
    feature_names = [str(n) for n in bundle["feature_names"]]

    # ---- V1 bundle (no 'feature_version' key): byte-identical 3-key return.
    # base_shots is IGNORED here and extract_features is called with exactly
    # the two positional arguments the V1 path always used (so a 2-arg
    # monkeypatched stub still binds). -------------------------------------
    if "feature_version" not in bundle:
        features = extract_features(circuit, backend_name)
        probabilities, technique, _ = _predict(model, feature_names, features, bundle)
        return {
            "technique": technique,
            "probabilities": probabilities,
            "features": {k: float(v) for k, v in features.items()},
        }

    # ---- V2 bundle: shots-aware features + optional abstain ---------------
    feature_version = int(bundle["feature_version"])
    if feature_version == 2 and base_shots is None:
        raise ValueError(
            "this model bundle was trained with feature_version 2 (its "
            "features include log2_shots), so recommend() needs the planned "
            "shot budget — pass base_shots=<int> (CLI: --shots)"
        )
    features = extract_features(
        circuit, backend_name, version=feature_version, base_shots=base_shots
    )
    probabilities, argmax_technique, proba = _predict(
        model, feature_names, features, bundle
    )

    # Abstain when the bundle carries a threshold the top probability fails
    # to clear — the recommended 'technique' becomes the literal 'abstain'
    # (never a class name); the caller falls back to a safe default / human.
    raw_threshold = bundle.get("abstain_threshold")
    abstain_threshold = None if raw_threshold is None else float(raw_threshold)
    abstained = abstain_threshold is not None and float(np.max(proba)) < abstain_threshold
    technique = "abstain" if abstained else argmax_technique

    return {
        "technique": technique,
        "probabilities": probabilities,
        "features": {k: float(v) for k, v in features.items()},
        "abstained": bool(abstained),
        "abstain_threshold": abstain_threshold,
        "feature_version": feature_version,
    }


def _predict(
    model, feature_names: list[str], features: dict, bundle: dict
) -> tuple[dict, str, np.ndarray]:
    """Align features to the bundle column order, predict, return
    ``(probabilities dict, argmax technique, proba array)``.

    Shared by the V1 and V2 recommend paths so the feature-mismatch guard,
    named-DataFrame construction and class-order handling live in one place.
    """
    unmatched = [
        fn for fn in feature_names if _strip_feat_prefix(fn) not in features
    ]
    if unmatched:
        raise ValueError(
            "feature mismatch between model bundle and extract_features: bundle "
            f"expects columns {unmatched} but extracted features only provide "
            f"{sorted(features.keys())} — the model was likely trained with a "
            "different qemsel.features version"
        )
    row = [float(features[_strip_feat_prefix(fn)]) for fn in feature_names]
    # DataFrame (not bare ndarray) so sklearn's fitted feature-name check
    # passes when the model was trained on a named-column DataFrame.
    x = pd.DataFrame([row], columns=feature_names)

    proba = np.asarray(model.predict_proba(x), dtype=float)[0]
    # Class order of predict_proba columns comes from the fitted estimator.
    classes = [str(c) for c in getattr(model, "classes_", bundle["classes"])]
    if len(classes) != len(proba):
        raise ValueError(
            f"malformed model bundle: {len(classes)} classes but predict_proba "
            f"returned {len(proba)} probabilities"
        )
    probabilities = {c: float(p) for c, p in zip(classes, proba)}
    technique = classes[int(np.argmax(proba))]
    return probabilities, technique, proba

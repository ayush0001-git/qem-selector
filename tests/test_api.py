"""Tests for qemsel.api (MitigatedExecutor)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from qiskit import QuantumCircuit

from qemsel.api import MitigatedExecutor, run


def test_mitigated_executor_init_raises():
    with pytest.raises(FileNotFoundError):
        MitigatedExecutor("non_existent_model.joblib")


@patch("qemsel.api.recommend")
@patch("qemsel.api.make_executor")
@patch("qemsel.api.apply_technique")
def test_mitigated_executor_execute(mock_apply, mock_make, mock_recommend, tmp_path):
    # Create a dummy model file so FileNotFoundError is not raised
    dummy_model_path = tmp_path / "dummy_model.joblib"
    dummy_model_path.touch()

    executor = MitigatedExecutor(dummy_model_path)
    qc = QuantumCircuit(2)

    # Set up mock returns
    mock_recommend.return_value = {
        "technique": "zne",
        "probabilities": {"zne": 0.8, "raw": 0.2},
        "features": {"n_qubits": 2.0},
        "abstained": False,
    }
    mock_apply.return_value = 0.95
    # make_executor returns an executor callable
    mock_make.return_value = lambda q, p: 0.9

    res = executor.execute(qc, "ZZ", "FakeManilaV2", 1024)

    assert res["value"] == 0.95
    assert res["technique"] == "zne"
    assert res["abstained"] is False
    assert res["features"] == {"n_qubits": 2.0}

    # Test fallback on mitigation failure
    mock_apply.side_effect = [Exception("mitiq error"), 0.85]
    res_fallback = executor.execute(qc, "ZZ", "FakeManilaV2", 1024, fallback_technique="raw")
    assert res_fallback["value"] == 0.85
    assert res_fallback["technique"] == "raw"


@patch("qemsel.api.MitigatedExecutor")
def test_run_convenience(mock_executor_class):
    mock_inst = MagicMock()
    mock_inst.execute.return_value = {"value": 0.88}
    mock_executor_class.return_value = mock_inst

    qc = QuantumCircuit(2)
    val = run(qc, "ZZ", "FakeManilaV2", 1024, model_path="dummy.joblib")

    assert val == 0.88
    mock_inst.execute.assert_called_once()

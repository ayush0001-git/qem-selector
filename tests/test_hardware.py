"""Tests for qemsel.hardware and its dispatch seams (all fully mocked).

NO test here touches the network or real credentials: QiskitRuntimeService /
SamplerV2 / Batch / generate_preset_pass_manager are monkeypatched at the
``qemsel.hardware`` import site, and credential loading is either pointed at
tmp files or monkeypatched. Tokens used below are obvious fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from qiskit import QuantumCircuit

from qemsel import backends, hardware
from qemsel.backends import expectation_from_counts
from qemsel.experiment import _validate_config

# ---------------------------------------------------------------------------
# Shared fakes + state hygiene
# ---------------------------------------------------------------------------

FAKE_CREDS = {
    "token": "fake-token-for-tests",
    "instance": "crn:v1:test:fake",
    "default_backend": "ibm_testchip",
    "channel": "ibm_quantum_platform",
    "qpu_seconds_cap": None,
}


@pytest.fixture(autouse=True)
def _clean_hardware_state():
    """Reset the module-global usage counter and info cache around each test."""
    hardware.reset_qpu_usage()
    hardware.clear_hardware_caches()
    yield
    hardware.reset_qpu_usage()
    hardware.clear_hardware_caches()


def _make_fake_stack(monkeypatch, counts, num_qubits=127):
    """Monkeypatch SamplerV2/Batch/pass manager; return (service, calls log).

    The fake sampler returns ``counts`` for every job and records every
    submission; the fake Batch counts instantiations (shared-batch check).
    """
    calls = {"sampler_runs": [], "shots": [], "batches": 0}

    class FakeBackend:
        def __init__(self, name):
            self.name = name
            self.num_qubits = num_qubits

    class FakeService:
        def backend(self, name):
            return FakeBackend(name)

    class FakePassManager:
        def run(self, circuit):
            return circuit  # identity: "already ISA"

    class FakeBitArray:
        def get_counts(self):
            return dict(counts)

    class FakePubResult:
        def join_data(self):
            return FakeBitArray()

    class FakeJob:
        def result(self):
            return [FakePubResult()]

    class FakeSampler:
        def __init__(self, mode=None):
            self.mode = mode

        def run(self, pubs, shots=None):
            calls["sampler_runs"].append(list(pubs))
            calls["shots"].append(shots)
            return FakeJob()

    class FakeBatch:
        def __init__(self, backend=None):
            calls["batches"] += 1
            self.backend = backend
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(hardware, "SamplerV2", FakeSampler)
    monkeypatch.setattr(hardware, "Batch", FakeBatch)
    monkeypatch.setattr(
        hardware, "generate_preset_pass_manager", lambda **kw: FakePassManager()
    )
    return FakeService(), calls


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# load_credentials
# ---------------------------------------------------------------------------


def test_load_credentials_missing_file_returns_none(tmp_path: Path):
    assert hardware.load_credentials(tmp_path / "nope.yaml") is None


@pytest.mark.parametrize(
    "token_line", ["ibm_token: null", 'ibm_token: ""', "ibm_token: YOUR_API_KEY"]
)
def test_load_credentials_blank_token_returns_none(tmp_path: Path, token_line):
    path = tmp_path / "hardware.yaml"
    path.write_text(f"{token_line}\ninstance: crn:v1:test\n", encoding="utf-8")
    assert hardware.load_credentials(path) is None


def test_load_credentials_reads_all_fields(tmp_path: Path):
    path = tmp_path / "hardware.yaml"
    path.write_text(
        'ibm_token: "fake-token-for-tests"\n'
        'instance: "crn:v1:test:fake"\n'
        "default_backend: ibm_testchip\n"
        "qpu_seconds_cap: 42.5\n",
        encoding="utf-8",
    )
    creds = hardware.load_credentials(path)
    assert creds == {
        "token": "fake-token-for-tests",
        "instance": "crn:v1:test:fake",
        "default_backend": "ibm_testchip",
        "channel": "ibm_quantum_platform",  # default for the new platform
        "qpu_seconds_cap": 42.5,
    }


def test_load_credentials_minimal_token_only(tmp_path: Path):
    path = tmp_path / "hardware.yaml"
    path.write_text("ibm_token: fake-token-for-tests\n", encoding="utf-8")
    creds = hardware.load_credentials(path)
    assert creds["token"] == "fake-token-for-tests"
    assert creds["instance"] is None
    assert creds["qpu_seconds_cap"] is None


def test_load_credentials_parse_error_never_quotes_content(tmp_path: Path):
    """A malformed YAML error must not leak file content (could be a token)."""
    path = tmp_path / "hardware.yaml"
    path.write_text("ibm_token: [SECRETVALUE\n  broken", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        hardware.load_credentials(path)
    assert "SECRETVALUE" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# get_service
# ---------------------------------------------------------------------------


def test_get_service_uses_new_platform_channel(monkeypatch):
    class FakeQRS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(hardware, "QiskitRuntimeService", FakeQRS)
    service = hardware.get_service(credentials=dict(FAKE_CREDS))
    assert service.kwargs == {
        "channel": "ibm_quantum_platform",
        "token": "fake-token-for-tests",
        "instance": "crn:v1:test:fake",
    }


def test_get_service_omits_instance_when_none(monkeypatch):
    class FakeQRS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(hardware, "QiskitRuntimeService", FakeQRS)
    creds = dict(FAKE_CREDS, instance=None)
    service = hardware.get_service(credentials=creds)
    assert "instance" not in service.kwargs


def test_get_service_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(hardware, "load_credentials", lambda path=None: None)
    with pytest.raises(hardware.HardwareUnavailableError, match="hardware.yaml"):
        hardware.get_service()


# ---------------------------------------------------------------------------
# list_real_backends / get_real_backend_info
# ---------------------------------------------------------------------------


def test_list_real_backends_reports_queue_depth():
    class GoodStatus:
        operational = True
        status_msg = "active"
        pending_jobs = 17

    class GoodBackend:
        name = "ibm_good"
        num_qubits = 127

        def status(self):
            return GoodStatus()

    class BadStatusBackend:
        name = "ibm_flaky"
        num_qubits = 133

        def status(self):
            raise RuntimeError("status endpoint down")

    class FakeService:
        def backends(self):
            return [GoodBackend(), BadStatusBackend()]

    out = hardware.list_real_backends(service=FakeService())
    assert out[0] == {
        "name": "ibm_good",
        "n_qubits": 127,
        "operational": True,
        "status_msg": "active",
        "pending_jobs": 17,
    }
    assert out[1]["name"] == "ibm_flaky"
    assert out[1]["pending_jobs"] is None
    assert out[1]["status_msg"] == "unknown"


def test_get_real_backend_info_matches_fake_reference(monkeypatch):
    """Summarizing a real target must equal the fake path's aggregation.

    Uses FakeManilaV2's target as a stand-in for a live device target and
    compares against backends.get_backend_info('FakeManilaV2').
    """
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    class TargetOnlyBackend:
        target = FakeManilaV2().target
        num_qubits = 5

    class FakeService:
        def backend(self, name):
            return TargetOnlyBackend()

    info = hardware.get_real_backend_info("ibm_manila_like", service=FakeService())
    ref = backends.get_backend_info("FakeManilaV2")
    assert set(info) == set(ref)
    assert info["name"] == "ibm_manila_like"
    assert info["n_qubits"] == ref["n_qubits"]
    for key in ("avg_1q_error", "avg_2q_error", "avg_readout_error", "max_readout_error"):
        assert info[key] == pytest.approx(ref[key])

    # Second call must be served from the cache — never from the network.
    def _boom(*args, **kwargs):
        raise AssertionError("cache miss: get_service must not be called")

    monkeypatch.setattr(hardware, "get_service", _boom)
    again = hardware.get_real_backend_info("ibm_manila_like")
    assert again == info


# ---------------------------------------------------------------------------
# Dispatch seam in qemsel.backends
# ---------------------------------------------------------------------------


def test_make_executor_routes_ibm_names_to_hardware(monkeypatch):
    sentinel = object()
    recorded = {}

    def fake_make_real_executor(name, shots, seed):
        recorded.update(name=name, shots=shots, seed=seed)
        return sentinel

    monkeypatch.setattr(hardware, "make_real_executor", fake_make_real_executor)
    out = backends.make_executor("ibm_testchip", 512, 7)
    assert out is sentinel
    assert recorded == {"name": "ibm_testchip", "shots": 512, "seed": 7}


def test_get_backend_info_routes_ibm_names_to_hardware(monkeypatch):
    fake_info = {
        "name": "ibm_testchip",
        "n_qubits": 127,
        "avg_1q_error": 1e-4,
        "avg_2q_error": 1e-2,
        "avg_readout_error": 2e-2,
        "max_readout_error": 9e-2,
    }
    monkeypatch.setattr(
        hardware, "get_real_backend_info", lambda name, **kw: dict(fake_info)
    )
    assert backends.get_backend_info("ibm_testchip") == fake_info
    # The fake-backend cache must never absorb real-backend entries.
    assert "ibm_testchip" not in backends._INFO_CACHE


def test_fake_backend_path_unchanged():
    info = backends.get_backend_info("FakeManilaV2")
    assert list(info) == [
        "name",
        "n_qubits",
        "avg_1q_error",
        "avg_2q_error",
        "avg_readout_error",
        "max_readout_error",
    ]
    with pytest.raises(ValueError, match="unknown backend"):
        backends.get_backend_info("NotABackend")
    with pytest.raises(ValueError, match="unknown backend"):
        backends.make_executor("NotABackend", 128, 0)


# ---------------------------------------------------------------------------
# experiment._validate_config confirmation gate
# ---------------------------------------------------------------------------


def _hw_config(**overrides) -> dict:
    cfg = {
        "circuits": {
            "families": ["mirror_circuit"],
            "n_qubits": [2],
            "depths": [4],
            "seeds": [0],
        },
        "backends": ["ibm_testchip"],
        "shots": 128,
        "techniques": ["raw", "zne", "rem"],
    }
    cfg.update(overrides)
    return cfg


def test_validate_config_ibm_without_credentials_refuses(monkeypatch):
    monkeypatch.setattr(hardware, "load_credentials", lambda path=None: None)
    with pytest.raises(ValueError) as excinfo:
        _validate_config(_hw_config())
    msg = str(excinfo.value)
    assert "hardware.yaml" in msg
    assert "QPU" in msg  # estimated cost is stated even in the creds error


def test_validate_config_ibm_without_confirmation_states_cost(monkeypatch):
    monkeypatch.setattr(
        hardware, "load_credentials", lambda path=None: dict(FAKE_CREDS)
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_config(_hw_config())  # hardware_confirmed absent
    msg = str(excinfo.value)
    assert "hardware_confirmed" in msg
    assert "QPU-seconds" in msg  # cost stated
    assert "estimate_hardware_cost" in msg  # how to inspect it
    # hardware_confirmed must be EXACTLY True — truthy strings do not count.
    with pytest.raises(ValueError, match="hardware_confirmed"):
        _validate_config(_hw_config(hardware_confirmed="yes"))


def test_validate_config_ibm_with_credentials_and_confirmation_passes(monkeypatch):
    monkeypatch.setattr(
        hardware, "load_credentials", lambda path=None: dict(FAKE_CREDS)
    )
    (
        circuits_cfg,
        backend_names,
        shots,
        pauli,
        techniques,
        min_abs_ideal,
    ) = _validate_config(_hw_config(hardware_confirmed=True))
    assert backend_names == ["ibm_testchip"]
    assert techniques == ["raw", "zne", "rem"]
    assert shots == 128


def test_validate_config_fake_backends_never_touch_hardware(monkeypatch):
    """Simulated-only configs must validate even with zero credentials."""
    monkeypatch.setattr(hardware, "load_credentials", lambda path=None: None)
    cfg = _hw_config(backends=["FakeManilaV2"])
    _, backend_names, _, _, _, _ = _validate_config(cfg)
    assert backend_names == ["FakeManilaV2"]


def test_validate_config_unknown_non_ibm_backend_still_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        _validate_config(_hw_config(backends=["NotARealDevice"]))


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def test_estimate_config_arithmetic():
    cfg = _hw_config(shots=1000)
    est = hardware.estimate_config_qpu_seconds(cfg)
    assert est["n_circuits"] == 1
    assert est["n_ibm_backends"] == 1
    assert est["n_units"] == 1
    # raw 1 + zne 3 + rem 3 executor calls per unit
    assert est["per_technique_jobs_per_unit"] == {"raw": 1, "zne": 3, "rem": 3}
    assert est["jobs_per_unit"] == 7
    assert est["total_jobs"] == 7
    per_job = hardware.EST_JOB_OVERHEAD_SECONDS + 1000 * hardware.EST_SECONDS_PER_SHOT
    assert est["est_seconds_per_job"] == pytest.approx(per_job)
    assert est["est_total_qpu_seconds"] == pytest.approx(7 * per_job)
    assert est["fits_free_plan"] is True


def test_estimate_config_no_ibm_backends_is_free():
    est = hardware.estimate_config_qpu_seconds(_hw_config(backends=["FakeManilaV2"]))
    assert est["n_units"] == 0
    assert est["total_jobs"] == 0
    assert est["est_total_qpu_seconds"] == 0.0
    assert est["fits_free_plan"] is True


def test_estimate_config_oversized_run_does_not_fit():
    cfg = _hw_config(
        shots=100_000,
        techniques=["raw", "zne", "cdr", "rem"],
        circuits={
            "families": ["mirror_circuit", "layered_random"],
            "n_qubits": [2, 3],
            "depths": [4],
            "seeds": [0, 1, 2],
        },
    )
    est = hardware.estimate_config_qpu_seconds(cfg)
    assert est["per_technique_jobs_per_unit"]["cdr"] == 11
    assert est["est_total_qpu_seconds"] > est["free_plan_monthly_seconds"]
    assert est["fits_free_plan"] is False


def test_estimate_config_unknown_technique_raises():
    with pytest.raises(ValueError, match="unknown technique"):
        hardware.estimate_config_qpu_seconds(_hw_config(techniques=["raw", "magic"]))


def test_shipped_hw_first_run_config_fits_two_minutes():
    """configs/hw_first_run.yaml must honor its own <= ~2 QPU-min design."""
    path = Path(__file__).resolve().parents[1] / "configs" / "hw_first_run.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["hardware_confirmed"] is False  # ships unconfirmed
    assert "cdr" not in config["techniques"]  # 11x cost excluded by design
    assert all(str(b).startswith("ibm_") for b in config["backends"])
    assert len(config["backends"]) == 1
    est = hardware.estimate_config_qpu_seconds(config)
    assert 0 < est["est_total_qpu_seconds"] <= 150  # ~2 min + slack
    assert est["fits_free_plan"] is True


# ---------------------------------------------------------------------------
# Real-hardware executor (fake sampler stack)
# ---------------------------------------------------------------------------


def test_executor_counts_to_expectation(monkeypatch):
    counts = {"00": 500, "01": 200, "10": 200, "11": 124}
    service, calls = _make_fake_stack(monkeypatch, counts)
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0, service=service)
    value = executor(_bell(), "ZZ")
    # ZZ parity: 00/11 -> +1, 01/10 -> -1 => (500 + 124 - 400) / 1024
    assert value == pytest.approx(224 / 1024)
    assert value == pytest.approx(expectation_from_counts(counts, "ZZ"))
    assert calls["shots"] == [1024]
    executor.close()


def test_executor_applies_x_basis_rotation_and_measures(monkeypatch):
    service, calls = _make_fake_stack(monkeypatch, {"00": 1024})
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0, service=service)
    executor(_bell(), "XX")
    (submitted,) = calls["sampler_runs"][0]
    ops = submitted.count_ops()
    assert ops["h"] == 3  # 1 from the Bell prep + 2 basis rotations
    assert ops["measure"] == 2
    # The caller's circuit is never mutated.
    assert "measure" not in _bell().count_ops()


def test_executor_identity_pauli_submits_nothing(monkeypatch):
    service, calls = _make_fake_stack(monkeypatch, {"00": 1024})
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0, service=service)
    assert executor(_bell(), "II") == 1.0
    assert calls["sampler_runs"] == []
    assert calls["batches"] == 0
    assert hardware.qpu_seconds_used() == 0.0


def test_executor_shares_one_batch_across_calls(monkeypatch):
    service, calls = _make_fake_stack(monkeypatch, {"00": 512, "11": 512})
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0, service=service)
    executor(_bell(), "ZZ")
    executor(_bell(), "ZZ")
    assert len(calls["sampler_runs"]) == 2
    assert calls["batches"] == 1  # shared Batch, opened once
    executor.close()


def test_executor_validates_pauli_and_width(monkeypatch):
    service, _calls = _make_fake_stack(monkeypatch, {"00": 1024}, num_qubits=1)
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0, service=service)
    with pytest.raises(ValueError, match="pauli length"):
        executor(_bell(), "ZZZ")
    with pytest.raises(ValueError, match="only 1"):
        executor(_bell(), "ZZ")  # 2-qubit circuit on a 1-qubit fake device


def test_budget_guard_hard_stops_before_submission(monkeypatch):
    service, calls = _make_fake_stack(monkeypatch, {"00": 512, "11": 512})
    per_call = hardware.estimate_job_qpu_seconds(1, 1024)
    executor = hardware.make_real_executor(
        "ibm_testchip", 1024, 0, service=service, max_qpu_seconds=per_call * 1.5
    )
    executor(_bell(), "ZZ")  # 1st call fits the cap
    assert hardware.qpu_seconds_used() == pytest.approx(per_call)
    with pytest.raises(hardware.HardwareBudgetExceededError, match="hard stop"):
        executor(_bell(), "ZZ")  # 2nd would exceed -> refused BEFORE submit
    assert len(calls["sampler_runs"]) == 1  # nothing was submitted
    assert hardware.qpu_seconds_used() == pytest.approx(per_call)  # not charged


def test_budget_cap_from_credentials_file(monkeypatch):
    service, calls = _make_fake_stack(monkeypatch, {"00": 1024})
    creds = dict(FAKE_CREDS, qpu_seconds_cap=0.5)  # below one job's estimate
    monkeypatch.setattr(hardware, "load_credentials", lambda path=None: creds)
    monkeypatch.setattr(
        hardware, "get_service", lambda path=None, credentials=None: service
    )
    executor = hardware.make_real_executor("ibm_testchip", 1024, 0)
    with pytest.raises(hardware.HardwareBudgetExceededError):
        executor(_bell(), "ZZ")
    assert calls["sampler_runs"] == []


def test_make_real_executor_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(hardware, "load_credentials", lambda path=None: None)
    with pytest.raises(hardware.HardwareUnavailableError):
        hardware.make_real_executor("ibm_testchip", 1024, 0)


def test_usage_counter_is_process_wide(monkeypatch):
    """Two executors draw from the SAME in-process budget."""
    service, _calls = _make_fake_stack(monkeypatch, {"00": 512, "11": 512})
    per_call = hardware.estimate_job_qpu_seconds(1, 1024)
    cap = per_call * 2.5
    ex1 = hardware.make_real_executor(
        "ibm_testchip", 1024, 0, service=service, max_qpu_seconds=cap
    )
    ex2 = hardware.make_real_executor(
        "ibm_testchip", 1024, 1, service=service, max_qpu_seconds=cap
    )
    ex1(_bell(), "ZZ")
    ex2(_bell(), "ZZ")
    assert hardware.qpu_seconds_used() == pytest.approx(2 * per_call)
    with pytest.raises(hardware.HardwareBudgetExceededError):
        ex1(_bell(), "ZZ")
    hardware.reset_qpu_usage()
    assert hardware.qpu_seconds_used() == 0.0

"""Environment verification for the QEM-Selector project.

Checks: core imports, a V2 fake backend from qiskit-ibm-runtime, a noisy
Bell-circuit run on AerSimulator using NoiseModel.from_backend, and a tiny
mitiq ZNE example. Prints all key versions.
"""
import sys
import traceback

FAILURES = []


def section(name):
    print("\n" + "=" * 60)
    print(f"[{name}]")
    print("=" * 60)


# ---------------------------------------------------------------- versions
section("VERSIONS")
import matplotlib
import mitiq
import pandas
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
import sklearn
import yaml

print(f"python            : {sys.version.split()[0]}")
print(f"qiskit            : {qiskit.__version__}")
print(f"qiskit-aer        : {qiskit_aer.__version__}")
print(f"qiskit-ibm-runtime: {qiskit_ibm_runtime.version.__version__}")
print(f"mitiq             : {mitiq.__version__}")
print(f"scikit-learn      : {sklearn.__version__}")
print(f"pandas            : {pandas.__version__}")
print(f"matplotlib        : {matplotlib.__version__}")
print(f"pyyaml            : {yaml.__version__}")

# ------------------------------------------------- fake backend discovery
section("FAKE BACKEND DISCOVERY")
from qiskit_ibm_runtime import fake_provider

v2_names = sorted(
    n for n in dir(fake_provider)
    if n.startswith("Fake") and n.endswith("V2")
)
print(f"Found {len(v2_names)} V2 fake backend classes. First 15:")
print(", ".join(v2_names[:15]))

# NOTE: FakeLagosV2's stored calibration has ~27% readout error on q0/q1
# (verified empirically) — usable for REM benchmarks but not as the default.
preferred = ["FakeManilaV2", "FakeJakartaV2", "FakeBelemV2", "FakeLimaV2",
             "FakeQuitoV2", "FakeLagosV2"]
chosen_name = next((n for n in preferred if n in v2_names), None)
if chosen_name is None:
    chosen_name = v2_names[0]
BackendCls = getattr(fake_provider, chosen_name)
backend = BackendCls()
print(f"Using fake backend: {chosen_name} "
      f"(name={backend.name}, num_qubits={backend.num_qubits})")

# --------------------------------------------- noisy Bell circuit on Aer
section("NOISY BELL CIRCUIT ON AerSimulator")
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    noise_model = NoiseModel.from_backend(backend)
    print(f"NoiseModel built from {chosen_name}: "
          f"basis_gates={noise_model.basis_gates}")

    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    sim = AerSimulator(noise_model=noise_model)
    tqc = transpile(bell, sim)
    counts = sim.run(tqc, shots=4096).result().get_counts()
    print(f"Bell counts (noisy): {counts}")
    total = sum(counts.values())
    good = counts.get("00", 0) + counts.get("11", 0)
    print(f"Fidelity proxy P(00)+P(11) = {good / total:.4f}")
    assert good / total > 0.8, "Bell state fidelity suspiciously low"
except Exception:
    FAILURES.append("noisy-bell")
    traceback.print_exc()

# ----------------------------------------------------------- mitiq ZNE
section("MITIQ ZNE EXAMPLE")
try:
    from mitiq import zne
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    noise_model = NoiseModel.from_backend(backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    ideal_sim = AerSimulator()

    circ = QuantumCircuit(1)
    for _ in range(10):
        circ.x(0)  # identity-equivalent; ideal <0|rho|0> = 1

    def executor(circuit):
        """Return probability of measuring |0...0>."""
        c = circuit.copy()
        c.measure_all()
        tc = transpile(c, noisy_sim, optimization_level=0)
        counts = noisy_sim.run(tc, shots=8192).result().get_counts()
        zero = "0" * circuit.num_qubits
        return counts.get(zero, 0) / sum(counts.values())

    unmitigated = executor(circ)
    mitigated = zne.execute_with_zne(circ, executor)
    print(f"Unmitigated <P0> : {unmitigated:.4f}")
    print(f"ZNE mitigated    : {mitigated:.4f}   (ideal = 1.0)")
    assert 0.0 < mitigated <= 1.5, "ZNE result out of sane range"
except Exception:
    FAILURES.append("mitiq-zne")
    traceback.print_exc()

# ------------------------------------------------- mitiq module layout
section("MITIQ MODULE INTROSPECTION")
print(f"mitiq version: {mitiq.__version__}")
top = [n for n in dir(mitiq) if not n.startswith("_")]
print(f"dir(mitiq): {top}")
for mod in ("zne", "cdr", "rem"):
    try:
        m = __import__(f"mitiq.{mod}", fromlist=[mod])
        entry = [f for f in dir(m) if f.startswith("execute_with")]
        print(f"import mitiq.{mod}  OK  entrypoints={entry}")
    except Exception as exc:
        FAILURES.append(f"mitiq-{mod}")
        print(f"import mitiq.{mod}  FAILED: {exc}")

section("RESULT")
if FAILURES:
    print(f"FAILED sections: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")

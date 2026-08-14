"""API spike: CDR with swappable sklearn regressors (cdr_nonlinear route decision).

mitiq 1.0.0 / qiskit 2.5.0 / sklearn 1.9.0. Run:
    & ".\\.venv\\Scripts\\python.exe" spikes\\spike_cdr_nl.py

QUESTION (Angle 2, docs/RESEARCH_ANGLES.md): to add CDR-with-nonlinear-regressor
as a selectable technique, which implementation route works in mitiq 1.0.0?

  Route A - ``execute_with_cdr(fit_function=..., num_fit_parameters=...)``.
  Route B - bypass mitiq's fit: ``mitiq.cdr.generate_training_circuits`` +
            run the executor/ideal simulator on the training circuits ourselves
            + fit ANY sklearn regressor + predict the target.

DECISION (verified below): **Route B** for sklearn regressors; Route A stays
usable only for unregularized parametric curve shapes (polynomials).

Why Route A cannot express Ridge or RandomForest (structural, from installed
source .venv/Lib/site-packages/mitiq/cdr/cdr.py):

* Lines 160-166: the regression is ``scipy.optimize.curve_fit(lambda x,
  *params: fit_function(x, params), noisy[1:].T, ideal, p0=zeros)``.
  ``curve_fit`` minimizes the plain sum of squared residuals over the
  internally generated training data; ``fit_function`` only shapes the
  PREDICTION f(x; params), never the LOSS. Ridge = OLS + alpha*||w||^2 -- the
  penalty term lives in the loss, so no choice of fit_function reproduces it.
  (The textbook workaround -- augment X with sqrt(alpha)*I pseudo-rows -- needs
  write access to the training matrix, which is built inside execute_with_cdr
  at lines 134-157 and is not injectable.)
* RandomForest has no finite parameter vector at all -> curve_fit-incompatible.
* Route A DOES work for parametric fits (demonstrated below with a quadratic,
  num_fit_parameters=3), which is what CDR_FIT_FUNCTION/CDR_NUM_FIT_PARAMETERS
  in qemsel.mitigation are for.

Route B facts verified by this spike:

* ``generate_training_circuits(circuit, num_training_circuits,
  fraction_non_clifford, method_select='uniform', method_replace='closest',
  random_state=int)`` is seedable and, called with the same args qemsel's
  ``_apply_cdr`` uses, produces the SAME training set mitiq would build
  internally (qemsel already pre-generates it for the degeneracy guard).
* Executor cost is EXACTLY 1 (target) + N (training circuits) noisy calls --
  identical to mitiq's own path (mitiq additionally runs
  ``scale_noise(c, 1)`` = no-op folding at scale 1). SHOT_MULTIPLIER for a
  future cdr_nonlinear technique is therefore the same formula as cdr:
  ``1 + num_training_circuits``.
* With the SAME training data, Route B + LinearRegression reproduces mitiq's
  Route A linear-CDR value (delta printed below; the qemsel executor is fully
  deterministic per circuit, so both routes see identical noisy values up to
  mitiq's scale-1 fold round-trip).
* Full pipeline is bit-reproducible under a fixed seed (asserted) and moves
  under a different seed (printed).

Comparison run: layered_random q3 d8 on FakeManilaV2 @ 4096 shots,
linear-CDR vs Ridge-CDR (alpha grid via LOO-CV RidgeCV) vs RF-CDR at
N = 10 and 30 training circuits.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from mitiq.cdr import execute_with_cdr, generate_training_circuits
from mitiq.cdr.clifford_utils import is_clifford

from qemsel import backends, circuits, ideal
from qemsel.mitigation import CDR_BASIS_GATES, CDR_FRACTION_NON_CLIFFORD

# ---------------------------------------------------------------------------
# Settings (match the research sweep where a choice exists).
# ---------------------------------------------------------------------------
BACKEND = "FakeManilaV2"
SHOTS = 4096  # research-config base shots
PAULI = "ZZZ"  # Z on every qubit, qemsel convention (symmetric -> endian-moot)
N_QUBITS, DEPTH = 3, 8
MIN_ABS_IDEAL = 0.25  # same signal floor the suite generator enforces
TRAIN_SIZES = (10, 30)
RIDGE_ALPHAS = np.logspace(-6, 3, 19)  # alpha grid, LOO-CV picks (deterministic)
RF_KWARGS = dict(n_estimators=100)  # random_state supplied per-run


class CountingExecutor:
    """Wraps a qemsel executor; counts noisy invocations exactly."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, circuit: QuantumCircuit, pauli: str) -> float:
        self.calls += 1
        return self._inner(circuit, pauli)


def pick_circuit() -> tuple[QuantumCircuit, int, float]:
    """First layered_random(q3, d8) seed in 0..9 with |ideal| >= floor.

    Deterministic stand-in for generate_suite's min_abs_ideal rejection
    sampling, so the spike's errors are signal, not a shot-noise lottery.
    """
    for seed in range(10):
        qc = circuits.layered_random(N_QUBITS, DEPTH, seed)
        mu0 = ideal.ideal_expectation(qc, PAULI)
        if abs(mu0) >= MIN_ABS_IDEAL:
            return qc, seed, mu0
    raise RuntimeError("no layered_random q3 d8 seed in 0..9 passes the floor")


def collect_training_data(
    compiled: QuantumCircuit,
    executor,
    n_train: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Route B data step: N training pairs (noisy, ideal) + target noisy value.

    Executor cost: exactly n_train + 1 noisy calls (N training + 1 target).
    Training set generated with the SAME (circuit, count, fraction,
    random_state) qemsel's _apply_cdr / mitiq's execute_with_cdr use.
    """
    train = generate_training_circuits(
        compiled,
        num_training_circuits=n_train,
        fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
        random_state=seed,
    )
    x_noisy = np.array([[executor(tc, PAULI)] for tc in train])  # (N, 1)
    y_ideal = np.array([ideal.ideal_expectation(tc, PAULI) for tc in train])
    target_noisy = float(executor(compiled, PAULI))
    return x_noisy, y_ideal, target_noisy


def fit_predict(regressor, x_noisy, y_ideal, target_noisy) -> float:
    """Route B fit step: sklearn regressor noisy -> ideal, applied to target."""
    regressor.fit(x_noisy, y_ideal)
    return float(regressor.predict(np.array([[target_noisy]]))[0])


def quadratic_fit_function(x, params):
    """Route A demo: y = a*x^2 + b*x + c (curve_fit-parametric, 3 params)."""
    a, b, c = params
    return a * x[0] ** 2 + b * x[0] + c


def main() -> None:
    t0 = time.perf_counter()
    qc, seed, mu0 = pick_circuit()
    print(f"circuit: layered_random q{N_QUBITS} d{DEPTH} seed={seed}  "
          f"ideal <{PAULI}> = {mu0:+.4f}")

    executor = CountingExecutor(backends.make_executor(BACKEND, SHOTS, seed))

    # Same pre-compile as qemsel.mitigation._apply_cdr (CDR hard requirement:
    # all non-Clifford content in rz).
    compiled = transpile(
        qc,
        basis_gates=list(CDR_BASIS_GATES),
        optimization_level=0,
        seed_transpiler=seed,
    )
    assert not is_clifford(compiled), "spike needs a non-Clifford circuit"

    raw = executor(compiled, PAULI)
    assert executor.calls == 1
    print(f"raw noisy = {raw:+.4f}   |err| = {abs(raw - mu0):.4f}   "
          f"[1 executor call]")
    print()

    rows = []  # (label, n_train, value, executor_calls)
    for n_train in TRAIN_SIZES:
        # ---- Route A: mitiq linear CDR (the production 'cdr' technique) ----
        before = executor.calls
        route_a_linear = float(np.real(execute_with_cdr(
            compiled,
            lambda circ: executor(circ, PAULI),
            simulator=lambda circ: ideal.ideal_expectation(circ, PAULI),
            num_training_circuits=n_train,
            fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
            random_state=seed,
        )))
        calls_a = executor.calls - before
        assert calls_a == n_train + 1, (
            f"route A executed {calls_a} noisy circuits, expected {n_train + 1}"
        )
        rows.append(("route A linear (mitiq)", n_train, route_a_linear, calls_a))

        # ---- Route A: custom parametric fit (quadratic) -- proves the
        # CDR_FIT_FUNCTION path works for curve shapes, its ceiling ----------
        before = executor.calls
        route_a_quad = float(np.real(execute_with_cdr(
            compiled,
            lambda circ: executor(circ, PAULI),
            simulator=lambda circ: ideal.ideal_expectation(circ, PAULI),
            num_training_circuits=n_train,
            fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
            random_state=seed,
            fit_function=quadratic_fit_function,
            num_fit_parameters=3,
        )))
        calls_aq = executor.calls - before
        assert calls_aq == n_train + 1
        rows.append(("route A quadratic fit", n_train, route_a_quad, calls_aq))

        # ---- Route B: one data collection, three regressors ---------------
        before = executor.calls
        x_noisy, y_ideal, target_noisy = collect_training_data(
            compiled, executor, n_train, seed
        )
        calls_b = executor.calls - before
        assert calls_b == n_train + 1, (
            f"route B executed {calls_b} noisy circuits, expected {n_train + 1}"
        )
        spread = float(np.ptp(y_ideal))
        print(f"N={n_train}: training ideal spread {spread:.4f}, "
              f"noisy spread {float(np.ptp(x_noisy)):.4f}, "
              f"target noisy {target_noisy:+.4f}")

        b_linear = fit_predict(LinearRegression(), x_noisy, y_ideal, target_noisy)
        b_ridge = fit_predict(
            RidgeCV(alphas=RIDGE_ALPHAS), x_noisy, y_ideal, target_noisy
        )
        b_rf = fit_predict(
            RandomForestRegressor(random_state=seed, **RF_KWARGS),
            x_noisy, y_ideal, target_noisy,
        )
        rows.append(("route B linear (sklearn)", n_train, b_linear, calls_b))
        rows.append(("route B ridge (LOO-CV alpha)", n_train, b_ridge, calls_b))
        rows.append(("route B random forest", n_train, b_rf, calls_b))

        # Equivalence: same training set + deterministic executor => route B
        # LinearRegression should land on mitiq's linear-CDR value (any gap
        # comes from mitiq's scale-1 fold round-trip of the executed circuits).
        print(f"N={n_train}: |route B linear - route A linear| = "
              f"{abs(b_linear - route_a_linear):.2e}")
        print()

    # ---- Results table -----------------------------------------------------
    print(f"{'technique':<30} {'N':>3} {'value':>9} {'|error|':>9} {'calls':>6}")
    for label, n_train, value, calls in rows:
        print(f"{label:<30} {n_train:>3} {value:>+9.4f} "
              f"{abs(value - mu0):>9.4f} {calls:>6}")
    print(f"{'raw (baseline)':<30} {'-':>3} {raw:>+9.4f} "
          f"{abs(raw - mu0):>9.4f} {1:>6}")
    print()

    # ---- Seedability: identical seed -> bit-identical; new seed -> moves ---
    n_check = TRAIN_SIZES[0]
    exec2 = CountingExecutor(backends.make_executor(BACKEND, SHOTS, seed))
    x2, y2, t2 = collect_training_data(compiled, exec2, n_check, seed)
    rerun_ridge = fit_predict(RidgeCV(alphas=RIDGE_ALPHAS), x2, y2, t2)
    rerun_rf = fit_predict(
        RandomForestRegressor(random_state=seed, **RF_KWARGS), x2, y2, t2
    )
    first_ridge = next(v for l, n, v, _ in rows
                       if l.startswith("route B ridge") and n == n_check)
    first_rf = next(v for l, n, v, _ in rows
                    if l.startswith("route B random") and n == n_check)
    assert rerun_ridge == first_ridge, "Ridge-CDR not reproducible under fixed seed"
    assert rerun_rf == first_rf, "RF-CDR not reproducible under fixed seed"
    assert exec2.calls == n_check + 1
    print(f"seedability: same seed ({seed}) -> bit-identical "
          f"(ridge {rerun_ridge:+.6f}, rf {rerun_rf:+.6f})  [PASS]")

    alt_seed = seed + 1
    exec3 = CountingExecutor(backends.make_executor(BACKEND, SHOTS, alt_seed))
    x3, y3, t3 = collect_training_data(compiled, exec3, n_check, alt_seed)
    alt_ridge = fit_predict(RidgeCV(alphas=RIDGE_ALPHAS), x3, y3, t3)
    print(f"seedability: seed {alt_seed} -> ridge {alt_ridge:+.6f} "
          f"(moved by {abs(alt_ridge - first_ridge):.2e}, as expected)")
    print()
    print(f"total noisy executor calls: {executor.calls + exec2.calls + exec3.calls}"
          f"  |  wall time {time.perf_counter() - t0:.1f} s")
    print("VERDICT: Route B (generate_training_circuits + own fit) is the "
          "implementation route for sklearn regressors; cost = 1 + N calls, "
          "same SHOT_MULTIPLIER formula as linear cdr.")


if __name__ == "__main__":
    main()

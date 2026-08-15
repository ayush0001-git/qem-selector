# testing-guide — qemsel Testing & Regression Standards

This guide outlines the testing practices, regression anchors, and development contracts that must be adhered to when maintaining or updating the `qemsel` package.

---

## 1. Frozen Regression Anchors

We employ **capture-first regression anchors** to ensure that any changes to the execution engines or noise models do not silently shift baseline physical values. The following anchors are hardcoded and must stay byte-identical:

* **Plain-Name / Upward Noise Anchor:**
  * **Backend:** `FakeManilaV2` (or `@x1.0`)
  * **Circuit:** 2-qubit Bell circuit (`h(0); cx(0,1)`)
  * **Pauli:** `ZZ`
  * **Parameters:** `shots=256`, `seed=7`
  * **Expected Value:** `0.8671875` (exactly $222/256$, mapped to `_PRE_CHANGE_BELL_ZZ`).

* **Sub-unity Noise Anchor 1 (Scale 0.5):**
  * **Backend:** `FakeManilaV2@x0.5`
  * **Parameters:** `shots=256`, `seed=7`
  * **Expected Value:** `0.9375` (exactly $240/256$, mapped to `_SUBUNITY_BELL_ZZ_X0_5`).

* **Sub-unity Noise Anchor 2 (Scale 0.25):**
  * **Backend:** `FakeManilaV2@x0.25`
  * **Parameters:** `shots=256`, `seed=7`
  * **Expected Value:** `0.96875` (exactly $248/256$, mapped to `_SUBUNITY_BELL_ZZ_X0_25`).

> [!WARNING]
> Any code change or "bug fix" in the noise model construction, execution wrappers, or scaling routines that shifts these values is a regression and will break the test suite.

---

## 2. The conftest.py Contract

* **Ownership:** `tests/conftest.py` is owned strictly by the core architecture skeleton.
* **Modification Policy:** Feature builders and script maintainers must **not** add, remove, or modify fixtures in `conftest.py`.
* **Testing Practice:** If a test requires specific mock configurations, use standard unit test monkeypatching or `unittest.mock.patch` inside the local test module itself rather than modifying global fixtures.

---

## 3. Running the Test Suite

We use `pytest` for all verifications. To run tests, run from the project root:

```powershell
# Run the full test suite
& ".\.venv\Scripts\python.exe" -m pytest

# Run only fast tests (excluding slow regression sweeps)
& ".\.venv\Scripts\python.exe" -m pytest -m "not slow"

# Run tests in quiet mode
& ".\.venv\Scripts\python.exe" -m pytest -q
```

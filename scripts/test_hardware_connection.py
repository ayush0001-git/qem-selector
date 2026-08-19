"""CLI: verify IBM Quantum credentials and connectivity. Submits NOTHING.

Usage (from the project root, venv python)::

    python scripts/test_hardware_connection.py
    python scripts/test_hardware_connection.py --credentials configs/hardware.yaml

Reads configs/hardware.yaml, connects to the new IBM Quantum Platform
(quantum.cloud.ibm.com), lists the real backends your instance can access
(with queue depth) and your remaining free-plan usage if the API exposes it.
No job is ever submitted. The API token is never printed.

Exit codes: 0 = connected OK, 1 = connection/auth failed, 2 = credentials
file missing/blank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qemsel import hardware


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Check IBM Quantum credentials + connectivity (read-only; "
            "submits nothing; never prints the token)."
        )
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help="credentials YAML (default: configs/hardware.yaml)",
    )
    return parser.parse_args(argv)


def _redact(text: str, secret: str) -> str:
    """Remove the token from a message, just in case a library echoed it."""
    return text.replace(secret, "<token hidden>") if secret else text


def main(argv: list[str] | None = None) -> int:
    """Run the connection check; print clear success/failure guidance."""
    args = parse_args(argv)
    path = args.credentials if args.credentials is not None else (
        hardware.DEFAULT_CREDENTIALS_PATH
    )

    # --- 1. credentials file ------------------------------------------------
    if not Path(path).exists():
        print(f"FAIL: credentials file not found: {path}")
        print()
        print("Fix: create a free account at https://quantum.cloud.ibm.com,")
        print("generate an API key, copy your instance CRN, and put both in")
        print("configs\\hardware.yaml like:")
        print('  ibm_token: "YOUR_API_KEY"')
        print('  instance: "crn:v1:bluemix:public:quantum-computing:..."')
        print("(the file is gitignored -- never commit or share it)")
        return 2
    try:
        creds = hardware.load_credentials(path)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 2
    if creds is None:
        print(f"FAIL: {path} exists but ibm_token is blank/placeholder.")
        print()
        print("Fix: paste your real API key from https://quantum.cloud.ibm.com")
        print('into the ibm_token field: ibm_token: "YOUR_API_KEY"')
        return 2
    print("credentials file: OK (token loaded -- value hidden)")
    print(f"  channel:         {creds['channel']}")
    print(f"  instance CRN:    {'set' if creds['instance'] else 'NOT SET (recommended: set it)'}")
    print(f"  default_backend: {creds['default_backend'] or '(none)'}")

    # --- 2. authenticate ----------------------------------------------------
    print()
    print("connecting to IBM Quantum Platform (quantum.cloud.ibm.com)...")
    try:
        service = hardware.get_service(credentials=creds)
        # Force a real API round-trip now so auth errors surface here.
        backends = hardware.list_real_backends(service=service)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: classify + guide
        msg = _redact(str(exc), creds["token"])
        low = msg.lower()
        print(f"FAIL: could not connect: {type(exc).__name__}: {msg}")
        print()
        if "401" in low or "unauthorized" in low or "denied" in low or (
            "invalid" in low and ("token" in low or "key" in low or "api" in low)
        ):
            print("This looks like a BAD API KEY (authentication rejected).")
            print("Fix: regenerate the key at https://quantum.cloud.ibm.com")
            print("(Access management -> API keys) and paste the new value")
            print("into ibm_token in configs\\hardware.yaml.")
        elif "crn" in low or "instance" in low or "404" in low or "not found" in low:
            print("This looks like a BAD INSTANCE CRN (account found, but the")
            print("instance is wrong/inaccessible).")
            print("Fix: copy the exact CRN of your Open Plan instance from")
            print("https://quantum.cloud.ibm.com (Instances page) into the")
            print("instance field in configs\\hardware.yaml. It starts with")
            print("'crn:v1:bluemix:public:quantum-computing:...'")
        else:
            print("Could not classify the failure. Check your internet")
            print("connection, then verify BOTH ibm_token and instance in")
            print("configs\\hardware.yaml against https://quantum.cloud.ibm.com.")
        return 1

    # --- 3. report ----------------------------------------------------------
    print(f"SUCCESS: authenticated. {len(backends)} real backend(s) visible:")
    print()
    print(f"  {'name':<22} {'qubits':>6} {'operational':>12} {'queue':>7}")
    for entry in backends:
        oper = {True: "yes", False: "NO"}.get(entry["operational"], "?")
        queue = entry["pending_jobs"] if entry["pending_jobs"] is not None else "?"
        print(
            f"  {entry['name']:<22} {entry['n_qubits']:>6} {oper:>12} "
            f"{queue:>7}"
        )
    print()
    try:
        usage = service.usage()
    except Exception:  # noqa: BLE001 - usage endpoint is best-effort
        usage = None
    if isinstance(usage, dict) and usage:
        limit = usage.get("usage_limit_seconds") or usage.get(
            "usage_allocation_seconds"
        )
        consumed = usage.get("usage_consumed_seconds")
        remaining = usage.get("usage_remaining_seconds")
        print("free-plan usage (this month):")
        if consumed is not None:
            print(f"  consumed:  {float(consumed):.1f} s")
        if limit is not None:
            print(f"  limit:     {float(limit):.1f} s ({float(limit) / 60.0:.0f} min)")
        if remaining is not None:
            print(f"  remaining: {float(remaining):.1f} s ({float(remaining) / 60.0:.1f} min)")
        if consumed is None and limit is None and remaining is None:
            print("  (API returned no recognizable usage fields)")
    else:
        print("usage info: not exposed by the API for this account/instance.")
    print()
    print("No jobs were submitted. Next steps:")
    print("  1. put an accessible backend name into configs\\hw_first_run.yaml")
    print("  2. python scripts\\estimate_hardware_cost.py --config configs\\hw_first_run.yaml")
    print("  3. set hardware_confirmed: true in that config (your cost consent)")
    print("  4. python scripts\\run_experiment.py --config configs\\hw_first_run.yaml --out results\\hw_first_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

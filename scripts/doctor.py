#!/usr/bin/env python3
"""Read-only environment and dependency checks for ASU Sol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "openpi": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "iotauth": "1a40ebece054880e4fe400c1c5880b05ac91ad79",
}


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    hostname = socket.gethostname()
    login_node = "login" in hostname.lower()
    checks: dict[str, object] = {
        "hostname": hostname,
        "login_node": login_node,
        "python": platform.python_version(),
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "openpi_data_home": os.environ.get("OPENPI_DATA_HOME"),
        "commands": {
            command: shutil.which(command)
            for command in ("apptainer", "git", "java", "mvn", "node", "npm", "openssl", "uv")
        },
        "submodules": {
            name: {
                "path": str(ROOT / "third_party" / name),
                "expected": expected,
                "actual": git_head(ROOT / "third_party" / name),
            }
            for name, expected in EXPECTED.items()
        },
    }
    checks["submodule_pins_ok"] = all(
        entry["actual"] == entry["expected"]  # type: ignore[index]
        for entry in checks["submodules"].values()  # type: ignore[union-attr]
    )
    checks["compute_work_allowed"] = not login_node
    ok = bool(checks["python_supported"] and checks["submodule_pins_ok"])
    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        print(f"host: {hostname} ({'login node' if login_node else 'compute or local node'})")
        python_state = "OK" if checks["python_supported"] else "requires 3.11"
        print(f"python: {checks['python']} ({python_state})")
        for name, entry in checks["submodules"].items():  # type: ignore[union-attr]
            state = "OK" if entry["actual"] == entry["expected"] else "MISMATCH"
            print(f"{name}: {state} {entry['actual'] or 'missing'}")
        for command, path in checks["commands"].items():  # type: ignore[union-attr]
            print(f"{command}: {path or 'missing'}")
        if login_node:
            print("NOTE: clone, edit, inspect, and submit only. Run compute work in an allocation.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


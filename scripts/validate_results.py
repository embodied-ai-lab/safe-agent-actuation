#!/usr/bin/env python3
"""Validate structured scenario records without claiming they are live output.

The validator enforces the committed action-record schema and the fail-closed
invariant that a held or blocked action was never executed.  It accepts either
directories (searched for ``action_record.json``) or individual JSON files, so
its documented invocations both work:

    python3 scripts/validate_results.py
    python3 scripts/validate_results.py results
    python3 scripts/validate_results.py results/low-risk/action_record.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

REQUIRED_TOP = (
    "run_id",
    "timestep",
    "action_source",
    "model_configuration",
    "motion_analysis",
    "authorization_requested",
    "actuation",
    "simulator_outcome",
)
REQUIRED_MOTION = ("valid", "classification", "motion_score", "threshold")
KNOWN_SOURCES = {
    "captured_live_pi0_fast",
    "diagnostic_synthetic",
    "generated_boundary_test",
}
HELD_DECISIONS = {"hold", "blocked"}
EXECUTED_OUTCOMES = {"executed", "executed_in_recording_harness"}


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(path: Path, record: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TOP:
        if field not in record:
            errors.append(f"{path}: missing {field}")

    source = record.get("action_source")
    if source is not None and source not in KNOWN_SOURCES:
        errors.append(f"{path}: unknown action_source {source!r}")

    motion = record.get("motion_analysis")
    if isinstance(motion, dict):
        for field in REQUIRED_MOTION:
            if field not in motion:
                errors.append(f"{path}: motion_analysis missing {field}")
        score = motion.get("motion_score")
        threshold = motion.get("threshold")
        if not _finite_number(score):
            errors.append(f"{path}: motion_score must be finite")
        if not _finite_number(threshold):
            errors.append(f"{path}: threshold must be finite")
        classification = motion.get("classification")
        if motion.get("valid") and _finite_number(score) and _finite_number(threshold):
            expected = "high" if score >= threshold else "low"
            if classification != expected:
                errors.append(
                    f"{path}: classification {classification!r} does not match the "
                    f">= threshold rule (expected {expected!r})"
                )
    elif motion is not None:
        errors.append(f"{path}: motion_analysis must be an object")

    actuation = record.get("actuation")
    decision = actuation.get("decision") if isinstance(actuation, dict) else None
    outcome = record.get("simulator_outcome")
    if decision in HELD_DECISIONS and outcome in EXECUTED_OUTCOMES:
        errors.append(f"{path}: a {decision} action was marked executed ({outcome})")

    # Secret material must never appear in a record.
    text = json.dumps(record).lower()
    for fragment in ("cipher_key", "mac_key", "private_key", "-----begin"):
        if fragment in text:
            errors.append(f"{path}: record contains secret-bearing text {fragment!r}")
    return errors


def discover(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("action_record.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"path not found: {path}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Directories to search or JSON records; defaults to results/.",
    )
    args = parser.parse_args(argv)
    search = args.paths or [Path("results")]
    records = discover(search)
    if not records:
        raise SystemExit("no action records found")

    errors: list[str] = []
    for path in records:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path}: record must be a JSON object")
            continue
        errors.extend(validate_record(path, record))

    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated {len(records)} action record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

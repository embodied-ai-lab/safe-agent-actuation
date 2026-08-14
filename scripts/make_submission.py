#!/usr/bin/env python3
"""Build one group ZIP from a fixed allowlist, independent of Git state.

The builder never inspects Git.  It collects a student-editable allowlist plus
required result categories, refuses symlinks that escape the repository, rejects
oversized files, and never lets the output ZIP overwrite one of its own inputs.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024

ALLOWED_FILES = {
    "answers.md",
    "README.md",
    "ASSIGNMENT.md",
    "SECURITY.md",
    "Makefile",
    "pyproject.toml",
    "containers/Apptainer.def",
}
# Every student-editable project area. Instructor and solution-only tools are
# removed by name below, so adding a new student script does not need a change
# here.
ALLOWED_PREFIXES = (
    "src/",
    "configs/",
    "data/",
    "scripts/",
    "sst/",
    "slurm/",
    "tests/",
    "results/",
)
ALLOWED_SUFFIXES = {
    ".py",
    ".json",
    ".jsonl",
    ".csv",
    ".npz",
    ".png",
    ".md",
    ".toml",
    ".sh",
    ".sbatch",
    ".graph",
    ".def",
}
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    ".venv-openpi",
    "__pycache__",
    "instructor",
    "runtime",
    "third_party",
    "submission",
}
# Solution-only and instructor-only tools never belong in a student ZIP, even
# though they live in an otherwise allowed directory.
FORBIDDEN_NAMES = {
    "export_student_template.py",
    "validate_student_release.py",
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".der",
    ".jks",
    ".key",
    ".pem",
    ".pfx",
    ".p12",
    ".safetensors",
    ".sif",
    ".simg",
    ".zip",
}
# At least one file matching each required category must be present, so a
# submission cannot silently omit an entire deliverable.
REQUIRED_CATEGORIES = {
    "answers": lambda name: name == "answers.md",
    "student source": lambda name: name.startswith("src/") and name.endswith(".py"),
    "fixtures": lambda name: name.startswith("data/fixtures/") and name.endswith(".npz"),
    "scenario results": lambda name: name.startswith("results/") and name.endswith(".json"),
    "sweep results": lambda name: name.startswith("results/") and name.endswith(".csv"),
}


def validate_groupid(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "GROUPID may contain letters, digits, underscore, and hyphen"
        )
    return value


def is_allowed(relative: Path) -> bool:
    posix = relative.as_posix()
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return False
    if relative.name in FORBIDDEN_NAMES:
        return False
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if posix in ALLOWED_FILES:
        return True
    return posix.startswith(ALLOWED_PREFIXES) and relative.suffix.lower() in ALLOWED_SUFFIXES


def category_summary(arcnames: Iterable[str]) -> dict[str, int]:
    """Count archive entries per student-visible area for the printed summary."""

    counts: dict[str, int] = {}
    for name in arcnames:
        if name == "answers.md":
            key = "report"
        elif "/" in name:
            key = name.split("/", 1)[0]
        else:
            key = "project files"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_regular_file(path: Path) -> bool:
    """Reject symlinks and any target that resolves outside the repository."""

    if path.is_symlink():
        return False
    try:
        resolved = path.resolve()
        resolved.relative_to(ROOT)
    except (ValueError, OSError):
        return False
    return resolved.is_file()


def collect(destination: Path) -> list[tuple[Path, str]]:
    answer = ROOT / "submission" / "answers.md"
    if not answer.is_file():
        raise SystemExit(
            "submission/answers.md is missing. Copy answers_template.md and complete it."
        )
    paths: list[tuple[Path, str]] = [(answer, "answers.md")]
    for path in sorted(ROOT.rglob("*")):
        if path == answer or path == destination:
            continue
        if path.is_dir():
            continue
        relative = path.relative_to(ROOT)
        if not is_allowed(relative):
            continue
        if not _safe_regular_file(path):
            raise SystemExit(f"refusing unsafe file (symlink or outside repo): {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"refusing oversized file (> {MAX_FILE_BYTES} bytes): {relative}")
        paths.append((path, relative.as_posix()))
    return paths


def check_categories(arcnames: Iterable[str]) -> list[str]:
    names = list(arcnames)
    missing = [
        category
        for category, predicate in REQUIRED_CATEGORIES.items()
        if not any(predicate(name) for name in names)
    ]
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groupid", required=True, type=validate_groupid)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--allow-missing-categories",
        action="store_true",
        help="Instructor override for partial in-progress checks only.",
    )
    args = parser.parse_args(argv)
    destination = ROOT / "submission" / f"group{args.groupid}_safe-agent-actuation.zip"

    files = collect(destination)
    arcnames = [arcname for _, arcname in files]
    if len(arcnames) != len(set(arcnames)):
        raise SystemExit("submission would contain duplicate archive names")

    missing = check_categories(arcnames)
    if missing and not args.allow_missing_categories:
        raise SystemExit(
            "submission is missing required result categories: "
            + ", ".join(missing)
            + ". Run the scenarios and sweep first, or pass --allow-missing-categories."
        )

    if args.check:
        print("\n".join(arcnames))
        if missing:
            print(f"WARNING missing categories: {', '.join(missing)}")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in files:
            archive.write(source, arcname)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        if not names or names[0] != "answers.md" or len(names) != len(set(names)):
            raise SystemExit("submission ZIP validation failed")
    print(f"Wrote {destination} with {len(files)} files")
    for area, count in sorted(category_summary(arcnames).items()):
        print(f"  {area}: {count}")
    print(
        "Inspect this ZIP before Canvas upload. "
        "This ZIP, not a newer GitHub or Sol copy, is graded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

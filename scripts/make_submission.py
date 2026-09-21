#!/usr/bin/env python3
"""Build one group ZIP from a fixed allowlist, independent of Git state.

The builder never inspects Git.  It collects a student-editable allowlist plus
required result categories, refuses symlinks that escape the repository, rejects
oversized files, and never lets the output ZIP overwrite one of its own inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_INPUT_BYTES = 100 * 1024 * 1024

ALLOWED_FILES = {
    "answers.md",
    "README.md",
    "ASSIGNMENT.md",
    "SECURITY.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".gitmodules",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "third_party/README.md",
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
    ".cache",
    ".git",
    ".mypy_cache",
    ".pyright",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-openpi",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "instructor",
    "models",
    "node_modules",
    "runtime",
    "third_party",
    "submission",
}
# Solution-only and instructor-only tools never belong in a student ZIP, even
# though they live in an otherwise allowed directory.
FORBIDDEN_NAMES = {
    "export_student_template.py",
    "validate_student_release.py",
    "capture_pi0_fast_record.py",
    "capture_pi0_fast_record.sh",
    "live_aloha.py",
    "model_setup.sh",
    "run_live.sh",
    "serve_fast_policy.py",
    "setup_openpi.sh",
    "stage_model.py",
    "test_cli.py",
    "test_course_fixtures.py",
    "test_exporter.py",
    "test_live_openpi.py",
    "test_release_validator.py",
    "test_result_tools.py",
    "test_scenarios.py",
    "test_student_release_defects.py",
    "test_sweeps.py",
}
FORBIDDEN_SUFFIXES = {
    ".crt",
    ".csr",
    ".db",
    ".der",
    ".gz",
    ".jks",
    ".key",
    ".keystore",
    ".pem",
    ".pfx",
    ".p12",
    ".sqlite",
    ".sqlite3",
    ".safetensors",
    ".sif",
    ".simg",
    ".tar",
    ".zip",
}
# At least one file matching each required category must be present, so a
# submission cannot silently omit an entire deliverable.
REQUIRED_CATEGORIES = {
    "answers": lambda name: name == "answers.md",
    "student source": lambda name: name.startswith("src/") and name.endswith(".py"),
    "fixtures": lambda name: name.startswith("data/fixtures/") and name.endswith(".npz"),
    "captured response metadata": lambda name: name
    == "data/captured/pi0_fast_action_record.json",
    "captured response chunk": lambda name: name
    == "data/captured/pi0_fast_action_chunk.npz",
    "Part 1 result": lambda name: name == "results/inspect-actions/response_summary.json",
    "low-risk result": lambda name: name == "results/low-risk/action_record.json",
    "allowed result": lambda name: name == "results/high-risk-allow/action_record.json",
    "denied result": lambda name: name == "results/high-risk-deny/action_record.json",
    "bypass result": lambda name: name == "results/bypass/action_record.json",
    "multi-threshold trials": lambda name: name == "results/threshold-sweep/trials.csv",
    "multi-threshold structured trials": lambda name: name
    == "results/threshold-sweep/trials.json",
    "multi-threshold summary": lambda name: name == "results/threshold-sweep/summary.json",
    "multi-threshold plot": lambda name: name
    == "results/threshold-sweep/threshold_sweep.png",
}
REQUIRED_GRAD_CATEGORIES = {
    "Part 5 live trials": lambda name: name
    == "results/grad-key-validity/live_trials.jsonl",
    "Part 5 tabular trials": lambda name: name
    == "results/grad-key-validity/trials.csv",
    "Part 5 summary": lambda name: name == "results/grad-key-validity/summary.json",
    "Part 5 plot": lambda name: name == "results/grad-key-validity/key_validity.png",
}

SECRET_FILENAMES = {
    "auth_password",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_TEXT_MARKERS = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN " + "RSA PRIVATE KEY-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    "ISC" + "PS-INSTRUCTOR-SOLUTION",
    "ISC" + "PS-INSTRUCTOR-ONLY-BEGIN",
    "ISC" + "PS-GRADED-BEGIN",
    "ISC" + "PS-GRADED-END",
)
SENSITIVE_RESULT_KEY = re.compile(
    r'"(?:cipher_key|distribution_key|mac_key|password|private_key|session_key|secret)"\s*:',
    re.IGNORECASE,
)


def validate_groupid(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "GROUPID may contain letters, digits, underscore, and hyphen"
        )
    return value


def is_allowed(relative: Path) -> bool:
    posix = relative.as_posix()
    if relative.name in FORBIDDEN_NAMES:
        return False
    if relative.name.lower() in SECRET_FILENAMES:
        return False
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if posix in ALLOWED_FILES:
        return True
    if any(part in FORBIDDEN_PARTS or part.startswith(".venv") for part in relative.parts):
        return False
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


def _check_file_contents(path: Path, relative: Path) -> None:
    """Reject known solution markers and secret material from allowed text files."""

    if relative.suffix.lower() not in {
        "",
        ".cfg",
        ".csv",
        ".def",
        ".graph",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".sbatch",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SystemExit(f"refusing non-UTF-8 text file: {relative}") from None
    for marker in FORBIDDEN_TEXT_MARKERS:
        if marker in text:
            raise SystemExit(f"refusing instructor/secret marker in {relative}: {marker}")
    if relative.parts and relative.parts[0] == "results" and relative.suffix in {".json", ".jsonl"}:
        if SENSITIVE_RESULT_KEY.search(text):
            raise SystemExit(f"refusing secret-bearing result field in {relative}")
        # Valid JSON files receive a structural pass too. JSONL is validated by
        # the result tool and still gets the conservative key scan above.
        if relative.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"refusing malformed JSON result {relative}: {exc}") from None


def collect(destination: Path) -> list[tuple[Path, str]]:
    answer = ROOT / "submission" / "answers.md"
    if not answer.is_file():
        raise SystemExit(
            "submission/answers.md is missing. Copy answers_template.md and complete it."
        )
    if not _safe_regular_file(answer):
        raise SystemExit("refusing unsafe submission/answers.md (symlink or outside repo)")
    if answer.stat().st_size > MAX_FILE_BYTES:
        raise SystemExit("refusing oversized submission/answers.md")
    _check_file_contents(answer, Path("answers.md"))
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
        _check_file_contents(path, relative)
        paths.append((path, relative.as_posix()))
    total_bytes = sum(source.stat().st_size for source, _ in paths)
    if total_bytes > MAX_ARCHIVE_INPUT_BYTES:
        raise SystemExit(
            f"refusing oversized submission inputs (> {MAX_ARCHIVE_INPUT_BYTES} bytes total)"
        )
    return paths


def check_categories(arcnames: Iterable[str], *, require_grad: bool = False) -> list[str]:
    names = list(arcnames)
    categories = dict(REQUIRED_CATEGORIES)
    if require_grad:
        categories.update(REQUIRED_GRAD_CATEGORIES)
    missing = [
        category
        for category, predicate in categories.items()
        if not any(predicate(name) for name in names)
    ]
    return missing


def validate_archive(path: Path, expected_names: list[str]) -> None:
    """Verify the finished ZIP before replacing the prior submission."""

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != expected_names or not names or names[0] != "answers.md":
            raise SystemExit("submission ZIP entry validation failed")
        if len(names) != len(set(names)):
            raise SystemExit("submission ZIP contains duplicate names")
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or info.is_dir():
                raise SystemExit(f"submission ZIP contains an unsafe entry: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise SystemExit(f"submission ZIP contains a symlink: {info.filename}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise SystemExit(f"submission ZIP CRC check failed: {corrupt}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groupid", required=True, type=validate_groupid)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--require-grad",
        action="store_true",
        help="Require all real-SST CSE 598 Part 5 artifacts in the ZIP.",
    )
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

    missing = check_categories(arcnames, require_grad=args.require_grad)
    if missing and not args.allow_missing_categories:
        raise SystemExit(
            "submission is missing required result categories: "
            + ", ".join(missing)
            + ". Run the required scenarios and evaluations first, or pass "
            "--allow-missing-categories."
        )

    if args.check:
        print("\n".join(arcnames))
        if missing:
            print(f"WARNING missing categories: {', '.join(missing)}")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, arcname in files:
                archive.write(source, arcname)
        validate_archive(temporary_path, arcnames)
        temporary_path.replace(destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
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

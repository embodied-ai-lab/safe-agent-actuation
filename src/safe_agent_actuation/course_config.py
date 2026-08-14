"""Locate the repository root and load committed course configuration.

Scenario commands must behave identically on Sol, in the container, and in a
local checkout.  Configuration therefore always comes from committed JSON
files resolved against one explicit repository root, never from ambient state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .context import AuthorizationContext, ContextPolicy
from .types import PhysicalLimits

ROOT_ENVIRONMENT_VARIABLE = "SAFE_ACTUATION_ROOT"

LIMITS_CONFIG = Path("configs/motion/aloha_limits.json")
ALLOWED_CONTEXT_CONFIG = Path("configs/context/allowed.json")
DENIED_CONTEXT_CONFIG = Path("configs/context/denied.json")
SCENARIOS_CONFIG = Path("configs/scenarios/scenarios.json")
FIXTURES_DIR = Path("data/fixtures")
RESULTS_DIR = Path("results")
SST_RUNTIME_DIR = Path("runtime/sst")


class CourseConfigError(RuntimeError):
    """Raised when required committed configuration cannot be loaded."""


def _looks_like_root(candidate: Path) -> bool:
    return (candidate / "configs" / "motion" / "aloha_limits.json").is_file()


def repository_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the course repository root deterministically.

    Precedence: an explicit argument, then ``SAFE_ACTUATION_ROOT`` (exported by
    the Makefile), then the package checkout location, then the working
    directory.  The chosen directory must contain the committed course
    configuration.
    """

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        from_environment = os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
        if from_environment:
            candidates.append(Path(from_environment))
        candidates.append(Path(__file__).resolve().parents[2])
        candidates.append(Path.cwd())
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _looks_like_root(resolved):
            return resolved
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise CourseConfigError(
        "could not locate the course repository root "
        f"(missing {LIMITS_CONFIG}); attempted: {attempted}"
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CourseConfigError(f"required configuration is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CourseConfigError(f"could not parse configuration {path}: {exc}") from exc


def load_physical_limits(root: Path) -> PhysicalLimits:
    """Load the committed ALOHA physical limits used by every scenario."""

    raw = _load_json(root / LIMITS_CONFIG)
    if not isinstance(raw, Mapping):
        raise CourseConfigError(f"{LIMITS_CONFIG} must contain a JSON object")
    try:
        return PhysicalLimits(
            lower=raw["lower_limits"],
            upper=raw["upper_limits"],
            dimension_names=tuple(raw["dimension_names"]),
            source=str(raw["source"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CourseConfigError(f"invalid physical limits in {LIMITS_CONFIG}: {exc}") from exc


def default_threshold(root: Path) -> float:
    raw = _load_json(root / LIMITS_CONFIG)
    try:
        value = float(raw["default_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CourseConfigError(f"invalid default_threshold in {LIMITS_CONFIG}") from exc
    return value


def load_context(root: Path, name: str) -> AuthorizationContext:
    """Load one of the committed scenario contexts (``allowed`` or ``denied``)."""

    paths = {
        "allowed": ALLOWED_CONTEXT_CONFIG,
        "denied": DENIED_CONTEXT_CONFIG,
    }
    if name not in paths:
        raise CourseConfigError(f"unknown context name {name!r}; expected allowed or denied")
    raw = _load_json(root / paths[name])
    if not isinstance(raw, Mapping):
        raise CourseConfigError(f"{paths[name]} must contain a JSON object")
    try:
        return AuthorizationContext.from_mapping(raw)
    except (TypeError, ValueError) as exc:
        raise CourseConfigError(f"invalid context in {paths[name]}: {exc}") from exc


def course_policy() -> ContextPolicy:
    """Return the local mirror of the registered SST communication policy."""

    return ContextPolicy.course_default()


def scenario_settings(root: Path, scenario: str) -> Mapping[str, Any]:
    raw = _load_json(root / SCENARIOS_CONFIG)
    if not isinstance(raw, Mapping) or scenario not in raw:
        raise CourseConfigError(f"scenario {scenario!r} is not declared in {SCENARIOS_CONFIG}")
    settings = raw[scenario]
    if not isinstance(settings, Mapping):
        raise CourseConfigError(f"scenario {scenario!r} settings must be a JSON object")
    return settings


def fixture_path(root: Path, filename: str) -> Path:
    """Resolve a committed fixture path and fail closed when it is absent."""

    if not filename or Path(filename).name != filename:
        raise CourseConfigError(f"fixture filename must be a bare file name: {filename!r}")
    path = root / FIXTURES_DIR / filename
    if not path.is_file():
        raise CourseConfigError(
            f"fixture is missing: {path}. Regenerate with scripts/generate_fixtures.py"
        )
    return path

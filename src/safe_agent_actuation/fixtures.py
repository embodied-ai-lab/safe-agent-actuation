"""Deterministic, explicitly labeled NumPy action fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .types import PhysicalLimits


class FixtureSource(StrEnum):
    """Honest provenance categories required by the assignment."""

    CAPTURED_LIVE_PI0_FAST = "captured_live_pi0_fast"
    GENERATED_BOUNDARY_TEST = "generated_boundary_test"
    DIAGNOSTIC_SYNTHETIC = "diagnostic_synthetic"


_SOURCE_ALIASES = {
    "captured_live_pi0_fast": FixtureSource.CAPTURED_LIVE_PI0_FAST,
    "captured from a live pi0-fast run": FixtureSource.CAPTURED_LIVE_PI0_FAST,
    "captured from a live π0-fast run": FixtureSource.CAPTURED_LIVE_PI0_FAST,
    "generated_boundary_test": FixtureSource.GENERATED_BOUNDARY_TEST,
    "generated for a boundary test": FixtureSource.GENERATED_BOUNDARY_TEST,
    "diagnostic_synthetic": FixtureSource.DIAGNOSTIC_SYNTHETIC,
    "diagnostic synthetic action chunk": FixtureSource.DIAGNOSTIC_SYNTHETIC,
}


def _source(value: object) -> FixtureSource:
    if isinstance(value, FixtureSource):
        return value
    if not isinstance(value, str):
        raise ValueError("fixture source label is required")
    normalized = value.strip().lower()
    try:
        return _SOURCE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(item.value for item in FixtureSource)
        raise ValueError(f"unknown fixture source label; expected one of: {allowed}") from exc


def _action(value: ArrayLike) -> np.ndarray:
    try:
        action = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fixture action_chunk must contain numeric values") from exc
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] == 0:
        raise ValueError("fixture action_chunk must have non-empty shape [H, D]")
    if not np.all(np.isfinite(action)):
        raise ValueError("fixture action_chunk contains NaN or infinity")
    result = np.ascontiguousarray(action, dtype=np.float64)
    result.setflags(write=False)
    return result


def _state(value: ArrayLike, dimensions: int) -> np.ndarray:
    try:
        state = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fixture current_state must contain numeric values") from exc
    if state.shape != (dimensions,) or not np.all(np.isfinite(state)):
        raise ValueError(f"fixture current_state must be a finite [{dimensions}] vector")
    result = np.array(state, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _tokens(value: ArrayLike | Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("raw_fast_token_ids must be a one-dimensional integer array")
    return tuple(int(item) for item in array)


def _metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("fixture metadata must be a mapping")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("fixture metadata must contain only finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise ValueError("fixture metadata must be a JSON object")
    return decoded


@dataclass(frozen=True, slots=True)
class ActionFixture:
    """A decoded action and all inputs required for controlled analysis."""

    action_chunk: ArrayLike = field(repr=False)
    current_state: ArrayLike = field(repr=False)
    limits: PhysicalLimits
    source: FixtureSource | str
    description: str
    raw_fast_token_ids: Sequence[int] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PhysicalLimits):
            raise ValueError("limits must be PhysicalLimits")
        action = _action(self.action_chunk)
        state = _state(self.current_state, action.shape[1])
        if action.shape[1] != self.limits.dimensions:
            raise ValueError("fixture action dimension does not match physical limits")
        if np.any(action < self.limits.lower) or np.any(action > self.limits.upper):
            raise ValueError("fixture action is outside configured physical limits")
        if np.any(state < self.limits.lower) or np.any(state > self.limits.upper):
            raise ValueError("fixture current_state is outside configured physical limits")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("fixture description must be a non-empty string")
        object.__setattr__(self, "action_chunk", action)
        object.__setattr__(self, "current_state", state)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "raw_fast_token_ids", _tokens(self.raw_fast_token_ids))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def source_label(self) -> str:
        return self.source.value  # type: ignore[union-attr]

    @property
    def action(self) -> np.ndarray:
        # ``__post_init__`` normalizes the public ArrayLike constructor input.
        return cast(np.ndarray, self.action_chunk)

    @property
    def tokens(self) -> tuple[int, ...] | None:
        return self.raw_fast_token_ids  # type: ignore[return-value]

    @property
    def lower_limits(self) -> np.ndarray:
        return self.limits.lower  # type: ignore[return-value]

    @property
    def upper_limits(self) -> np.ndarray:
        return self.limits.upper  # type: ignore[return-value]

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return self.limits.dimension_names

    @property
    def physical_limits_source(self) -> str:
        return self.limits.source


def _scalar_text(array: np.ndarray, field_name: str) -> str:
    if array.shape not in {(), (1,)} or array.dtype.kind not in "US":
        raise ValueError(f"fixture field {field_name!r} must be one text scalar")
    value = str(array.reshape(-1)[0])
    if not value:
        raise ValueError(f"fixture field {field_name!r} must not be empty")
    return value


def _select(files: set[str], alternatives: tuple[str, ...], label: str) -> str:
    found = [name for name in alternatives if name in files]
    if len(found) != 1:
        raise ValueError(f"fixture requires exactly one {label} field")
    return found[0]


def load_action_fixture(path: str | Path) -> ActionFixture:
    """Load an NPZ fixture without pickle or dependence on archive key order."""

    fixture_path = Path(path)
    if fixture_path.suffix.lower() != ".npz":
        raise ValueError("action fixture must use the .npz format")
    try:
        archive_context = np.load(fixture_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load action fixture: {fixture_path}") from exc
    with archive_context as archive:
        files = set(archive.files)
        action_key = _select(
            files, ("action_chunk", "decoded_action_chunk", "actions"), "action chunk"
        )
        state_key = _select(files, ("current_state", "observed_state"), "current state")
        lower_key = _select(files, ("lower_limits", "lower"), "lower limits")
        upper_key = _select(files, ("upper_limits", "upper"), "upper limits")
        names_key = _select(files, ("dimension_names", "joint_names"), "dimension names")
        source_key = _select(files, ("source_label", "source"), "source label")
        description_key = _select(files, ("description",), "description")
        limits_source_key = _select(
            files, ("physical_limits_source", "limits_source"), "physical-limits source"
        )

        names_array = np.asarray(archive[names_key])
        if names_array.ndim != 1 or names_array.dtype.kind not in "US" or names_array.size == 0:
            raise ValueError("dimension_names must be a non-empty text vector")
        names = tuple(str(item) for item in names_array)
        limits = PhysicalLimits(
            lower=np.array(archive[lower_key], copy=True),
            upper=np.array(archive[upper_key], copy=True),
            dimension_names=names,
            source=_scalar_text(np.asarray(archive[limits_source_key]), limits_source_key),
        )
        token_key = next(
            (name for name in ("raw_fast_token_ids", "fast_token_ids", "tokens") if name in files),
            None,
        )
        tokens = np.array(archive[token_key], copy=True) if token_key is not None else None
        metadata: dict[str, Any] = {}
        if "metadata_json" in files:
            raw_metadata = _scalar_text(np.asarray(archive["metadata_json"]), "metadata_json")
            try:
                parsed_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise ValueError("fixture metadata_json is malformed") from exc
            if not isinstance(parsed_metadata, dict):
                raise ValueError("fixture metadata_json must contain an object")
            metadata = parsed_metadata
        return ActionFixture(
            action_chunk=np.array(archive[action_key], copy=True),
            current_state=np.array(archive[state_key], copy=True),
            limits=limits,
            source=_scalar_text(np.asarray(archive[source_key]), source_key),
            description=_scalar_text(np.asarray(archive[description_key]), description_key),
            raw_fast_token_ids=_tokens(tokens),
            metadata=metadata,
        )


load_fixture = load_action_fixture


def save_action_fixture(fixture: ActionFixture, path: str | Path) -> Path:
    """Save the stable fixture schema consumed by :func:`load_action_fixture`."""

    if not isinstance(fixture, ActionFixture):
        raise TypeError("fixture must be an ActionFixture")
    output = Path(path)
    if output.suffix.lower() != ".npz":
        raise ValueError("action fixture path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "action_chunk": np.asarray(fixture.action_chunk, dtype="<f8"),
        "current_state": np.asarray(fixture.current_state, dtype="<f8"),
        "lower_limits": np.asarray(fixture.limits.lower, dtype="<f8"),
        "upper_limits": np.asarray(fixture.limits.upper, dtype="<f8"),
        "dimension_names": np.asarray(fixture.dimension_names, dtype=np.str_),
        "source_label": np.asarray(fixture.source_label, dtype=np.str_),
        "description": np.asarray(fixture.description, dtype=np.str_),
        "physical_limits_source": np.asarray(fixture.physical_limits_source, dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(
                fixture.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            dtype=np.str_,
        ),
    }
    if fixture.raw_fast_token_ids is not None:
        payload["raw_fast_token_ids"] = np.asarray(fixture.raw_fast_token_ids, dtype="<i8")
    np.savez_compressed(output, **payload)
    return output


write_action_fixture = save_action_fixture

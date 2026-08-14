"""Shared, dependency-light types for safe action monitoring.

The classes in this module deliberately contain no key material and avoid
including large action arrays in their representations.  They are suitable for
records and for the boundaries between the monitor and simulated actuator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class RiskClassification(StrEnum):
    """The threshold gate's three possible outcomes."""

    LOW = "low"
    HIGH = "high"
    INVALID = "invalid"


class ActuatorDecision(StrEnum):
    """Whether the simulated actuator executed or held its pose."""

    EXECUTE = "execute"
    HOLD = "hold"


class AuthorizationOutcome(StrEnum):
    """Result of a context-based authorization request."""

    ALLOWED = "allowed"
    DENIED = "denied"
    NOT_REQUESTED = "not_requested"
    ERROR = "error"


def _float_vector(value: ArrayLike, *, field_name: str) -> FloatArray:
    """Copy *value* into an immutable, finite one-dimensional float array."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a numeric vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{field_name} must have shape [D]")
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PhysicalLimits:
    """Named physical limits for every command dimension.

    Different dimensions may use different physical units.  Normalization by
    each dimension's own range is therefore required before their motion scores
    can be compared.
    """

    lower: ArrayLike = field(repr=False)
    upper: ArrayLike = field(repr=False)
    dimension_names: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        lower = _float_vector(self.lower, field_name="lower")
        upper = _float_vector(self.upper, field_name="upper")
        names = tuple(self.dimension_names)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper limits must have the same shape")
        if len(names) != lower.size:
            raise ValueError("dimension_names must contain one name per dimension")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("dimension_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("dimension_names must be unique")
        if np.any(upper <= lower):
            raise ValueError("every physical range must be finite and greater than zero")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("physical limit source must be a non-empty string")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "dimension_names", names)
        object.__setattr__(self, "source", self.source.strip())

    @property
    def ranges(self) -> FloatArray:
        ranges = np.asarray(self.upper) - np.asarray(self.lower)
        ranges.setflags(write=False)
        return ranges

    @property
    def dimensions(self) -> int:
        return len(self.dimension_names)


# The more explicit name is preferred, but JointLimits is convenient in course
# code and retained as a public alias.
JointLimits = PhysicalLimits


@dataclass(frozen=True, slots=True)
class MotionAnalysis:
    """Fail-closed output from the Joint Motion Analyzer."""

    valid: bool
    classification: RiskClassification
    motion_score: float
    threshold: float
    dimension_names: tuple[str, ...]
    intra: tuple[float, ...]
    inter: tuple[float, ...]
    max_dimension_index: int | None
    max_dimension: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        names = tuple(self.dimension_names)
        intra = tuple(float(value) for value in self.intra)
        inter = tuple(float(value) for value in self.inter)
        object.__setattr__(self, "dimension_names", names)
        object.__setattr__(self, "intra", intra)
        object.__setattr__(self, "inter", inter)
        if self.valid:
            if self.classification is RiskClassification.INVALID:
                raise ValueError("a valid analysis cannot be classified invalid")
            if not np.isfinite(self.motion_score) or self.motion_score < 0:
                raise ValueError("a valid motion score must be finite and non-negative")
            if len(names) == 0 or len(intra) != len(names) or len(inter) != len(names):
                raise ValueError("valid per-dimension scores must match dimension_names")
            if self.max_dimension_index is None or not 0 <= self.max_dimension_index < len(names):
                raise ValueError("valid analysis requires a maximum dimension index")
            if self.max_dimension != names[self.max_dimension_index]:
                raise ValueError("max_dimension must match max_dimension_index")
            if self.reason is not None:
                raise ValueError("valid analysis must not include a rejection reason")
        else:
            if self.classification is not RiskClassification.INVALID:
                raise ValueError("an invalid analysis must be classified invalid")
            if not self.reason:
                raise ValueError("invalid analysis requires a rejection reason")

    @property
    def high_risk(self) -> bool:
        """Return true only for a valid action at or above the threshold."""

        return self.valid and self.classification is RiskClassification.HIGH

    @property
    def is_high_risk(self) -> bool:
        return self.high_risk

    @property
    def is_low_risk(self) -> bool:
        return self.valid and self.classification is RiskClassification.LOW

    @property
    def requires_authorization(self) -> bool:
        return self.high_risk

    @property
    def blocked(self) -> bool:
        return not self.valid

    @property
    def risk_classification(self) -> str:
        return self.classification.value

    @property
    def intra_scores(self) -> tuple[float, ...]:
        return self.intra

    @property
    def inter_scores(self) -> tuple[float, ...]:
        return self.inter

    def per_dimension(self) -> dict[str, dict[str, float]]:
        return {
            name: {"intra": self.intra[index], "inter": self.inter[index]}
            for index, name in enumerate(self.dimension_names)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "classification": self.classification.value,
            "motion_score": self.motion_score,
            "threshold": self.threshold,
            "dimension_names": list(self.dimension_names),
            "intra": list(self.intra),
            "inter": list(self.inter),
            "max_dimension_index": self.max_dimension_index,
            "max_dimension": self.max_dimension,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """A secret-free authorization result suitable for action records."""

    outcome: AuthorizationOutcome
    reason: str
    key_id: str | None = None
    session_key_reused: bool = False
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("authorization reason must be a non-empty string")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("authorization latency must be finite and non-negative")
        if self.outcome is not AuthorizationOutcome.ALLOWED and self.key_id is not None:
            raise ValueError("a denied or failed authorization cannot return a key ID")

    @property
    def allowed(self) -> bool:
        return self.outcome is AuthorizationOutcome.ALLOWED

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "key_id": self.key_id,
            "session_key_reused": self.session_key_reused,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class ActuationResult:
    """Outcome returned by the actuator enforcement boundary."""

    decision: ActuatorDecision
    reason: str
    path: str
    action_shape: tuple[int, int] | None = None
    decryption_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.reason or not self.path:
            raise ValueError("actuation result requires a reason and path")
        for value in (self.decryption_latency_ms, self.total_latency_ms):
            if not np.isfinite(value) or value < 0:
                raise ValueError("latencies must be finite and non-negative")

    @property
    def executed(self) -> bool:
        return self.decision is ActuatorDecision.EXECUTE

    @property
    def held(self) -> bool:
        return self.decision is ActuatorDecision.HOLD

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "path": self.path,
            "action_shape": list(self.action_shape) if self.action_shape else None,
            "decryption_latency_ms": self.decryption_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }

"""Concise, secret-free JSON action records and latency measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .context import AuthorizationContext
from .types import ActuationResult, AuthorizationDecision, MotionAnalysis

_SECRET_FIELD_FRAGMENTS = (
    "secret",
    "private_key",
    "cipher_key",
    "mac_key",
    "key_material",
    "password",
    "credential",
)


def _latency(value: float | int | None, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class LatencyRecord:
    """Timing schema shared by live and recorded-action scenarios."""

    inference_ms: float | None = None
    decode_ms: float | None = None
    monitoring_ms: float = 0.0
    authorization_ms: float | None = None
    encryption_ms: float | None = None
    decryption_ms: float | None = None
    total_monitor_to_actuator_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "inference_ms",
            "decode_ms",
            "monitoring_ms",
            "authorization_ms",
            "encryption_ms",
            "decryption_ms",
            "total_monitor_to_actuator_ms",
        ):
            object.__setattr__(self, name, _latency(getattr(self, name), name))

    @property
    def monitoring_latency_ms(self) -> float:
        return self.monitoring_ms

    @property
    def authorization_latency_ms(self) -> float | None:
        return self.authorization_ms

    @property
    def encryption_latency_ms(self) -> float | None:
        return self.encryption_ms

    @property
    def decryption_latency_ms(self) -> float | None:
        return self.decryption_ms

    @property
    def total_monitor_to_actuator_latency_ms(self) -> float:
        return self.total_monitor_to_actuator_ms

    def to_dict(self) -> dict[str, float | None]:
        return {
            "inference_ms": self.inference_ms,
            "decode_ms": self.decode_ms,
            "monitoring_ms": self.monitoring_ms,
            "authorization_ms": self.authorization_ms,
            "encryption_ms": self.encryption_ms,
            "decryption_ms": self.decryption_ms,
            "total_monitor_to_actuator_ms": self.total_monitor_to_actuator_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LatencyRecord:
        if not isinstance(value, Mapping):
            raise ValueError("latencies must be a mapping")
        allowed = {
            "inference_ms",
            "decode_ms",
            "monitoring_ms",
            "authorization_ms",
            "encryption_ms",
            "decryption_ms",
            "total_monitor_to_actuator_ms",
        }
        if set(value) - allowed:
            raise ValueError("latencies contain unknown fields")
        return cls(**value)  # type: ignore[arg-type]


def _clean_tokens(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError("raw_fast_token_ids must be a sequence of integers")
    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise ValueError("raw_fast_token_ids must contain integers")
        tokens.append(int(token))
    return tuple(tokens)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("records cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if any(fragment in normalized for fragment in _SECRET_FIELD_FRAGMENTS):
                raise ValueError(f"secret-bearing field {key!r} must not be written to records")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    raise ValueError(f"record value of type {type(value).__name__} is not JSON-safe")


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One policy response, monitoring decision, and actuator outcome.

    Large decoded arrays belong in ``action_npz_path``; the JSON contains their
    shape and concise metrics.  ``raw_fast_token_ids`` are observability data,
    not physical motion measurements.
    """

    run_id: str
    timestep: int
    task_prompt: str
    model_configuration: str
    action_source: str
    raw_fast_token_ids: Sequence[int] | None = None
    decoded_action_shape: tuple[int, int] | None = None
    action_npz_path: str | None = None
    executed_horizon: int | None = None
    current_observed_joint_state: Sequence[float] | None = None
    physical_limits_source: str | None = None
    motion_analysis: MotionAnalysis | None = None
    context: AuthorizationContext | Mapping[str, object] | None = None
    authorization_requested: bool = False
    authorization: AuthorizationDecision | None = None
    latencies: LatencyRecord = field(default_factory=LatencyRecord)
    actuation: ActuationResult | None = None
    simulator_outcome: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "run_id"),
            (self.task_prompt, "task_prompt"),
            (self.model_configuration, "model_configuration"),
            (self.action_source, "action_source"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if (
            isinstance(self.timestep, bool)
            or not isinstance(self.timestep, int)
            or self.timestep < 0
        ):
            raise ValueError("timestep must be a non-negative integer")
        tokens = _clean_tokens(self.raw_fast_token_ids)
        object.__setattr__(self, "raw_fast_token_ids", tokens)
        if self.decoded_action_shape is not None:
            if (
                not isinstance(self.decoded_action_shape, tuple)
                or len(self.decoded_action_shape) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in self.decoded_action_shape
                )
            ):
                raise ValueError("decoded_action_shape must be a non-empty [H, D] tuple")
        if self.executed_horizon is not None:
            if (
                isinstance(self.executed_horizon, bool)
                or not isinstance(self.executed_horizon, int)
                or self.executed_horizon <= 0
            ):
                raise ValueError("executed_horizon must be a positive integer")
            if (
                self.decoded_action_shape is not None
                and self.executed_horizon > self.decoded_action_shape[0]
            ):
                raise ValueError("executed_horizon cannot exceed the predicted horizon")
        if self.current_observed_joint_state is not None:
            state = tuple(float(value) for value in self.current_observed_joint_state)
            if not state or not np.all(np.isfinite(state)):
                raise ValueError("current_observed_joint_state must be a finite vector")
            if self.decoded_action_shape is not None and len(state) != self.decoded_action_shape[1]:
                raise ValueError("current state dimension must match decoded action dimension")
            object.__setattr__(self, "current_observed_joint_state", state)
        if isinstance(self.context, Mapping) and not isinstance(self.context, AuthorizationContext):
            object.__setattr__(self, "context", AuthorizationContext.from_mapping(self.context))
        if self.authorization_requested and self.authorization is None:
            raise ValueError("authorization_requested requires an authorization result")
        if not isinstance(self.latencies, LatencyRecord):
            raise ValueError("latencies must be a LatencyRecord")
        object.__setattr__(self, "metadata", _json_safe(self.metadata))

    @property
    def token_count(self) -> int | None:
        return None if self.raw_fast_token_ids is None else len(self.raw_fast_token_ids)

    def to_dict(self) -> dict[str, Any]:
        context = self.context.to_dict() if isinstance(self.context, AuthorizationContext) else None
        motion = self.motion_analysis.to_dict() if self.motion_analysis is not None else None
        authorization = self.authorization.to_dict() if self.authorization is not None else None
        actuation = dict(self.actuation.to_dict()) if self.actuation is not None else None
        result = {
            "schema_version": 1,
            "run_id": self.run_id,
            "timestep": self.timestep,
            "task_prompt": self.task_prompt,
            "model_configuration": self.model_configuration,
            "action_source": self.action_source,
            "raw_fast_token_ids": list(self.raw_fast_token_ids)
            if self.raw_fast_token_ids is not None
            else None,
            "token_count": self.token_count,
            "decoded_action_shape": list(self.decoded_action_shape)
            if self.decoded_action_shape is not None
            else None,
            "action_npz_path": self.action_npz_path,
            "executed_horizon": self.executed_horizon,
            "current_observed_joint_state": list(self.current_observed_joint_state)
            if self.current_observed_joint_state is not None
            else None,
            "physical_limits_source": self.physical_limits_source,
            "motion_analysis": motion,
            "context": context,
            "authorization_requested": self.authorization_requested,
            "authorization": authorization,
            "latencies": self.latencies.to_dict(),
            "actuation": actuation,
            "simulator_outcome": self.simulator_outcome,
            "metadata": self.metadata,
        }
        return _json_safe(result)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False) + "\n"


def write_action_record(
    record: ActionRecord,
    json_path: str | Path,
    *,
    action_chunk: ArrayLike | None = None,
    npz_path: str | Path | None = None,
) -> Path:
    """Write a JSON record and, when supplied, one compact action array.

    The function records a relative NPZ path when it can.  No secret material is
    accepted by :class:`ActionRecord` metadata or emitted here.
    """

    if not isinstance(record, ActionRecord):
        raise TypeError("record must be an ActionRecord")
    output = Path(json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written_record = record
    if action_chunk is not None:
        try:
            action = np.asarray(action_chunk, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("action_chunk must contain numeric values") from exc
        if action.ndim != 2 or 0 in action.shape or not np.all(np.isfinite(action)):
            raise ValueError("action_chunk must have finite non-empty shape [H, D]")
        array_path = Path(npz_path) if npz_path is not None else output.with_suffix(".npz")
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, action_chunk=np.asarray(action, dtype="<f8"))
        try:
            stored_path = str(array_path.relative_to(output.parent))
        except ValueError:
            stored_path = str(array_path)
        if record.decoded_action_shape not in (None, action.shape):
            raise ValueError("record decoded_action_shape does not match action_chunk")
        written_record = replace(
            record,
            decoded_action_shape=action.shape,  # type: ignore[arg-type]
            action_npz_path=stored_path,
        )
    output.write_text(written_record.to_json(), encoding="utf-8")
    return output


save_action_record = write_action_record

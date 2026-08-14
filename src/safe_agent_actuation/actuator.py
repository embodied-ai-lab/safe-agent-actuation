"""Trusted actuator-side verification and fail-closed simulator enforcement."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .protection import (
    AESGCMProtector,
    LowRiskEnvelope,
    ProtectedActionEnvelope,
    ProtectionError,
    TrustedLowRiskChannel,
)
from .types import ActuationResult, ActuatorDecision


class SimulatorActuator(Protocol):
    """Narrow adapter boundary implemented by the headless ALOHA client."""

    def execute(self, action_chunk: np.ndarray) -> object: ...

    def hold(self, current_pose: np.ndarray) -> object: ...


def _current_pose(value: ArrayLike) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("current pose must contain numeric values") from exc
    if pose.ndim != 1 or pose.size == 0:
        raise ValueError("current pose must have non-empty shape [D]")
    if not np.all(np.isfinite(pose)):
        raise ValueError("current pose contains NaN or infinity")
    result = np.array(pose, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


class SafeActuator:
    """Enforce trusted low-risk or authenticated high-risk paths only.

    Any raw command, malformed input, authentication failure, expired key, or
    simulator error results in a request to hold the current commanded pose.
    Verification completes before ``simulator.execute`` is called.
    """

    def __init__(
        self,
        simulator: SimulatorActuator,
        *,
        high_risk_protector: AESGCMProtector,
        low_risk_channel: TrustedLowRiskChannel,
    ) -> None:
        if not callable(getattr(simulator, "execute", None)):
            raise TypeError("simulator must provide execute(action_chunk)")
        if not callable(getattr(simulator, "hold", None)):
            raise TypeError("simulator must provide hold(current_pose)")
        self.simulator = simulator
        self.high_risk_protector = high_risk_protector
        self.low_risk_channel = low_risk_channel

    def actuate(
        self,
        envelope: LowRiskEnvelope | ProtectedActionEnvelope | Mapping[str, object] | object,
        current_state: ArrayLike,
    ) -> ActuationResult:
        started = time.perf_counter_ns()
        try:
            pose = _current_pose(current_state)
        except ValueError as exc:
            # There is no validated pose to send.  Still request a hold with an
            # empty sentinel; a real adapter should preserve its last setpoint.
            return self._reject(
                np.empty(0, dtype=np.float64),
                reason=f"invalid current pose: {exc}",
                path="rejected",
                started=started,
            )

        decryption_latency_ms = 0.0
        if isinstance(envelope, LowRiskEnvelope):
            path = "trusted_low_risk"
            try:
                action = self.low_risk_channel.verify(envelope)
            except (ProtectionError, TypeError, ValueError):
                return self._reject(
                    pose,
                    reason="low-risk envelope verification failed",
                    path=path,
                    started=started,
                )
        else:
            path = "sst_protected_high_risk"
            try:
                protected = (
                    ProtectedActionEnvelope.from_dict(envelope)
                    if isinstance(envelope, Mapping)
                    else envelope
                )
                decrypt_started = time.perf_counter_ns()
                action = self.high_risk_protector.unprotect(protected)  # type: ignore[arg-type]
                decryption_latency_ms = (
                    time.perf_counter_ns() - decrypt_started
                ) / 1_000_000
            except (ProtectionError, TypeError, ValueError):
                return self._reject(
                    pose,
                    reason="protected high-risk envelope verification failed",
                    path=(
                        path
                        if isinstance(envelope, (ProtectedActionEnvelope, Mapping))
                        else "bypass"
                    ),
                    started=started,
                )

        if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != pose.size:
            return self._reject(
                pose,
                reason="verified action dimension does not match current pose",
                path=path,
                started=started,
                decryption_latency_ms=decryption_latency_ms,
            )
        if not np.all(np.isfinite(action)):
            return self._reject(
                pose,
                reason="verified action contains NaN or infinity",
                path=path,
                started=started,
                decryption_latency_ms=decryption_latency_ms,
            )

        try:
            self.simulator.execute(np.array(action, dtype=np.float64, copy=True))
        except Exception:  # noqa: BLE001 - adapter failures must always fail closed.
            return self._reject(
                pose,
                reason="simulator rejected the verified action",
                path=path,
                started=started,
                decryption_latency_ms=decryption_latency_ms,
            )
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        return ActuationResult(
            decision=ActuatorDecision.EXECUTE,
            reason="verified action forwarded to the simulator",
            path=path,
            action_shape=action.shape,  # type: ignore[arg-type]
            decryption_latency_ms=decryption_latency_ms,
            total_latency_ms=total_ms,
        )

    enforce = actuate
    receive = actuate

    def _reject(
        self,
        pose: np.ndarray,
        *,
        reason: str,
        path: str,
        started: int,
        decryption_latency_ms: float = 0.0,
    ) -> ActuationResult:
        try:
            self.simulator.hold(np.array(pose, dtype=np.float64, copy=True))
        except Exception:  # noqa: BLE001 - decision remains fail-closed on adapter failure.
            reason = f"{reason}; simulator hold request failed"
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        return ActuationResult(
            decision=ActuatorDecision.HOLD,
            reason=reason,
            path=path,
            action_shape=None,
            decryption_latency_ms=decryption_latency_ms,
            total_latency_ms=total_ms,
        )


Actuator = SafeActuator


@dataclass(slots=True)
class RecordingSimulator:
    """Deterministic simulator spy used by offline safety tests."""

    executed_actions: list[np.ndarray] = field(default_factory=list)
    held_poses: list[np.ndarray] = field(default_factory=list)

    def execute(self, action_chunk: np.ndarray) -> None:
        self.executed_actions.append(np.array(action_chunk, dtype=np.float64, copy=True))

    def hold(self, current_pose: np.ndarray) -> None:
        self.held_poses.append(np.array(current_pose, dtype=np.float64, copy=True))

    @property
    def execute_count(self) -> int:
        return len(self.executed_actions)

    @property
    def hold_count(self) -> int:
        return len(self.held_poses)

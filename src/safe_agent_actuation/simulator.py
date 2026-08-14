"""Headless ALOHA simulator boundary and controlled policy clients.

The controlled client has no direct actuation call: every executable chunk is
first converted to an envelope by a supplied monitor/gate and then passed to a
supplied actuator.  The only direct stepping API is the explicitly named
baseline path, which is simulator-only and observation-only by default.

``gym_aloha`` and ``gymnasium`` are imported lazily so unit tests and recorded
trace experiments do not require MuJoCo.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .openpi_adapter import (
    DEFAULT_EXECUTED_ACTION_HORIZON,
    PREDICTED_ACTION_HORIZON,
    SIMULATOR_ACTION_DIMENSION,
    FastPolicyClientRecorder,
    InvalidPolicyResponse,
    require_executable_actions,
)

DEFAULT_ALOHA_TASK = "gym_aloha/AlohaTransferCube-v0"


class PolicyLike(Protocol):
    def infer(self, observation: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]: ...


class ActuatorLike(Protocol):
    def actuate(self, envelope: Any, current_state: ArrayLike) -> Any: ...


@dataclass(frozen=True, slots=True)
class BaselineObservation:
    """One policy response captured without stepping the simulator."""

    response: Mapping[str, Any]
    current_state: NDArray[np.float64]
    stepped: bool = False


@dataclass(frozen=True, slots=True)
class ControlledActuation:
    """Outcome of one policy-chunk attempt through monitor and actuator."""

    response: Mapping[str, Any]
    current_state: NDArray[np.float64]
    proposed_shape: tuple[int, int] | None
    executed_horizon: int
    actuator_result: Any
    blocked_before_monitor: bool
    reason: str | None = None


def _finite_state(value: ArrayLike, *, name: str = "current_state") -> NDArray[np.float64]:
    try:
        state = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric 14-D vector") from exc
    if state.shape != (SIMULATOR_ACTION_DIMENSION,):
        raise ValueError(
            f"{name} has shape {state.shape}; expected ({SIMULATOR_ACTION_DIMENSION},)"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} contains NaN or infinity")
    return np.array(state, dtype=np.float64, copy=True)


def _finite_action_chunk(
    value: ArrayLike,
    *,
    max_horizon: int,
) -> NDArray[np.float64]:
    try:
        chunk = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action_chunk must be a numeric [H, 14] array") from exc
    if chunk.ndim != 2 or chunk.shape[1:] != (SIMULATOR_ACTION_DIMENSION,):
        raise ValueError("action_chunk must have shape [H, 14]")
    if not 1 <= chunk.shape[0] <= max_horizon:
        raise ValueError(f"action_chunk horizon must be between 1 and {max_horizon}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("action_chunk contains NaN or infinity")
    return np.array(chunk, dtype=np.float64, copy=True)


class HeadlessAlohaSimulator:
    """Small simulator adapter suitable for the actuator enforcement boundary.

    ``execute`` and ``hold`` implement the interface expected by the course
    ``SafeActuator``.  ``step`` remains public only to support the explicitly
    requested ungated baseline experiment.
    """

    def __init__(
        self,
        environment: Any,
        *,
        max_execution_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    ) -> None:
        if not 1 <= int(max_execution_horizon) <= PREDICTED_ACTION_HORIZON:
            raise ValueError(
                f"max_execution_horizon must be between 1 and {PREDICTED_ACTION_HORIZON}"
            )
        self._environment = environment
        self.max_execution_horizon = int(max_execution_horizon)
        self._observation: Mapping[str, Any] | None = None
        self._done = True
        self.step_count = 0
        self.last_info: Mapping[str, Any] = {}

    @classmethod
    def create(
        cls,
        *,
        task: str = DEFAULT_ALOHA_TASK,
        seed: int = 0,
        observation_type: str = "pixels_agent_pos",
        max_episode_steps: int | None = None,
        max_execution_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    ) -> HeadlessAlohaSimulator:
        """Create gym-aloha with EGL rendering and no display."""

        os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            import gym_aloha  # noqa: F401  # type: ignore[import-not-found]
            import gymnasium  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "headless ALOHA requires gym-aloha, gymnasium, MuJoCo, and EGL support"
            ) from exc
        kwargs: dict[str, Any] = {"obs_type": observation_type}
        if max_episode_steps is not None:
            if int(max_episode_steps) <= 0:
                raise ValueError("max_episode_steps must be positive")
            kwargs["max_episode_steps"] = int(max_episode_steps)
        simulator = cls(
            gymnasium.make(task, **kwargs),
            max_execution_horizon=max_execution_horizon,
        )
        simulator.reset(seed=seed)
        return simulator

    @property
    def done(self) -> bool:
        return self._done

    @property
    def current_state(self) -> NDArray[np.float64]:
        if self._observation is None:
            raise RuntimeError("simulator has not been reset")
        if "agent_pos" not in self._observation:
            raise RuntimeError("ALOHA observation is missing agent_pos")
        return _finite_state(self._observation["agent_pos"])

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        reset_result = self._environment.reset(seed=seed)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if not isinstance(observation, Mapping):
            raise RuntimeError("ALOHA reset returned a non-mapping observation")
        self._observation = observation
        self._done = False
        self.step_count = 0
        self.last_info = {}
        # Validate state immediately so malformed simulator input fails closed.
        _ = self.current_state
        return observation

    def policy_observation(self, prompt: str) -> dict[str, Any]:
        """Convert the latest gym observation to OpenPI's ALOHA client schema."""

        if self._observation is None:
            raise RuntimeError("simulator has not been reset")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        pixels = self._observation.get("pixels")
        if not isinstance(pixels, Mapping) or "top" not in pixels:
            raise RuntimeError("ALOHA observation is missing pixels.top")
        image = np.asarray(pixels["top"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError("pixels.top must have shape [height, width, 3]")
        if image.dtype != np.uint8:
            if not np.all(np.isfinite(image)):
                raise RuntimeError("pixels.top contains NaN or infinity")
            if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
                image = np.clip(image * 255.0, 0, 255)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return {
            "state": self.current_state,
            "images": {"cam_high": np.transpose(image, (2, 0, 1))},
            "prompt": prompt.strip(),
        }

    def step(self, action: ArrayLike) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        """Explicitly step the simulator once.

        Controlled experiments do not call this method; their actuator calls
        :meth:`execute` only after successful enforcement.  The method exists
        for the clearly labeled baseline simulator experiment.
        """

        command = _finite_state(action, name="action")
        result = self._environment.step(command)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("ALOHA step must return (obs, reward, terminated, truncated, info)")
        observation, reward, terminated, truncated, info = result
        if not isinstance(observation, Mapping):
            raise RuntimeError("ALOHA step returned a non-mapping observation")
        self._observation = observation
        self._done = bool(terminated or truncated)
        self.step_count += 1
        self.last_info = info if isinstance(info, Mapping) else {"raw_info": repr(info)}
        _ = self.current_state
        return observation, float(reward), self._done, self.last_info

    def execute(
        self, action_chunk: ArrayLike
    ) -> list[tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]]:
        """Execute a previously authorized chunk, bounded by the course horizon."""

        chunk = _finite_action_chunk(action_chunk, max_horizon=self.max_execution_horizon)
        outcomes = []
        for command in chunk:
            if self.done:
                break
            outcomes.append(self.step(command))
        return outcomes

    def hold(
        self, current_pose: ArrayLike
    ) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        """Fail closed by commanding the validated current pose for one sim step."""

        pose = _finite_state(current_pose, name="hold_pose")
        # Reject a stale or fabricated hold pose rather than moving to it.
        if not np.allclose(pose, self.current_state, rtol=0.0, atol=1e-6):
            raise ValueError("hold_pose does not match the simulator's current observed pose")
        return self.step(pose)

    def close(self) -> None:
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


class BaselineAlohaClient:
    """Observation-only baseline with a separate, explicit simulator-step call."""

    def __init__(
        self,
        policy: PolicyLike,
        simulator: HeadlessAlohaSimulator,
        *,
        recorder: FastPolicyClientRecorder | None = None,
        executed_action_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    ) -> None:
        if not 1 <= int(executed_action_horizon) <= simulator.max_execution_horizon:
            raise ValueError("baseline execution horizon exceeds the simulator limit")
        self.policy = policy
        self.simulator = simulator
        self.recorder = recorder
        self.executed_action_horizon = int(executed_action_horizon)

    def observe(self, prompt: str) -> BaselineObservation:
        observation = self.simulator.policy_observation(prompt)
        response = self.policy.infer(observation)
        if self.recorder is not None:
            self.recorder.record(response)
        return BaselineObservation(
            response=response,
            current_state=self.simulator.current_state,
            stepped=False,
        )

    def step_observed(self, observed: BaselineObservation) -> list[Any]:
        """Explicitly execute a valid observed chunk in the simulator only.

        This intentionally bypasses the access-control experiment and must only
        be used for the assignment's labeled baseline scenario.
        """

        chunk = require_executable_actions(
            observed.response,
            executed_action_horizon=self.executed_action_horizon,
        )
        return self.simulator.execute(chunk)


def _invoke_gate(
    gate: Any,
    action_chunk: NDArray[np.float64],
    current_state: NDArray[np.float64],
    context: Mapping[str, Any],
) -> Any:
    process = getattr(gate, "process", None)
    if callable(process):
        return process(action_chunk, current_state, context)
    if callable(gate):
        return gate(action_chunk, current_state, context)
    raise TypeError("monitor_gate must be callable or provide process(action, state, context)")


class ControlledAlohaClient:
    """Run one pi0-FAST chunk exclusively through monitor and actuator."""

    def __init__(
        self,
        policy: PolicyLike,
        simulator: HeadlessAlohaSimulator,
        monitor_gate: Any,
        actuator: ActuatorLike,
        *,
        recorder: FastPolicyClientRecorder | None = None,
        executed_action_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    ) -> None:
        if not 1 <= int(executed_action_horizon) <= simulator.max_execution_horizon:
            raise ValueError("controlled execution horizon exceeds the simulator limit")
        if not callable(getattr(actuator, "actuate", None)):
            raise TypeError("actuator must provide actuate(envelope, current_state)")
        self.policy = policy
        self.simulator = simulator
        self.monitor_gate = monitor_gate
        self.actuator = actuator
        self.recorder = recorder
        self.executed_action_horizon = int(executed_action_horizon)

    def run_once(
        self,
        prompt: str,
        context: Mapping[str, Any],
    ) -> ControlledActuation:
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        observation = self.simulator.policy_observation(prompt)
        current_state = self.simulator.current_state
        response = self.policy.infer(observation)
        if self.recorder is not None:
            self.recorder.record(response)

        try:
            chunk = require_executable_actions(
                response,
                executed_action_horizon=self.executed_action_horizon,
            )
        except InvalidPolicyResponse as exc:
            # None is deliberately sent to the trusted actuator.  SafeActuator
            # treats malformed/bypass input as HOLD.  No action reaches monitor
            # or simulator from this branch.
            actuator_result = self.actuator.actuate(None, current_state)
            return ControlledActuation(
                response=response,
                current_state=current_state,
                proposed_shape=None,
                executed_horizon=0,
                actuator_result=actuator_result,
                blocked_before_monitor=True,
                reason=str(exc),
            )

        try:
            envelope = _invoke_gate(self.monitor_gate, chunk, current_state, dict(context))
        except Exception as exc:
            # Monitor errors also become malformed actuator input and therefore
            # a fail-closed HOLD, rather than an unprotected array send.
            actuator_result = self.actuator.actuate(None, current_state)
            return ControlledActuation(
                response=response,
                current_state=current_state,
                proposed_shape=tuple(chunk.shape),
                executed_horizon=0,
                actuator_result=actuator_result,
                blocked_before_monitor=False,
                reason=f"monitor failed closed: {exc}",
            )

        actuator_result = self.actuator.actuate(envelope, current_state)
        executed = (
            self.executed_action_horizon
            if bool(getattr(actuator_result, "executed", False))
            else 0
        )
        return ControlledActuation(
            response=response,
            current_state=current_state,
            proposed_shape=tuple(chunk.shape),
            executed_horizon=executed,
            actuator_result=actuator_result,
            blocked_before_monitor=False,
            reason=None,
        )


# More explicit names retained for assignment prose and external callers.
HeadlessAloha = HeadlessAlohaSimulator
BaselineObservationClient = BaselineAlohaClient
ControlledAlohaPolicyClient = ControlledAlohaClient

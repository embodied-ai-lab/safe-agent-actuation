from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from safe_agent_actuation.openpi_adapter import CHECKPOINT_COMPATIBILITY
from safe_agent_actuation.simulator import (
    BaselineAlohaClient,
    ControlledAlohaClient,
    HeadlessAlohaSimulator,
)


def gym_observation(state: np.ndarray | None = None) -> dict:
    return {
        "agent_pos": (
            np.zeros(14, dtype=np.float64) if state is None else np.array(state, copy=True)
        ),
        "pixels": {"top": np.zeros((24, 32, 3), dtype=np.uint8)},
    }


class FakeGymEnvironment:
    def __init__(self) -> None:
        self.observation = gym_observation()
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.observation = gym_observation()
        return self.observation, {"seed": seed}

    def step(self, action: np.ndarray):
        command = np.asarray(action, dtype=np.float64)
        self.actions.append(np.array(command, copy=True))
        self.observation = gym_observation(command)
        return self.observation, 0.0, False, False, {"step": len(self.actions)}

    def close(self) -> None:
        self.closed = True


def make_simulator() -> tuple[HeadlessAlohaSimulator, FakeGymEnvironment]:
    gym = FakeGymEnvironment()
    simulator = HeadlessAlohaSimulator(gym, max_execution_horizon=10)
    simulator.reset(seed=7)
    return simulator, gym


def final_actions() -> np.ndarray:
    actions = np.full((32, 14), 0.1, dtype=np.float64)
    actions[:, 0] = np.linspace(0.01, 0.32, 32)
    return actions


def valid_response() -> dict:
    return {
        "actions": final_actions(),
        "executable": True,
        "openpi_observability": {
            "decode_ok": True,
            "decode_error": None,
            "decoded_all_zero": False,
            "decoded_model_action_shape": [32, 32],
            "decoded_physical_action_shape": [32, 14],
            "raw_paligemma_token_ids": [22, 23, 1, 0],
            "fast_action_token_ids": [4, 8, 15],
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
        },
    }


class FakePolicy:
    def __init__(self, response: dict):
        self.response = response
        self.observations: list[dict] = []

    def infer(self, observation: dict, **kwargs: object) -> dict:
        self.observations.append(observation)
        return self.response


class RecordingGate:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, dict]] = []

    def process(self, action: np.ndarray, state: np.ndarray, context: dict) -> dict:
        self.calls.append((np.array(action, copy=True), np.array(state, copy=True), dict(context)))
        return {"protected_action": np.array(action, copy=True)}


@dataclass
class FakeActuationResult:
    executed: bool
    held: bool


class EnforcingFakeActuator:
    """Fake trusted actuator: envelopes execute; malformed input holds."""

    def __init__(self, simulator: HeadlessAlohaSimulator) -> None:
        self.simulator = simulator
        self.calls: list[tuple[object, np.ndarray]] = []

    def actuate(self, envelope: object, current_state: np.ndarray) -> FakeActuationResult:
        self.calls.append((envelope, np.array(current_state, copy=True)))
        if not isinstance(envelope, dict) or "protected_action" not in envelope:
            self.simulator.hold(current_state)
            return FakeActuationResult(executed=False, held=True)
        self.simulator.execute(envelope["protected_action"])
        return FakeActuationResult(executed=True, held=False)


def test_policy_observation_has_aloha_state_and_channel_first_image() -> None:
    simulator, _ = make_simulator()
    observation = simulator.policy_observation("Transfer cube")

    assert observation["state"].shape == (14,)
    assert observation["images"]["cam_high"].shape == (3, 24, 32)
    assert observation["prompt"] == "Transfer cube"


def test_baseline_is_observation_only_until_explicit_step() -> None:
    simulator, gym = make_simulator()
    client = BaselineAlohaClient(FakePolicy(valid_response()), simulator)

    observed = client.observe("Transfer cube")
    assert observed.stepped is False
    assert gym.actions == []

    outcomes = client.step_observed(observed)
    assert len(outcomes) == 10
    assert len(gym.actions) == 10
    np.testing.assert_allclose(np.stack(gym.actions), final_actions()[:10])


def test_invalid_baseline_response_cannot_be_explicitly_stepped() -> None:
    simulator, gym = make_simulator()
    invalid = {
        "executable": False,
        "openpi_observability": {"decode_ok": False, "decode_error": "all-zero fallback"},
    }
    client = BaselineAlohaClient(FakePolicy(invalid), simulator)
    observed = client.observe("Transfer cube")

    with pytest.raises(Exception, match="non-executable"):
        client.step_observed(observed)
    assert gym.actions == []


def test_controlled_client_executes_exactly_first_ten_only_via_gate_and_actuator() -> None:
    simulator, gym = make_simulator()
    gate = RecordingGate()
    actuator = EnforcingFakeActuator(simulator)
    client = ControlledAlohaClient(
        FakePolicy(valid_response()), simulator, gate, actuator, executed_action_horizon=10
    )

    outcome = client.run_once("Transfer cube", {"nearby_people": False})

    assert outcome.proposed_shape == (10, 14)
    assert outcome.executed_horizon == 10
    assert len(gate.calls) == 1
    np.testing.assert_allclose(gate.calls[0][0], final_actions()[:10])
    assert len(actuator.calls) == 1
    assert len(gym.actions) == 10
    np.testing.assert_allclose(np.stack(gym.actions), final_actions()[:10])


def test_invalid_decode_never_reaches_monitor_or_executes_an_action() -> None:
    simulator, gym = make_simulator()
    gate = RecordingGate()
    actuator = EnforcingFakeActuator(simulator)
    invalid = {
        "actions": np.full((32, 14), 99.0),  # stale/malicious array must be ignored
        "executable": False,
        "openpi_observability": {
            "decode_ok": False,
            "decode_error": "all-zero decode fallback",
        },
    }
    client = ControlledAlohaClient(FakePolicy(invalid), simulator, gate, actuator)

    outcome = client.run_once("Transfer cube", {})

    assert outcome.blocked_before_monitor is True
    assert outcome.executed_horizon == 0
    assert gate.calls == []
    assert actuator.calls[0][0] is None
    # Safe failure is one hold command equal to the observed pose, never 99s.
    assert len(gym.actions) == 1
    np.testing.assert_allclose(gym.actions[0], np.zeros(14))


def test_monitor_exception_fails_closed_at_actuator() -> None:
    simulator, gym = make_simulator()

    def broken_gate(action: np.ndarray, state: np.ndarray, context: dict) -> object:
        raise RuntimeError("authorization unavailable")

    actuator = EnforcingFakeActuator(simulator)
    client = ControlledAlohaClient(FakePolicy(valid_response()), simulator, broken_gate, actuator)
    outcome = client.run_once("Transfer cube", {})

    assert outcome.executed_horizon == 0
    assert "failed closed" in (outcome.reason or "")
    assert actuator.calls[0][0] is None
    assert len(gym.actions) == 1
    np.testing.assert_allclose(gym.actions[0], np.zeros(14))


@pytest.mark.parametrize(
    "chunk",
    [
        np.ones((11, 14)),
        np.ones((10, 13)),
        np.full((10, 14), np.inf),
        np.empty((0, 14)),
    ],
)
def test_simulator_rejects_malformed_or_over_horizon_chunks(chunk: np.ndarray) -> None:
    simulator, gym = make_simulator()
    with pytest.raises(ValueError):
        simulator.execute(chunk)
    assert gym.actions == []


def test_hold_rejects_pose_that_differs_from_current_observation() -> None:
    simulator, gym = make_simulator()
    with pytest.raises(ValueError, match="does not match"):
        simulator.hold(np.ones(14))
    assert gym.actions == []

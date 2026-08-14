from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _analysis import low_risk_analysis

from safe_agent_actuation.actuator import RecordingSimulator, SafeActuator
from safe_agent_actuation.protection import (
    AESGCMProtector,
    InMemorySessionKeyStore,
    SessionKey,
    TrustedLowRiskChannel,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def make_actuator(clock: ManualClock):
    simulator = RecordingSimulator()
    store = InMemorySessionKeyStore(clock=clock)
    protector = AESGCMProtector(store, clock=clock, nonce_source=lambda size: b"0" * size)
    low_channel = TrustedLowRiskChannel(clock=clock, secret=b"l" * 32)
    actuator = SafeActuator(
        simulator, high_risk_protector=protector, low_risk_channel=low_channel
    )
    return actuator, simulator, store, low_channel


def low_analysis():
    return low_risk_analysis(("dimension_0", "dimension_1"))


def test_explicit_trusted_low_risk_envelope_executes() -> None:
    clock = ManualClock()
    actuator, simulator, _, channel = make_actuator(clock)
    envelope = channel.issue([[0.0, 0.0], [0.05, 0.0]], low_analysis())
    result = actuator.actuate(envelope, [0.0, 0.0])
    assert result.executed
    assert result.path == "trusted_low_risk"
    assert simulator.execute_count == 1
    assert simulator.hold_count == 0


@pytest.mark.parametrize(
    "bypass",
    [
        np.array([[0.9, 0.9]]),
        [[0.9, 0.9]],
        {"action_chunk": [[0.9, 0.9]], "risk": "low"},
        None,
    ],
)
def test_unauthenticated_bypass_holds_and_never_executes(bypass: object) -> None:
    actuator, simulator, _, _ = make_actuator(ManualClock())
    result = actuator.actuate(bypass, [0.2, -0.2])
    assert result.held
    assert simulator.execute_count == 0
    assert simulator.hold_count == 1
    np.testing.assert_array_equal(simulator.held_poses[0], [0.2, -0.2])


def test_tampered_low_risk_envelope_holds_and_never_executes() -> None:
    actuator, simulator, _, channel = make_actuator(ManualClock())
    envelope = channel.issue([[0.0, 0.0]], low_analysis())
    tampered = replace(envelope, action_chunk=np.array([[0.9, 0.9]]))
    result = actuator.actuate(tampered, [0.0, 0.0])
    assert result.held
    assert simulator.execute_count == 0
    assert simulator.hold_count == 1


def test_authenticated_high_risk_envelope_executes() -> None:
    if not AESGCMProtector.available():
        pytest.skip("cryptography is unavailable")
    clock = ManualClock()
    actuator, simulator, store, _ = make_actuator(clock)
    store.install(SessionKey("high", b"h" * 32, 100.0, 110.0))
    envelope = actuator.high_risk_protector.protect([[0.8, -0.8]], "high")
    result = actuator.actuate(envelope, [0.0, 0.0])
    assert result.executed
    assert result.path == "sst_protected_high_risk"
    assert simulator.execute_count == 1


def test_expired_high_risk_action_holds_and_never_executes() -> None:
    if not AESGCMProtector.available():
        pytest.skip("cryptography is unavailable")
    clock = ManualClock()
    actuator, simulator, store, _ = make_actuator(clock)
    store.install(SessionKey("high", b"h" * 32, 100.0, 101.0))
    envelope = actuator.high_risk_protector.protect([[0.8, -0.8]], "high")
    clock.value = 101.0
    result = actuator.actuate(envelope, [0.0, 0.0])
    assert result.held
    assert simulator.execute_count == 0
    assert simulator.hold_count == 1


def test_verified_dimension_mismatch_holds() -> None:
    actuator, simulator, _, channel = make_actuator(ManualClock())
    analysis = low_risk_analysis(("dimension_0",))
    envelope = channel.issue([[0.0]], analysis)
    result = actuator.actuate(envelope, [0.0, 0.0])
    assert result.held
    assert simulator.execute_count == 0


def test_invalid_current_pose_fails_closed() -> None:
    actuator, simulator, _, channel = make_actuator(ManualClock())
    envelope = channel.issue([[0.0, 0.0]], low_analysis())
    result = actuator.actuate(envelope, [np.nan, 0.0])
    assert result.held
    assert simulator.execute_count == 0
    assert simulator.hold_count == 1

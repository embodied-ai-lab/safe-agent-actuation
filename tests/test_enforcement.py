"""Public Part 3 acceptance tests for the graded scenario functions.

These state the required behavior of the monitor send and actuator enforcement
TODOs without revealing an implementation. They fail until the graded regions
are complete, and they do not require a live SST deployment.
"""

from __future__ import annotations

import numpy as np

from safe_agent_actuation.actuator import RecordingSimulator
from safe_agent_actuation.context import AuthorizationContext
from safe_agent_actuation.course_config import load_physical_limits, repository_root
from safe_agent_actuation.scenarios import (
    enforce_received_action,
    send_high_risk_action_live,
)
from safe_agent_actuation.sst_adapter import SSTSendResult


def _limits():
    return load_physical_limits(repository_root())


def _allowed_context() -> AuthorizationContext:
    return AuthorizationContext(
        location="ASU Robotics Lab",
        nearby_people=0,
        permitted_time="14:00",
        task_name="transfer_cube",
        robot_id="aloha-sim-01",
    )


class _RecordingTransport:
    """Stand-in for SSTMonitorTransport that records one send call."""

    def __init__(self):
        self.calls = []

    def send(self, action_chunk, *, run_id, timestep, context):
        self.calls.append((np.asarray(action_chunk), run_id, timestep, dict(context)))
        return SSTSendResult(
            authorized=True,
            sent=True,
            key_reused=False,
            authorization_latency_ms=1.0,
            handshake_latency_ms=1.0,
            encryption_send_latency_ms=1.0,
            total_latency_ms=3.0,
            reason="recorded",
            key_id="deadbeef",
        )


def test_monitor_send_forwards_action_and_context():
    transport = _RecordingTransport()
    action = np.zeros((10, 14))
    result = send_high_risk_action_live(
        transport, action, run_id="r", timestep=0, context=_allowed_context()
    )
    assert result.sent and result.authorized
    assert len(transport.calls) == 1
    sent_action, run_id, timestep, context = transport.calls[0]
    assert run_id == "r"
    assert timestep == 0
    assert context == _allowed_context().to_dict()
    assert sent_action.shape == (10, 14)


def _enforce(payload, *, run_id="run", state=None):
    simulator = RecordingSimulator()
    current = np.zeros(14) if state is None else state
    result = enforce_received_action(
        payload,
        expected_run_id=run_id,
        current_state=current,
        limits=_limits(),
        simulator=simulator,
    )
    return result, simulator


def test_enforce_executes_valid_action():
    result, simulator = _enforce({"run_id": "run", "action_chunk": np.zeros((10, 14))})
    assert result.executed
    assert simulator.execute_count == 1
    assert simulator.hold_count == 0  # no hold on the accepted path


def test_enforce_holds_on_missing_payload():
    result, simulator = _enforce(None)
    assert result.held
    assert simulator.execute_count == 0


def test_enforce_holds_on_run_id_mismatch():
    result, simulator = _enforce({"run_id": "other", "action_chunk": np.zeros((10, 14))})
    assert result.held
    assert simulator.execute_count == 0


def test_enforce_holds_on_dimension_mismatch():
    result, simulator = _enforce({"run_id": "run", "action_chunk": np.zeros((10, 13))})
    assert result.held
    assert simulator.execute_count == 0


def test_enforce_holds_when_action_exceeds_limits():
    action = np.zeros((10, 14))
    action[0, 0] = 100.0
    result, simulator = _enforce({"run_id": "run", "action_chunk": action})
    assert result.held
    assert simulator.execute_count == 0


def test_enforce_holds_on_nonfinite_action():
    action = np.zeros((10, 14))
    action[3, 2] = np.nan
    result, simulator = _enforce({"run_id": "run", "action_chunk": action})
    assert result.held
    assert simulator.execute_count == 0

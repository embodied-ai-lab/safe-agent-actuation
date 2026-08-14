from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from safe_agent_actuation.sst_adapter import (
    SSTAdapterError,
    context_to_sst,
    decode_protected_action,
    encode_protected_action,
    session_key_valid,
)


def allowed_context() -> dict[str, object]:
    return {
        "location": "ASU Robotics Lab",
        "nearby_people": 0,
        "permitted_time": "14:00",
        "task_name": "transfer_cube",
        "robot_id": "aloha-sim-01",
    }


def test_context_maps_to_implemented_sst_schema() -> None:
    assert context_to_sst(allowed_context()) == {
        "Location": "ASU Robotics Lab",
        "Number of People": 0,
        "Time of Day": "14:00",
        "Task Name": "transfer_cube",
        "Robot ID": "aloha-sim-01",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"nearby_people": -1},
        {"nearby_people": True},
        {"permitted_time": "2pm"},
        {"task_name": ""},
    ],
)
def test_malformed_context_fails_closed(mutation: dict[str, object]) -> None:
    context = allowed_context()
    context.update(mutation)
    with pytest.raises(SSTAdapterError):
        context_to_sst(context)


def test_missing_context_fails_closed() -> None:
    context = allowed_context()
    context.pop("robot_id")
    with pytest.raises(SSTAdapterError):
        context_to_sst(context)


def test_protected_action_round_trip() -> None:
    action = np.zeros((10, 14))
    encoded = encode_protected_action(
        action, run_id="test", timestep=3, context=allowed_context()
    )
    decoded = decode_protected_action(encoded)
    np.testing.assert_array_equal(decoded["action_chunk"], action)


def test_malformed_protected_payload_rejected() -> None:
    with pytest.raises(SSTAdapterError):
        decode_protected_action(b"not-json")


def test_expired_session_key_is_not_valid() -> None:
    key = SimpleNamespace(abs_validity=999, rel_validity=None, first_use_ms=None)
    assert not session_key_valid(key, now_ms=1000)


def test_relative_expiration_is_enforced() -> None:
    key = SimpleNamespace(abs_validity=None, rel_validity=100, first_use_ms=1000)
    assert session_key_valid(key, now_ms=1099)
    assert not session_key_valid(key, now_ms=1100)


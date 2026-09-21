from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from safe_agent_actuation.sst_adapter import (
    SSTActuatorTransport,
    SSTAdapterError,
    SSTMonitorTransport,
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


class _Channel:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.sent: list[bytes] = []
        self.closed = False

    def send(self, payload: bytes) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


def _transport_with(ctx, channel: _Channel) -> SSTMonitorTransport:
    transport = SSTMonitorTransport(Path("unused.config"), clock_ms=lambda: 1_000)

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def connect(self):
            return channel

    transport._load = lambda: (ctx, Client)  # type: ignore[method-assign]
    return transport


def _valid_key():
    return SimpleNamespace(
        id=b"\x01\x02",
        abs_validity=10_000,
        rel_validity=None,
        first_use_ms=None,
    )


def test_monitor_transport_records_request_then_reuse() -> None:
    class Context:
        def __init__(self) -> None:
            self.calls = 0

        def request_session_keys(self, **_kwargs):
            self.calls += 1
            return [_valid_key()]

    context = Context()
    channel = _Channel()
    transport = _transport_with(context, channel)
    action = np.zeros((2, 14))

    first = transport.send(
        action, run_id="first", timestep=0, context=allowed_context()
    )
    second = transport.send(
        action, run_id="second", timestep=1, context=allowed_context()
    )

    assert context.calls == 1
    assert first.authorization_status == "allowed"
    assert first.key_request_attempts == 1
    assert first.successful_key_requests == 1
    assert first.key_reused is False
    assert first.transfer_attempted and first.sent
    assert second.key_request_attempts == 0
    assert second.successful_key_requests == 0
    assert second.key_reused is True
    assert second.transfer_attempted and second.sent
    assert transport.key_request_attempts == 1
    assert transport.key_requests == 1
    assert transport.key_reuses == 1


def test_auth_policy_alert_is_structured_without_transfer() -> None:
    class AuthProtocolError(Exception):
        pass

    class Context:
        def request_session_keys(self, **_kwargs):
            raise AuthProtocolError("Auth returned alert 1: invalid session key request")

    channel = _Channel()
    result = _transport_with(Context(), channel).send(
        np.zeros((2, 14)), run_id="denied", timestep=0, context=allowed_context()
    )

    assert result.authorized is False
    assert result.sent is False
    assert result.authorization_status == "auth_rejected_request"
    assert result.auth_error_type == "AuthProtocolError"
    assert result.auth_alert_code == 1
    assert result.key_request_attempts == 1
    assert result.successful_key_requests == 0
    assert result.transfer_attempted is False
    assert channel.sent == []


def test_auth_timeout_is_infrastructure_error_not_policy_denial() -> None:
    class Context:
        def request_session_keys(self, **_kwargs):
            raise TimeoutError("Auth did not respond")

    result = _transport_with(Context(), _Channel()).send(
        np.zeros((2, 14)), run_id="timeout", timestep=0, context=allowed_context()
    )

    assert result.authorization_status == "infrastructure_error"
    assert result.auth_error_type == "TimeoutError"
    assert result.auth_alert_code is None
    assert result.key_request_attempts == 1
    assert result.successful_key_requests == 0
    assert result.transfer_attempted is False


def test_transfer_failure_remains_distinct_from_authorization() -> None:
    class Context:
        def request_session_keys(self, **_kwargs):
            return [_valid_key()]

    channel = _Channel(OSError("injected transfer failure"))
    result = _transport_with(Context(), channel).send(
        np.zeros((2, 14)), run_id="transfer", timestep=0, context=allowed_context()
    )

    assert result.authorized is True
    assert result.authorization_status == "allowed"
    assert result.sent is False
    assert result.transfer_attempted is True
    assert result.transfer_error_type == "OSError"
    assert result.key_request_attempts == 1
    assert result.successful_key_requests == 1
    assert channel.closed is True


def test_invalid_key_response_is_structured_as_infrastructure_error() -> None:
    class Context:
        def request_session_keys(self, **_kwargs):
            return []

    result = _transport_with(Context(), _Channel()).send(
        np.zeros((2, 14)), run_id="invalid", timestep=0, context=allowed_context()
    )

    assert result.authorization_status == "infrastructure_error"
    assert result.auth_error_type == "InvalidSessionKeyResponse"
    assert result.key_request_attempts == 1
    assert result.successful_key_requests == 0
    assert result.transfer_attempted is False


def test_receive_failure_exposes_error_type(monkeypatch) -> None:
    module = ModuleType("iotauth")
    module.IoTAuthContext = object  # type: ignore[attr-defined]
    module.SecureServer = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iotauth", module)

    class Server:
        def __init__(self) -> None:
            self.closed = False

        def serve_once(self):
            raise TimeoutError("no peer arrived: diagnostic sentinel")

        def close(self) -> None:
            self.closed = True

    server = Server()
    transport = SSTActuatorTransport(Path("unused.config"))
    transport._server = server
    result = transport.receive_once()

    assert result.authenticated is False
    assert result.received is False
    assert result.receive_error_type == "TimeoutError"
    assert "diagnostic sentinel" in result.reason
    assert server.closed is True


def test_actuator_transport_reuses_entity_context_across_connections(monkeypatch) -> None:
    module = ModuleType("iotauth")

    class Context:
        loads = 0

        @classmethod
        def from_config(cls, _path):
            cls.loads += 1
            return object()

    class Server:
        contexts = []

        def __init__(self, ctx, **_kwargs) -> None:
            self.contexts.append(ctx)

        def listen(self) -> None:
            pass

        def close(self) -> None:
            pass

    module.IoTAuthContext = Context  # type: ignore[attr-defined]
    module.SecureServer = Server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iotauth", module)

    transport = SSTActuatorTransport(Path("unused.config"))
    transport.listen()
    transport.close()
    transport.listen()
    transport.close()

    assert Context.loads == 1
    assert len(Server.contexts) == 2
    assert Server.contexts[0] is Server.contexts[1]

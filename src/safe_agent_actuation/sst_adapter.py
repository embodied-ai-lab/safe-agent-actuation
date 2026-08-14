"""Narrow adapter around the public IoTAuth Python API.

The adapter never implements SST cryptography. It adds strict course payload
validation, explicit context mapping, and key-reuse accounting around the
public IoTAuthContext, SecureClient, and SecureServer classes.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COURSE_CONTEXT_FIELDS = (
    "location",
    "nearby_people",
    "permitted_time",
    "task_name",
    "robot_id",
)


class SSTAdapterError(RuntimeError):
    """Raised when SST is unavailable or a protected payload is invalid."""


def context_to_sst(context: Mapping[str, object]) -> dict[str, object]:
    """Validate course context and map it to the pinned Auth policy schema."""

    if not isinstance(context, Mapping):
        raise SSTAdapterError("context must be a mapping")
    missing = [name for name in COURSE_CONTEXT_FIELDS if name not in context]
    extra = sorted(set(context) - set(COURSE_CONTEXT_FIELDS))
    if missing or extra:
        raise SSTAdapterError(f"context fields invalid: missing={missing}, extra={extra}")

    location = context["location"]
    people = context["nearby_people"]
    permitted_time = context["permitted_time"]
    task_name = context["task_name"]
    robot_id = context["robot_id"]
    text_fields = (location, permitted_time, task_name, robot_id)
    if not all(isinstance(value, str) and value for value in text_fields):
        raise SSTAdapterError("context string fields must be non-empty strings")
    if isinstance(people, bool) or not isinstance(people, int) or people < 0:
        raise SSTAdapterError("nearby_people must be a non-negative integer")
    try:
        parsed = time.strptime(permitted_time, "%H:%M")
    except ValueError as exc:
        raise SSTAdapterError("permitted_time must use HH:MM") from exc
    if time.strftime("%H:%M", parsed) != permitted_time:
        raise SSTAdapterError("permitted_time must use zero-padded HH:MM")
    return {
        "Location": location,
        "Number of People": people,
        "Time of Day": permitted_time,
        "Task Name": task_name,
        "Robot ID": robot_id,
    }


def session_key_valid(key: Any, *, now_ms: int | None = None) -> bool:
    """Apply the absolute and relative validity rules exposed by IoTAuth."""

    if key is None:
        return False
    current = int(time.time() * 1000) if now_ms is None else int(now_ms)
    absolute = getattr(key, "abs_validity", None)
    relative = getattr(key, "rel_validity", None)
    first_use = getattr(key, "first_use_ms", None)
    if absolute is not None and current >= int(absolute):
        return False
    if relative is not None and first_use is not None and current >= int(first_use) + int(relative):
        return False
    return True


def encode_protected_action(
    action_chunk: np.ndarray,
    *,
    run_id: str,
    timestep: int,
    context: Mapping[str, object],
) -> bytes:
    """Create the application payload that SST protects on the wire."""

    action = np.asarray(action_chunk, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] == 0:
        raise SSTAdapterError("protected action must have non-empty [H, D] shape")
    if not np.isfinite(action).all():
        raise SSTAdapterError("protected action contains NaN or infinity")
    mapped = context_to_sst(context)
    body = {
        "schema": "safe-agent-actuation/protected-action/v1",
        "run_id": str(run_id),
        "timestep": int(timestep),
        "action_shape": list(action.shape),
        "action_chunk": action.tolist(),
        "context": dict(context),
        "sst_context": mapped,
    }
    return json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_protected_action(payload: bytes, *, expected_dimension: int = 14) -> dict[str, Any]:
    """Strictly validate plaintext returned only after an SST secure receive."""

    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SSTAdapterError("protected payload is not valid UTF-8 JSON") from exc
    expected_schema = "safe-agent-actuation/protected-action/v1"
    if not isinstance(body, dict) or body.get("schema") != expected_schema:
        raise SSTAdapterError("protected payload schema is missing or unsupported")
    if not isinstance(body.get("run_id"), str) or not body["run_id"]:
        raise SSTAdapterError("protected payload has no run_id")
    if isinstance(body.get("timestep"), bool) or not isinstance(body.get("timestep"), int):
        raise SSTAdapterError("protected payload timestep must be an integer")
    action = np.asarray(body.get("action_chunk"), dtype=np.float64)
    shape = body.get("action_shape")
    if (
        action.ndim != 2
        or action.shape[0] == 0
        or action.shape[1] != expected_dimension
        or shape != list(action.shape)
        or not np.isfinite(action).all()
    ):
        raise SSTAdapterError("protected action shape or values are invalid")
    context_to_sst(body.get("context", {}))
    if body.get("sst_context") != context_to_sst(body["context"]):
        raise SSTAdapterError("protected context mapping is inconsistent")
    body["action_chunk"] = action
    return body


@dataclass(frozen=True)
class SSTSendResult:
    authorized: bool
    sent: bool
    key_reused: bool
    authorization_latency_ms: float
    handshake_latency_ms: float
    encryption_send_latency_ms: float
    total_latency_ms: float
    reason: str
    key_id: str | None = None


@dataclass(frozen=True)
class SSTReceiveResult:
    authenticated: bool
    received: bool
    receive_decryption_latency_ms: float
    reason: str
    payload: dict[str, Any] | None = None


class SSTMonitorTransport:
    """Request or reuse a context-authorized key, then send through SST."""

    def __init__(
        self,
        config_path: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 23100,
        timeout: float = 5.0,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._ctx: Any = None
        self._cached_key: Any = None
        self.key_requests = 0

    def _load(self) -> tuple[Any, Any]:
        try:
            from iotauth import IoTAuthContext, SecureClient
        except ImportError as exc:
            raise SSTAdapterError(
                "iotauth is unavailable. Install third_party/iotauth/entity/python"
            ) from exc
        if self._ctx is None:
            self._ctx = IoTAuthContext.from_config(self.config_path)
        return self._ctx, SecureClient

    def clear_cached_key(self) -> None:
        self._cached_key = None

    def cached_key_validity(self) -> dict[str, int | None]:
        """Report the cached key's validity metadata without exposing key bytes."""

        key = self._cached_key
        return {
            "rel_validity_ms": getattr(key, "rel_validity", None),
            "abs_validity_ms": getattr(key, "abs_validity", None),
            "first_use_ms": getattr(key, "first_use_ms", None),
        }

    def send(
        self,
        action_chunk: np.ndarray,
        *,
        run_id: str,
        timestep: int,
        context: Mapping[str, object],
    ) -> SSTSendResult:
        start = time.perf_counter()
        payload = encode_protected_action(
            action_chunk,
            run_id=run_id,
            timestep=timestep,
            context=context,
        )
        ctx, secure_client_type = self._load()
        reused = session_key_valid(self._cached_key, now_ms=self.clock_ms())
        auth_ms = 0.0
        if not reused:
            self._cached_key = None
            auth_start = time.perf_counter()
            try:
                keys = ctx.request_session_keys(
                    purpose={"group": "Actuators", "context": context_to_sst(context)},
                    count=1,
                    timeout=self.timeout,
                )
            except Exception as exc:
                auth_ms = (time.perf_counter() - auth_start) * 1000
                return SSTSendResult(
                    authorized=False,
                    sent=False,
                    key_reused=False,
                    authorization_latency_ms=auth_ms,
                    handshake_latency_ms=0.0,
                    encryption_send_latency_ms=0.0,
                    total_latency_ms=(time.perf_counter() - start) * 1000,
                    reason=f"authorization denied or unavailable: {type(exc).__name__}",
                )
            auth_ms = (time.perf_counter() - auth_start) * 1000
            self.key_requests += 1
            if not keys or not session_key_valid(keys[0], now_ms=self.clock_ms()):
                return SSTSendResult(
                    authorized=False,
                    sent=False,
                    key_reused=False,
                    authorization_latency_ms=auth_ms,
                    handshake_latency_ms=0.0,
                    encryption_send_latency_ms=0.0,
                    total_latency_ms=(time.perf_counter() - start) * 1000,
                    reason="Auth returned no currently valid session key",
                )
            self._cached_key = keys[0]

        handshake_start = time.perf_counter()
        try:
            client = secure_client_type(
                ctx,
                key=self._cached_key,
                host=self.host,
                port=self.port,
                timeout=self.timeout,
            )
            channel = client.connect()
            handshake_ms = (time.perf_counter() - handshake_start) * 1000
            send_start = time.perf_counter()
            channel.send(payload)
            send_ms = (time.perf_counter() - send_start) * 1000
            channel.close()
        except Exception as exc:
            return SSTSendResult(
                authorized=True,
                sent=False,
                key_reused=reused,
                authorization_latency_ms=auth_ms,
                handshake_latency_ms=(time.perf_counter() - handshake_start) * 1000,
                encryption_send_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start) * 1000,
                reason=f"secure transfer failed: {type(exc).__name__}",
                key_id=getattr(self._cached_key, "id", b"").hex() or None,
            )
        return SSTSendResult(
            authorized=True,
            sent=True,
            key_reused=reused,
            authorization_latency_ms=auth_ms,
            handshake_latency_ms=handshake_ms,
            encryption_send_latency_ms=send_ms,
            total_latency_ms=(time.perf_counter() - start) * 1000,
            reason="authorized and sent through an authenticated SST channel",
            key_id=getattr(self._cached_key, "id", b"").hex() or None,
        )


class SSTActuatorTransport:
    """Receive one authenticated SST action for actuator-side enforcement."""

    def __init__(
        self,
        config_path: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 23100,
        accept_timeout: float = 5.0,
        handshake_timeout: float = 5.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.host = host
        self.port = int(port)
        self.accept_timeout = float(accept_timeout)
        self.handshake_timeout = float(handshake_timeout)
        self._server: Any = None

    def listen(self) -> None:
        """Bind the actuator port before a sender starts, avoiding a race.

        Scenario orchestration calls this in the main thread and then runs
        :meth:`receive_once` in a worker thread, so a monitor connecting
        immediately afterwards cannot be refused by an unbound port.
        """

        try:
            from iotauth import IoTAuthContext, SecureServer
        except ImportError as exc:
            raise SSTAdapterError(
                "iotauth is unavailable. Install third_party/iotauth/entity/python"
            ) from exc
        if self._server is not None:
            return
        ctx = IoTAuthContext.from_config(self.config_path)
        server = SecureServer(
            ctx,
            host=self.host,
            port=self.port,
            accept_timeout=self.accept_timeout,
            handshake_timeout=self.handshake_timeout,
        )
        server.listen()
        self._server = server

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()

    def receive_once(self) -> SSTReceiveResult:
        try:
            from iotauth import IoTAuthContext, SecureServer
        except ImportError as exc:
            raise SSTAdapterError(
                "iotauth is unavailable. Install third_party/iotauth/entity/python"
            ) from exc
        start = time.perf_counter()
        server = None
        channel = None
        try:
            if self._server is not None:
                server, self._server = self._server, None
            else:
                ctx = IoTAuthContext.from_config(self.config_path)
                server = SecureServer(
                    ctx,
                    host=self.host,
                    port=self.port,
                    accept_timeout=self.accept_timeout,
                    handshake_timeout=self.handshake_timeout,
                )
                server.listen()
            channel = server.serve_once()
            receive_start = time.perf_counter()
            raw = channel.recv(timeout=self.handshake_timeout)
            payload = decode_protected_action(raw)
            receive_ms = (time.perf_counter() - receive_start) * 1000
            return SSTReceiveResult(
                authenticated=True,
                received=True,
                receive_decryption_latency_ms=receive_ms,
                reason="SST handshake and protected receive succeeded",
                payload=payload,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return SSTReceiveResult(
                authenticated=False,
                received=False,
                receive_decryption_latency_ms=elapsed,
                reason=f"protected receive rejected: {type(exc).__name__}",
            )
        finally:
            if channel is not None:
                channel.close()
            if server is not None:
                server.close()


def finite_latency(value: float) -> bool:
    """Small schema helper used by live-record validation."""

    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


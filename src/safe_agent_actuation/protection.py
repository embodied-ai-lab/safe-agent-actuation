"""Offline authorization and authenticated high-risk action protection.

This module mirrors the enforcement properties of an SST secure channel while
remaining deterministic and unit-testable without a live Auth process.  A live
adapter can install SST-derived key bytes in :class:`InMemorySessionKeyStore`.
Key material is deliberately excluded from representations and records.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .context import AuthorizationContext, ContextPolicy
from .types import (
    AuthorizationDecision,
    AuthorizationOutcome,
    MotionAnalysis,
)

try:  # pragma: no cover - which branch runs depends on the offline environment.
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None  # type: ignore[assignment,misc]

    class InvalidTag(Exception):
        """Fallback name when the optional cryptography package is absent."""


Clock = Callable[[], float]
NonceSource = Callable[[int], bytes]


class ProtectionError(ValueError):
    """Base class for fail-closed protected-envelope failures."""


class UnknownKeyError(ProtectionError):
    pass


class ExpiredKeyError(ProtectionError):
    pass


class AuthenticationError(ProtectionError):
    pass


class MalformedEnvelopeError(ProtectionError):
    pass


def _now(clock: Clock) -> float:
    value = float(clock())
    if not np.isfinite(value):
        raise ValueError("clock returned a non-finite time")
    return value


def _key_fingerprint(key: bytes) -> str:
    """Return a non-secret identifier; never serialize the key itself."""

    return hashlib.sha256(key).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SessionKey:
    """Session key installed at both monitor and actuator endpoints."""

    key_id: str
    key: bytes = field(repr=False)
    issued_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise ValueError("key_id must be a non-empty string")
        key_bytes = bytes(self.key)
        if len(key_bytes) not in {16, 24, 32}:
            raise ValueError("AES-GCM key must contain 16, 24, or 32 bytes")
        if not np.isfinite(self.issued_at) or not np.isfinite(self.expires_at):
            raise ValueError("session-key times must be finite")
        if self.expires_at <= self.issued_at:
            raise ValueError("session key must expire after it is issued")
        object.__setattr__(self, "key_id", self.key_id.strip())
        object.__setattr__(self, "key", key_bytes)

    @classmethod
    def generate(
        cls,
        validity_seconds: float,
        *,
        clock: Clock = time.time,
        key_id: str | None = None,
        key: bytes | None = None,
    ) -> SessionKey:
        try:
            validity = float(validity_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("validity_seconds must be finite and positive") from exc
        if not np.isfinite(validity) or validity <= 0:
            raise ValueError("validity_seconds must be finite and positive")
        issued = _now(clock)
        material = secrets.token_bytes(32) if key is None else bytes(key)
        identifier = key_id or _key_fingerprint(material)
        return cls(identifier, material, issued, issued + validity)

    def is_valid(self, now: float) -> bool:
        """Validity is half-open: ``issued_at <= now < expires_at``."""

        return np.isfinite(now) and self.issued_at <= now < self.expires_at

    def to_public_dict(self) -> dict[str, str | float]:
        return {
            "key_id": self.key_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


class InMemorySessionKeyStore:
    """Small endpoint-local key cache with explicit expiration checks."""

    def __init__(self, *, clock: Clock = time.time) -> None:
        self._clock = clock
        self._keys: dict[str, SessionKey] = {}

    def install(self, session_key: SessionKey) -> None:
        if not isinstance(session_key, SessionKey):
            raise TypeError("session_key must be a SessionKey")
        self._keys[session_key.key_id] = session_key

    put = install

    def get(self, key_id: str, *, now: float | None = None) -> SessionKey:
        if not isinstance(key_id, str) or not key_id:
            raise UnknownKeyError("missing session key identifier")
        key = self._keys.get(key_id)
        if key is None:
            raise UnknownKeyError("unknown session key identifier")
        checked_at = _now(self._clock) if now is None else float(now)
        if not key.is_valid(checked_at):
            raise ExpiredKeyError("session key is missing, not yet valid, or expired")
        return key

    require = get

    def has_valid(self, key_id: str) -> bool:
        try:
            self.get(key_id)
        except ProtectionError:
            return False
        return True

    def remove(self, key_id: str) -> None:
        self._keys.pop(key_id, None)

    def public_key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))


SessionKeyStore = InMemorySessionKeyStore


@dataclass(frozen=True, slots=True)
class ProtectedActionEnvelope:
    """Authenticated ciphertext carried on the high-risk path."""

    version: int
    algorithm: str
    key_id: str
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)
    created_at: float
    action_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.version != 1 or self.algorithm != "AES-256-GCM":
            raise MalformedEnvelopeError("unsupported protected-envelope version or algorithm")
        if not isinstance(self.key_id, str) or not self.key_id:
            raise MalformedEnvelopeError("protected envelope has no key identifier")
        if not isinstance(self.nonce, bytes) or len(self.nonce) != 12:
            raise MalformedEnvelopeError("AES-GCM nonce must contain 12 bytes")
        if not isinstance(self.ciphertext, bytes) or len(self.ciphertext) < 16:
            raise MalformedEnvelopeError("protected envelope ciphertext is malformed")
        if not np.isfinite(self.created_at):
            raise MalformedEnvelopeError("protected envelope creation time is invalid")
        if (
            not isinstance(self.action_shape, tuple)
            or len(self.action_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.action_shape
            )
        ):
            raise MalformedEnvelopeError("protected envelope action shape must be [H, D]")

    def to_dict(self) -> dict[str, object]:
        """Serialize transport data without ever including session key material."""

        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "created_at": self.created_at,
            "action_shape": list(self.action_shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProtectedActionEnvelope:
        required = {
            "version",
            "algorithm",
            "key_id",
            "nonce",
            "ciphertext",
            "created_at",
            "action_shape",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise MalformedEnvelopeError("protected envelope fields are missing or unexpected")
        try:
            nonce = base64.b64decode(value["nonce"], validate=True)  # type: ignore[arg-type]
            ciphertext = base64.b64decode(value["ciphertext"], validate=True)  # type: ignore[arg-type]
            shape_raw = value["action_shape"]
            if not isinstance(shape_raw, (list, tuple)):
                raise TypeError
            shape = tuple(shape_raw)
            return cls(
                version=value["version"],  # type: ignore[arg-type]
                algorithm=value["algorithm"],  # type: ignore[arg-type]
                key_id=value["key_id"],  # type: ignore[arg-type]
                nonce=nonce,
                ciphertext=ciphertext,
                created_at=float(value["created_at"]),  # type: ignore[arg-type]
                action_shape=shape,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, OverflowError, KeyError) as exc:
            raise MalformedEnvelopeError("protected envelope encoding is malformed") from exc

    def __repr__(self) -> str:
        return (
            "ProtectedActionEnvelope(version=1, algorithm='AES-256-GCM', "
            f"key_id={self.key_id!r}, action_shape={self.action_shape!r}, "
            "nonce=<redacted>, ciphertext=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class LowRiskEnvelope:
    """Capability minted only by a configured trusted low-risk channel."""

    action_chunk: np.ndarray = field(repr=False)
    analysis: MotionAnalysis
    channel_id: str
    issued_at: float
    authenticator: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, MotionAnalysis) or not self.analysis.is_low_risk:
            raise ValueError("low-risk envelope requires a valid low-risk analysis")
        action = _validated_action(self.action_chunk)
        if action.shape[1] != len(self.analysis.dimension_names):
            raise ValueError("low-risk envelope action dimension does not match analysis")
        if not self.channel_id or not np.isfinite(self.issued_at):
            raise ValueError("low-risk envelope metadata is malformed")
        if not isinstance(self.authenticator, bytes) or len(self.authenticator) != 32:
            raise ValueError("low-risk envelope authenticator is malformed")
        object.__setattr__(self, "action_chunk", action)

    @property
    def action_shape(self) -> tuple[int, int]:
        return self.action_chunk.shape  # type: ignore[return-value]


def _validated_action(action_chunk: ArrayLike) -> np.ndarray:
    try:
        action = np.asarray(action_chunk, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MalformedEnvelopeError("action chunk must contain numeric values") from exc
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] == 0:
        raise MalformedEnvelopeError("action chunk must have non-empty shape [H, D]")
    if not np.all(np.isfinite(action)):
        raise MalformedEnvelopeError("action chunk contains NaN or infinity")
    result = np.ascontiguousarray(action, dtype="<f8")
    result.setflags(write=False)
    return result


def _action_bytes(action: np.ndarray) -> bytes:
    return action.tobytes(order="C")


def _low_risk_auth_data(
    channel_id: str, issued_at: float, analysis: MotionAnalysis, action: np.ndarray
) -> bytes:
    metadata = json.dumps(
        {
            "channel_id": channel_id,
            "issued_at": issued_at,
            "action_shape": list(action.shape),
            "classification": analysis.classification.value,
            "motion_score": analysis.motion_score,
            "threshold": analysis.threshold,
            "dimensions": list(analysis.dimension_names),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return metadata + b"\0" + _action_bytes(action)


class TrustedLowRiskChannel:
    """Unforgeable-in-process capability for the monitor's trusted direct path."""

    def __init__(
        self,
        channel_id: str = "trusted-monitor-low-risk",
        *,
        clock: Clock = time.time,
        secret: bytes | None = None,
        max_age_seconds: float = 30.0,
    ) -> None:
        if not isinstance(channel_id, str) or not channel_id.strip():
            raise ValueError("channel_id must be a non-empty string")
        if not np.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be finite and positive")
        self.channel_id = channel_id.strip()
        self._clock = clock
        self._secret = secrets.token_bytes(32) if secret is None else bytes(secret)
        if len(self._secret) < 16:
            raise ValueError("trusted-channel secret must contain at least 16 bytes")
        self.max_age_seconds = float(max_age_seconds)

    def issue(self, action_chunk: ArrayLike, analysis: MotionAnalysis) -> LowRiskEnvelope:
        if not isinstance(analysis, MotionAnalysis) or not analysis.is_low_risk:
            raise ValueError("only a valid low-risk analysis may use the direct path")
        action = _validated_action(action_chunk)
        issued = _now(self._clock)
        tag = hmac.digest(
            self._secret,
            _low_risk_auth_data(self.channel_id, issued, analysis, action),
            "sha256",
        )
        return LowRiskEnvelope(action, analysis, self.channel_id, issued, tag)

    wrap = issue

    def verify(self, envelope: LowRiskEnvelope) -> np.ndarray:
        if not isinstance(envelope, LowRiskEnvelope):
            raise AuthenticationError("input is not a trusted low-risk envelope")
        now = _now(self._clock)
        if envelope.channel_id != self.channel_id:
            raise AuthenticationError("low-risk channel identifier is not trusted")
        if envelope.issued_at > now or now - envelope.issued_at > self.max_age_seconds:
            raise AuthenticationError("trusted low-risk envelope is expired or not yet valid")
        expected = hmac.digest(
            self._secret,
            _low_risk_auth_data(
                envelope.channel_id, envelope.issued_at, envelope.analysis, envelope.action_chunk
            ),
            "sha256",
        )
        if not hmac.compare_digest(expected, envelope.authenticator):
            raise AuthenticationError("trusted low-risk envelope authentication failed")
        return envelope.action_chunk


def _aad(
    *, version: int, algorithm: str, key_id: str, created_at: float, action_shape: tuple[int, int]
) -> bytes:
    return json.dumps(
        {
            "version": version,
            "algorithm": algorithm,
            "key_id": key_id,
            "created_at": created_at,
            "action_shape": list(action_shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class AESGCMProtector:
    """Encrypt and authenticate action chunks with endpoint session keys."""

    def __init__(
        self,
        key_store: InMemorySessionKeyStore,
        *,
        clock: Clock = time.time,
        nonce_source: NonceSource = secrets.token_bytes,
    ) -> None:
        if not isinstance(key_store, InMemorySessionKeyStore):
            raise TypeError("key_store must be an InMemorySessionKeyStore")
        self.key_store = key_store
        self._clock = clock
        self._nonce_source = nonce_source

    @staticmethod
    def available() -> bool:
        return AESGCM is not None

    @staticmethod
    def _require_crypto() -> None:
        if AESGCM is None:
            raise ProtectionError(
                "authenticated action protection requires the 'cryptography' package"
            )

    def protect(self, action_chunk: ArrayLike, key_id: str) -> ProtectedActionEnvelope:
        self._require_crypto()
        created = _now(self._clock)
        key = self.key_store.get(key_id, now=created)
        action = _validated_action(action_chunk)
        nonce = bytes(self._nonce_source(12))
        if len(nonce) != 12:
            raise ProtectionError("nonce source did not return exactly 12 bytes")
        shape = action.shape  # type: ignore[assignment]
        associated_data = _aad(
            version=1,
            algorithm="AES-256-GCM",
            key_id=key.key_id,
            created_at=created,
            action_shape=shape,
        )
        assert AESGCM is not None
        ciphertext = AESGCM(key.key).encrypt(nonce, _action_bytes(action), associated_data)
        return ProtectedActionEnvelope(
            version=1,
            algorithm="AES-256-GCM",
            key_id=key.key_id,
            nonce=nonce,
            ciphertext=ciphertext,
            created_at=created,
            action_shape=shape,
        )

    encrypt = protect

    def unprotect(self, envelope: ProtectedActionEnvelope) -> np.ndarray:
        self._require_crypto()
        if not isinstance(envelope, ProtectedActionEnvelope):
            raise MalformedEnvelopeError("input is not a protected high-risk envelope")
        now = _now(self._clock)
        key = self.key_store.get(envelope.key_id, now=now)
        # AES-GCM authenticates these metadata as AAD, including the shape used
        # to interpret plaintext bytes.
        associated_data = _aad(
            version=envelope.version,
            algorithm=envelope.algorithm,
            key_id=envelope.key_id,
            created_at=envelope.created_at,
            action_shape=envelope.action_shape,
        )
        try:
            assert AESGCM is not None
            plaintext = AESGCM(key.key).decrypt(
                envelope.nonce, envelope.ciphertext, associated_data
            )
        except (InvalidTag, ValueError) as exc:
            raise AuthenticationError("protected action authentication failed") from exc
        expected_bytes = int(np.prod(envelope.action_shape)) * 8
        if len(plaintext) != expected_bytes:
            raise MalformedEnvelopeError("protected action length does not match its shape")
        action = np.frombuffer(plaintext, dtype="<f8").reshape(envelope.action_shape).copy()
        if not np.all(np.isfinite(action)):
            raise MalformedEnvelopeError("decrypted action contains NaN or infinity")
        action.setflags(write=False)
        return action

    decrypt = unprotect


ActionProtector = AESGCMProtector
ProtectedEnvelope = ProtectedActionEnvelope


class Authorizer(Protocol):
    def authorize(
        self, context: AuthorizationContext | Mapping[str, object]
    ) -> AuthorizationDecision: ...


class OfflineContextAuthorizer:
    """Deterministic Auth substitute for offline tests and fixture scenarios.

    The local policy mirrors SST syntax but does not claim to validate a live
    SST deployment.  An allowed request creates a session key and installs the
    same key at both endpoints, like successful Auth distribution.
    """

    def __init__(
        self,
        policy: ContextPolicy,
        monitor_store: InMemorySessionKeyStore,
        actuator_store: InMemorySessionKeyStore,
        *,
        validity_seconds: float = 30.0,
        clock: Clock = time.time,
        key_factory: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        self.policy = policy
        self.monitor_store = monitor_store
        self.actuator_store = actuator_store
        self.validity_seconds = float(validity_seconds)
        if not np.isfinite(self.validity_seconds) or self.validity_seconds <= 0:
            raise ValueError("validity_seconds must be finite and positive")
        self._clock = clock
        self._key_factory = key_factory
        self._cached_by_context: dict[AuthorizationContext, str] = {}

    def authorize(
        self, context: AuthorizationContext | Mapping[str, object]
    ) -> AuthorizationDecision:
        started = time.perf_counter_ns()
        try:
            parsed = (
                context
                if isinstance(context, AuthorizationContext)
                else AuthorizationContext.from_mapping(context)
            )
        except (TypeError, ValueError) as exc:
            return self._decision(
                AuthorizationOutcome.DENIED,
                f"malformed context: {exc}",
                started=started,
            )
        policy_decision = self.policy.evaluate(parsed)
        if not policy_decision.allowed:
            return self._decision(
                AuthorizationOutcome.DENIED,
                policy_decision.reason,
                started=started,
            )

        cached_id = self._cached_by_context.get(parsed)
        if cached_id is not None:
            try:
                self.monitor_store.get(cached_id)
                self.actuator_store.get(cached_id)
            except ProtectionError:
                self._cached_by_context.pop(parsed, None)
            else:
                return self._decision(
                    AuthorizationOutcome.ALLOWED,
                    "authorized using a cached valid session key",
                    key_id=cached_id,
                    reused=True,
                    started=started,
                )

        key = SessionKey.generate(
            self.validity_seconds,
            clock=self._clock,
            key=bytes(self._key_factory(32)),
        )
        self.monitor_store.install(key)
        self.actuator_store.install(key)
        self._cached_by_context[parsed] = key.key_id
        return self._decision(
            AuthorizationOutcome.ALLOWED,
            "context authorized and a session key was issued",
            key_id=key.key_id,
            reused=False,
            started=started,
        )

    @staticmethod
    def _decision(
        outcome: AuthorizationOutcome,
        reason: str,
        *,
        started: int,
        key_id: str | None = None,
        reused: bool = False,
    ) -> AuthorizationDecision:
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return AuthorizationDecision(outcome, reason, key_id, reused, elapsed_ms)

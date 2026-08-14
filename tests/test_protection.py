from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _analysis import low_risk_analysis

from safe_agent_actuation.context import ContextPolicy, DeterministicContextProvider
from safe_agent_actuation.protection import (
    AESGCMProtector,
    AuthenticationError,
    ExpiredKeyError,
    InMemorySessionKeyStore,
    MalformedEnvelopeError,
    OfflineContextAuthorizer,
    ProtectedActionEnvelope,
    SessionKey,
    TrustedLowRiskChannel,
    UnknownKeyError,
)
from safe_agent_actuation.types import AuthorizationOutcome


class ManualClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_session_key_expiration_uses_injected_clock() -> None:
    clock = ManualClock()
    store = InMemorySessionKeyStore(clock=clock)
    key = SessionKey.generate(2.0, clock=clock, key=b"k" * 32, key_id="test-key")
    store.install(key)
    assert store.get("test-key") is key
    clock.advance(2.0)
    with pytest.raises(ExpiredKeyError, match="expired"):
        store.get("test-key")
    with pytest.raises(UnknownKeyError, match="unknown"):
        store.get("not-installed")


def test_offline_authorizer_allows_denies_and_reuses_valid_keys() -> None:
    clock = ManualClock()
    monitor_store = InMemorySessionKeyStore(clock=clock)
    actuator_store = InMemorySessionKeyStore(clock=clock)
    authorizer = OfflineContextAuthorizer(
        ContextPolicy.course_default(),
        monitor_store,
        actuator_store,
        clock=clock,
        validity_seconds=3.0,
        key_factory=lambda size: b"a" * size,
    )
    denied = authorizer.authorize(DeterministicContextProvider.denied_nearby_person().build())
    assert denied.outcome is AuthorizationOutcome.DENIED
    assert denied.key_id is None
    assert monitor_store.public_key_ids() == ()

    first = authorizer.authorize(DeterministicContextProvider.allowed().build())
    second = authorizer.authorize(DeterministicContextProvider.allowed().build())
    assert first.allowed and second.allowed
    assert not first.session_key_reused
    assert second.session_key_reused
    assert first.key_id == second.key_id
    assert actuator_store.has_valid(first.key_id or "")

    clock.advance(3.0)
    third = authorizer.authorize(DeterministicContextProvider.allowed().build())
    assert third.allowed
    assert not third.session_key_reused


def test_malformed_authorization_context_is_denied() -> None:
    store_a = InMemorySessionKeyStore()
    store_b = InMemorySessionKeyStore()
    decision = OfflineContextAuthorizer(ContextPolicy.course_default(), store_a, store_b).authorize(
        {"nearby_people": 0}
    )
    assert decision.outcome is AuthorizationOutcome.DENIED
    assert "malformed" in decision.reason


def test_trusted_low_risk_channel_rejects_tampering() -> None:
    clock = ManualClock()
    channel = TrustedLowRiskChannel(clock=clock, secret=b"s" * 32)
    analysis = low_risk_analysis(("dimension_0", "dimension_1"))
    envelope = channel.issue([[0.0, 0.0]], analysis)
    np.testing.assert_array_equal(channel.verify(envelope), [[0.0, 0.0]])
    tampered = replace(envelope, action_chunk=np.array([[0.05, 0.0]]))
    with pytest.raises(AuthenticationError, match="authentication"):
        channel.verify(tampered)


def _protector(clock: ManualClock) -> tuple[AESGCMProtector, SessionKey]:
    if not AESGCMProtector.available():
        pytest.skip("cryptography is unavailable")
    store = InMemorySessionKeyStore(clock=clock)
    key = SessionKey.generate(2.0, clock=clock, key=b"x" * 32, key_id="action-key")
    store.install(key)
    return AESGCMProtector(store, clock=clock, nonce_source=lambda size: b"n" * size), key


def test_aes_gcm_round_trip_and_transport_serialization() -> None:
    clock = ManualClock()
    protector, _ = _protector(clock)
    action = np.array([[0.1, -0.2], [0.3, -0.4]])
    envelope = protector.protect(action, "action-key")
    decoded = ProtectedActionEnvelope.from_dict(envelope.to_dict())
    np.testing.assert_array_equal(protector.unprotect(decoded), action)
    assert "xxxxxxxx" not in repr(envelope)
    assert "key_material" not in envelope.to_dict()


def test_tampered_high_risk_envelope_is_rejected() -> None:
    clock = ManualClock()
    protector, _ = _protector(clock)
    envelope = protector.protect([[0.1, 0.2]], "action-key")
    corrupted = bytearray(envelope.ciphertext)
    corrupted[0] ^= 1
    tampered = replace(envelope, ciphertext=bytes(corrupted))
    with pytest.raises(AuthenticationError, match="authentication"):
        protector.unprotect(tampered)


def test_expired_high_risk_envelope_is_rejected_before_decryption() -> None:
    clock = ManualClock()
    protector, _ = _protector(clock)
    envelope = protector.protect([[0.1, 0.2]], "action-key")
    clock.advance(2.0)
    with pytest.raises(ExpiredKeyError):
        protector.unprotect(envelope)


def test_malformed_protected_transport_is_rejected() -> None:
    with pytest.raises(MalformedEnvelopeError):
        ProtectedActionEnvelope.from_dict({"key_id": "fake"})

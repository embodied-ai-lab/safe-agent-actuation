"""Course scenario orchestration for fixture-driven monitoring experiments.

Every scenario follows the same shape: load a committed fixture, analyze only
the executed prefix, enforce the correct authorization path, deliver or
withhold the action at the actuator boundary, and write a structured, secret-
free action record.  High-risk authorization is real SST by default; the
offline authorizer is allowed only for tests and explicitly labeled
diagnostics.
"""

from __future__ import annotations

import datetime as _datetime
import json
import queue
import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actuator import RecordingSimulator, SafeActuator
from .context import AuthorizationContext, ContextPolicy
from .course_config import (
    SST_RUNTIME_DIR,
    CourseConfigError,
    fixture_path,
    load_context,
    load_physical_limits,
    scenario_settings,
)
from .fixtures import ActionFixture, load_action_fixture
from .motion import analyze_motion
from .protection import (
    AESGCMProtector,
    InMemorySessionKeyStore,
    OfflineContextAuthorizer,
    TrustedLowRiskChannel,
)
from .records import ActionRecord, LatencyRecord, write_action_record
from .sst_adapter import (
    SSTActuatorTransport,
    SSTMonitorTransport,
    SSTReceiveResult,
    SSTSendResult,
)
from .types import (
    ActuationResult,
    ActuatorDecision,
    AuthorizationDecision,
    AuthorizationOutcome,
    MotionAnalysis,
    PhysicalLimits,
)

MODEL_CONFIGURATION_FOR_FIXTURES = "none (committed deterministic fixture, no model inference)"
SST_ACTUATOR_HOST = "127.0.0.1"
SST_ACTUATOR_PORT = 23100
OFFLINE_AUTHORIZATION_MODE = "offline_diagnostic (OfflineContextAuthorizer, not live SST)"
LIVE_AUTHORIZATION_MODE = "live_sst (pinned IoTAuth Auth, Python entity API)"


class ScenarioError(RuntimeError):
    """Infrastructure or unexpected-outcome failure; the command must exit nonzero."""


def utc_run_id(scenario: str) -> str:
    stamp = _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario}-{stamp}"


def run_directory(root: Path, scenario: str) -> Path:
    directory = root / "results" / scenario
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass(frozen=True, slots=True)
class LoadedScenarioAction:
    """A fixture plus the derived values every scenario consumes."""

    fixture: ActionFixture
    fixture_file: str
    executed_prefix: np.ndarray
    executed_horizon: int
    limits: PhysicalLimits


def load_scenario_action(root: Path, scenario: str) -> LoadedScenarioAction:
    """Resolve the committed fixture for a scenario and validate its limits."""

    settings = scenario_settings(root, scenario)
    fixture_file = settings.get("fixture")
    if not isinstance(fixture_file, str) or not fixture_file:
        raise CourseConfigError(f"scenario {scenario!r} does not declare a fixture")
    fixture = load_action_fixture(fixture_path(root, fixture_file))
    limits = load_physical_limits(root)
    if fixture.dimension_names != limits.dimension_names:
        raise ScenarioError(
            f"fixture {fixture_file} dimension names do not match configs/motion/aloha_limits.json"
        )
    metadata_horizon = fixture.metadata.get("executed_horizon")
    horizon = int(metadata_horizon) if metadata_horizon else fixture.action.shape[0]
    if not 1 <= horizon <= fixture.action.shape[0]:
        raise ScenarioError(f"fixture {fixture_file} has an invalid executed horizon {horizon}")
    prefix = np.array(fixture.action[:horizon], dtype=np.float64, copy=True)
    return LoadedScenarioAction(
        fixture=fixture,
        fixture_file=fixture_file,
        executed_prefix=prefix,
        executed_horizon=horizon,
        limits=limits,
    )


def analyze_prefix(action: LoadedScenarioAction, threshold: float) -> MotionAnalysis:
    """Run the Joint Motion Analyzer on only the executed prefix."""

    return analyze_motion(
        action_chunk=action.executed_prefix,
        current_state=action.fixture.current_state,
        lower_limits=action.limits.lower,
        upper_limits=action.limits.upper,
        threshold=threshold,
        dimension_names=action.limits.dimension_names,
    )


def _offline_stack(
    validity_seconds: float = 30.0,
) -> tuple[OfflineContextAuthorizer, AESGCMProtector, AESGCMProtector, TrustedLowRiskChannel]:
    monitor_store = InMemorySessionKeyStore()
    actuator_store = InMemorySessionKeyStore()
    authorizer = OfflineContextAuthorizer(
        ContextPolicy.course_default(),
        monitor_store,
        actuator_store,
        validity_seconds=validity_seconds,
    )
    return (
        authorizer,
        AESGCMProtector(monitor_store),
        AESGCMProtector(actuator_store),
        TrustedLowRiskChannel(),
    )


def _record_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "action_record.json", directory / "action_record.npz"


def _base_record(
    *,
    run_id: str,
    scenario: str,
    action: LoadedScenarioAction,
    analysis: MotionAnalysis,
    context: AuthorizationContext | None,
    authorization_requested: bool,
    authorization: AuthorizationDecision | None,
    latencies: LatencyRecord,
    actuation: ActuationResult,
    simulator_outcome: str,
    metadata: Mapping[str, Any],
) -> ActionRecord:
    return ActionRecord(
        run_id=run_id,
        timestep=0,
        task_prompt=context.task_name if context is not None else "transfer_cube",
        model_configuration=MODEL_CONFIGURATION_FOR_FIXTURES,
        action_source=action.fixture.source_label,
        raw_fast_token_ids=action.fixture.raw_fast_token_ids,
        decoded_action_shape=tuple(action.executed_prefix.shape),  # type: ignore[arg-type]
        executed_horizon=action.executed_horizon,
        current_observed_joint_state=tuple(
            float(value) for value in np.asarray(action.fixture.current_state)
        ),
        physical_limits_source=action.limits.source,
        motion_analysis=analysis,
        context=context,
        authorization_requested=authorization_requested,
        authorization=authorization,
        latencies=latencies,
        actuation=actuation,
        simulator_outcome=simulator_outcome,
        metadata={"scenario": scenario, "fixture_file": action.fixture_file, **metadata},
    )


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """What the CLI reports and uses for its exit status."""

    scenario: str
    run_id: str
    record_path: Path
    expected: bool
    summary: dict[str, Any]


def _write(
    directory: Path, record: ActionRecord, action_chunk: np.ndarray | None
) -> Path:
    json_path, npz_path = _record_paths(directory)
    write_action_record(record, json_path, action_chunk=action_chunk, npz_path=npz_path)
    return json_path


def run_low_risk(root: Path, *, threshold: float) -> ScenarioOutcome:
    """Trusted low-risk direct path; SST must not be involved."""

    scenario = "low-risk"
    run_id = utc_run_id(scenario)
    directory = run_directory(root, scenario)
    action = load_scenario_action(root, scenario)
    monitor_started = time.perf_counter()
    analysis = analyze_prefix(action, threshold)
    monitoring_ms = (time.perf_counter() - monitor_started) * 1000.0
    if not analysis.is_low_risk:
        raise ScenarioError(
            "low-risk scenario requires a valid below-threshold fixture; got "
            f"classification={analysis.risk_classification} score={analysis.motion_score}"
        )

    _, _, actuator_protector, channel = _offline_stack()
    simulator = RecordingSimulator()
    actuator = SafeActuator(
        simulator,
        high_risk_protector=actuator_protector,
        low_risk_channel=channel,
    )
    envelope = channel.issue(action.executed_prefix, analysis)
    actuation = actuator.actuate(envelope, action.fixture.current_state)
    total_ms = (time.perf_counter() - monitor_started) * 1000.0
    if not actuation.executed or simulator.execute_count != 1 or simulator.hold_count != 0:
        raise ScenarioError(f"trusted low-risk path did not execute: {actuation.reason}")

    record = _base_record(
        run_id=run_id,
        scenario=scenario,
        action=action,
        analysis=analysis,
        context=None,
        authorization_requested=False,
        authorization=AuthorizationDecision(
            AuthorizationOutcome.NOT_REQUESTED,
            "low-risk classification used the trusted direct path; SST was not involved",
        ),
        latencies=LatencyRecord(
            monitoring_ms=monitoring_ms,
            total_monitor_to_actuator_ms=total_ms,
        ),
        actuation=actuation,
        simulator_outcome="executed_in_recording_harness",
        metadata={"sst_required": False, "authorization_mode": "not_requested"},
    )
    record_path = _write(directory, record, action.executed_prefix)
    return ScenarioOutcome(
        scenario=scenario,
        run_id=run_id,
        record_path=record_path,
        expected=True,
        summary={
            "classification": analysis.risk_classification,
            "motion_score": analysis.motion_score,
            "max_dimension": analysis.max_dimension,
            "actuator_decision": actuation.decision.value,
            "simulator_executions": simulator.execute_count,
        },
    )


# ---------------------------------------------------------------------------
# Live SST plumbing
# ---------------------------------------------------------------------------


def sst_config_paths(root: Path) -> tuple[Path, Path]:
    configs = root / SST_RUNTIME_DIR / "configs"
    return configs / "monitor.config", configs / "actuator.config"


def require_sst_runtime(root: Path) -> tuple[Path, Path]:
    monitor_config, actuator_config = sst_config_paths(root)
    for path in (monitor_config, actuator_config):
        if not path.is_file():
            raise ScenarioError(
                f"SST runtime state is missing ({path}). Run: make build-auth && "
                "make generate-sst && make auth-start"
            )
    return monitor_config, actuator_config


def require_auth_running(root: Path, *, host: str = "127.0.0.1", port: int = 22900) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        if probe.connect_ex((host, port)) != 0:
            raise ScenarioError(
                f"Auth is not reachable on {host}:{port}. Run: make auth-start"
            )
    finally:
        probe.close()


class _ActuatorReceiver:
    """Run one SST receive on a worker thread with the port pre-bound."""

    def __init__(self, transport: SSTActuatorTransport) -> None:
        self.transport = transport
        self._results: queue.Queue[SSTReceiveResult | BaseException] = queue.Queue()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _ActuatorReceiver:
        self.transport.listen()
        thread = threading.Thread(target=self._run, name="sst-actuator-receiver", daemon=True)
        thread.start()
        self._thread = thread
        return self

    def _run(self) -> None:
        try:
            self._results.put(self.transport.receive_once())
        except BaseException as exc:  # noqa: BLE001 - delivered to the joining thread.
            self._results.put(exc)

    def result(self, timeout: float) -> SSTReceiveResult:
        try:
            value = self._results.get(timeout=timeout)
        except queue.Empty as exc:
            raise ScenarioError("SST actuator receiver did not finish in time") from exc
        if isinstance(value, BaseException):
            raise ScenarioError(f"SST actuator receiver failed: {value}") from value
        return value

    def __exit__(self, *_exc: object) -> None:
        self.transport.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)


# The next two functions are the graded monitor-send and actuator-enforcement
# steps.  They are deliberately small and covered by offline unit tests.


def send_high_risk_action_live(
    transport: SSTMonitorTransport,
    action_chunk: np.ndarray,
    *,
    run_id: str,
    timestep: int,
    context: AuthorizationContext,
) -> SSTSendResult:
    """Request context authorization and send the protected high-risk action.

    The transport requests or reuses an SST session key for the ``Actuators``
    group using the supplied context, encrypts the payload with the pinned SST
    secure channel, and reports authorization, reuse, and latency facts.
    """

    # TODO(student): deliver the high-risk action through SST.
    # Input: transport (SSTMonitorTransport), action_chunk [H, D] float array,
    #   run_id, timestep, and the AuthorizationContext for this request.
    # Output: return the SSTSendResult from exactly one transport.send call,
    #   passing context.to_dict() so Auth can evaluate the course context.
    # Invalid input: let transport.send raise SSTAdapterError for malformed
    #   actions or contexts; do not catch it here.
    # Verify: pytest tests/test_enforcement.py -k monitor_send ; make high-risk-allow
    raise NotImplementedError("TODO(student): monitor-protected-send")


def enforce_received_action(
    payload: Mapping[str, Any] | None,
    *,
    expected_run_id: str,
    current_state: np.ndarray,
    limits: PhysicalLimits,
    simulator: RecordingSimulator,
) -> ActuationResult:
    """Actuator-side enforcement after an authenticated SST receive.

    ``payload`` is the decoded protected-action body produced by
    :func:`safe_agent_actuation.sst_adapter.decode_protected_action`, or
    ``None`` when no authenticated action arrived.  The function must fail
    closed: anything except a fully valid action holds the current pose.
    """

    started = time.perf_counter()

    def _hold(reason: str) -> ActuationResult:
        simulator.hold(np.asarray(current_state, dtype=np.float64))
        return ActuationResult(
            decision=ActuatorDecision.HOLD,
            reason=reason,
            path="sst_protected_high_risk",
            action_shape=None,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # TODO(student): verify the received action before any simulator call.
    # Input: payload (mapping with run_id and a finite [H, D] action_chunk) or
    #   None; expected_run_id; current_state [D]; limits; simulator.
    # Output: ActuationResult EXECUTE only when the payload matches the run,
    #   the dimension matches current_state, and every command is inside the
    #   physical limits; call simulator.execute exactly once in that case.
    # Invalid input: return _hold(reason) without touching simulator.execute.
    # Verify: pytest tests/test_enforcement.py -k enforce ; make high-risk-deny
    raise NotImplementedError("TODO(student): actuator-enforce")


def _authorization_from_send(
    send: SSTSendResult, *, denied_expected: bool
) -> AuthorizationDecision:
    if send.authorized:
        outcome = AuthorizationOutcome.ALLOWED
    elif "AuthConnectionError" in send.reason:
        outcome = AuthorizationOutcome.ERROR
    else:
        outcome = AuthorizationOutcome.DENIED if denied_expected else AuthorizationOutcome.ERROR
    return AuthorizationDecision(
        outcome=outcome,
        reason=send.reason,
        key_id=send.key_id if send.authorized else None,
        session_key_reused=send.key_reused,
        latency_ms=send.authorization_latency_ms,
    )


def run_high_risk_live(
    root: Path,
    *,
    scenario: str,
    threshold: float,
    context_name: str,
    expect_allowed: bool,
) -> ScenarioOutcome:
    """Graded high-risk path through the real pinned SST deployment."""

    run_id = utc_run_id(scenario)
    directory = run_directory(root, scenario)
    action = load_scenario_action(root, scenario)
    context = load_context(root, context_name)
    monitor_config, actuator_config = require_sst_runtime(root)
    require_auth_running(root)

    monitor_started = time.perf_counter()
    analysis = analyze_prefix(action, threshold)
    monitoring_ms = (time.perf_counter() - monitor_started) * 1000.0
    if not analysis.high_risk:
        raise ScenarioError(
            f"{scenario} requires an at-or-above-threshold fixture; got "
            f"score={analysis.motion_score} threshold={threshold}"
        )

    simulator = RecordingSimulator()
    transport = SSTMonitorTransport(
        monitor_config, host=SST_ACTUATOR_HOST, port=SST_ACTUATOR_PORT
    )
    receiver_transport = SSTActuatorTransport(
        actuator_config,
        host=SST_ACTUATOR_HOST,
        port=SST_ACTUATOR_PORT,
        accept_timeout=10.0,
    )

    if expect_allowed:
        with _ActuatorReceiver(receiver_transport) as receiver:
            send = send_high_risk_action_live(
                transport,
                action.executed_prefix,
                run_id=run_id,
                timestep=0,
                context=context,
            )
            received = receiver.result(timeout=20.0)
    else:
        # A denied context must never open the actuator path, so no receiver
        # runs; an unexpected successful send fails on connection instead.
        send = send_high_risk_action_live(
            transport,
            action.executed_prefix,
            run_id=run_id,
            timestep=0,
            context=context,
        )
        received = SSTReceiveResult(
            authenticated=False,
            received=False,
            receive_decryption_latency_ms=0.0,
            reason="not started: authorization was denied before any transfer",
        )

    authorization = _authorization_from_send(send, denied_expected=not expect_allowed)
    if authorization.outcome is AuthorizationOutcome.ERROR:
        raise ScenarioError(
            f"SST authorization infrastructure failure: {send.reason}. "
            "Check make auth-start and runtime/sst/logs/auth.log"
        )

    actuation = enforce_received_action(
        received.payload if received.received else None,
        expected_run_id=run_id,
        current_state=np.asarray(action.fixture.current_state, dtype=np.float64),
        limits=action.limits,
        simulator=simulator,
    )
    total_ms = (time.perf_counter() - monitor_started) * 1000.0

    if expect_allowed:
        expected = (
            send.authorized
            and send.sent
            and received.authenticated
            and actuation.executed
            and simulator.execute_count == 1
        )
        failure_reason = None if expected else (
            f"allowed run did not complete: send={send.reason}; "
            f"receive={received.reason}; actuation={actuation.reason}"
        )
    else:
        expected = (
            not send.authorized
            and not send.sent
            and actuation.held
            and simulator.execute_count == 0
        )
        failure_reason = None if expected else (
            "denied run unexpectedly progressed: "
            f"authorized={send.authorized} sent={send.sent} decision={actuation.decision.value}"
        )

    record = _base_record(
        run_id=run_id,
        scenario=scenario,
        action=action,
        analysis=analysis,
        context=context,
        authorization_requested=True,
        authorization=authorization,
        latencies=LatencyRecord(
            monitoring_ms=monitoring_ms,
            authorization_ms=send.authorization_latency_ms,
            encryption_ms=send.encryption_send_latency_ms,
            decryption_ms=received.receive_decryption_latency_ms if received.received else None,
            total_monitor_to_actuator_ms=total_ms,
        ),
        actuation=actuation,
        simulator_outcome=(
            "executed_in_recording_harness" if actuation.executed
            else "held_in_recording_harness"
        ),
        metadata={
            "sst_required": True,
            "authorization_mode": LIVE_AUTHORIZATION_MODE,
            "context_name": context_name,
            "handshake_latency_ms": send.handshake_latency_ms,
            "send_latency_ms": send.encryption_send_latency_ms,
            "sst_send_reason": send.reason,
            "sst_receive_reason": received.reason,
            "session_key_reused": send.key_reused,
            "failure_reason": failure_reason,
        },
    )
    record_path = _write(directory, record, action.executed_prefix)
    if not expected:
        raise ScenarioError(failure_reason or f"{scenario} did not reach its expected outcome")
    return ScenarioOutcome(
        scenario=scenario,
        run_id=run_id,
        record_path=record_path,
        expected=True,
        summary={
            "classification": analysis.risk_classification,
            "motion_score": analysis.motion_score,
            "authorization": authorization.outcome.value,
            "session_key_reused": send.key_reused,
            "actuator_decision": actuation.decision.value,
            "simulator_executions": simulator.execute_count,
            "authorization_latency_ms": send.authorization_latency_ms,
            "handshake_latency_ms": send.handshake_latency_ms,
        },
    )


def run_high_risk_offline_diagnostic(
    root: Path,
    *,
    scenario: str,
    threshold: float,
    context_name: str,
    expect_allowed: bool,
) -> ScenarioOutcome:
    """Explicitly labeled offline diagnostic; never a live SST claim."""

    run_id = utc_run_id(scenario)
    directory = run_directory(root, scenario)
    action = load_scenario_action(root, scenario)
    context = load_context(root, context_name)

    monitor_started = time.perf_counter()
    analysis = analyze_prefix(action, threshold)
    monitoring_ms = (time.perf_counter() - monitor_started) * 1000.0
    if not analysis.high_risk:
        raise ScenarioError(
            f"{scenario} requires an at-or-above-threshold fixture; got "
            f"score={analysis.motion_score}"
        )

    authorizer, monitor_protector, actuator_protector, channel = _offline_stack()
    simulator = RecordingSimulator()
    actuator = SafeActuator(
        simulator, high_risk_protector=actuator_protector, low_risk_channel=channel
    )
    decision = authorizer.authorize(context)

    if decision.allowed:
        assert decision.key_id is not None
        envelope = monitor_protector.protect(action.executed_prefix, decision.key_id)
        actuation = actuator.actuate(envelope, action.fixture.current_state)
    else:
        simulator.hold(np.asarray(action.fixture.current_state, dtype=np.float64))
        actuation = ActuationResult(
            decision=ActuatorDecision.HOLD,
            reason="authorization denied; the monitor withheld the action before protection",
            path="withheld_before_actuator",
        )
    total_ms = (time.perf_counter() - monitor_started) * 1000.0

    expected = (
        decision.allowed == expect_allowed
        and actuation.executed == expect_allowed
        and simulator.execute_count == (1 if expect_allowed else 0)
    )
    record = _base_record(
        run_id=run_id,
        scenario=scenario,
        action=action,
        analysis=analysis,
        context=context,
        authorization_requested=True,
        authorization=decision,
        latencies=LatencyRecord(
            monitoring_ms=monitoring_ms,
            authorization_ms=decision.latency_ms,
            total_monitor_to_actuator_ms=total_ms,
        ),
        actuation=actuation,
        simulator_outcome=(
            "executed_in_recording_harness" if actuation.executed
            else "held_in_recording_harness"
        ),
        metadata={
            "sst_required": False,
            "authorization_mode": OFFLINE_AUTHORIZATION_MODE,
            "context_name": context_name,
        },
    )
    record_path = _write(directory, record, action.executed_prefix)
    if not expected:
        raise ScenarioError(
            f"offline diagnostic for {scenario} did not reach its expected outcome: "
            f"authorization={decision.outcome.value} actuation={actuation.decision.value}"
        )
    return ScenarioOutcome(
        scenario=scenario,
        run_id=run_id,
        record_path=record_path,
        expected=True,
        summary={
            "mode": "offline_diagnostic",
            "classification": analysis.risk_classification,
            "motion_score": analysis.motion_score,
            "authorization": decision.outcome.value,
            "actuator_decision": actuation.decision.value,
            "simulator_executions": simulator.execute_count,
        },
    )


@dataclass(frozen=True, slots=True)
class _LiveSendRound:
    step: str
    authorized: bool
    sent: bool
    key_reused: bool
    received: bool
    authorization_latency_ms: float
    handshake_latency_ms: float
    send_latency_ms: float
    reason: str


def _live_send_round(
    transport: SSTMonitorTransport,
    receiver_transport: SSTActuatorTransport,
    action_chunk: np.ndarray,
    *,
    step: str,
    run_id: str,
    timestep: int,
    context: AuthorizationContext,
    expect_sent: bool,
) -> _LiveSendRound:
    received = False
    if expect_sent:
        with _ActuatorReceiver(receiver_transport) as receiver:
            send = send_high_risk_action_live(
                transport, action_chunk, run_id=run_id, timestep=timestep, context=context
            )
            if send.sent:
                result = receiver.result(timeout=20.0)
                received = result.received
    else:
        send = send_high_risk_action_live(
            transport, action_chunk, run_id=run_id, timestep=timestep, context=context
        )
    return _LiveSendRound(
        step=step,
        authorized=send.authorized,
        sent=send.sent,
        key_reused=send.key_reused,
        received=received,
        authorization_latency_ms=send.authorization_latency_ms,
        handshake_latency_ms=send.handshake_latency_ms,
        send_latency_ms=send.encryption_send_latency_ms,
        reason=send.reason,
    )


def run_key_validity_live_trial(
    root: Path,
    *,
    label: str,
    output_dir: Path,
    expiry_margin_seconds: float = 3.0,
) -> dict[str, Any]:
    """One real-SST validity trial for the currently generated relative validity.

    The controlled sequence is: authorize in the allowed context, reuse the
    valid key, keep sending after the context changes to a denied state while
    the key is still valid (the security window), let the key expire, confirm a
    denied context is then refused, and finally reissue in the allowed context.
    """

    scenario = "grad-key-validity"
    run_id = utc_run_id(f"{scenario}-{label}")
    action = load_scenario_action(root, "high-risk-allow")
    monitor_config, actuator_config = require_sst_runtime(root)
    require_auth_running(root)
    allowed = load_context(root, "allowed")
    denied = load_context(root, "denied")

    transport = SSTMonitorTransport(
        monitor_config, host=SST_ACTUATOR_HOST, port=SST_ACTUATOR_PORT
    )

    def receiver() -> SSTActuatorTransport:
        return SSTActuatorTransport(
            actuator_config,
            host=SST_ACTUATOR_HOST,
            port=SST_ACTUATOR_PORT,
            accept_timeout=10.0,
        )

    rounds: list[_LiveSendRound] = []
    rounds.append(
        _live_send_round(
            transport, receiver(), action.executed_prefix,
            step="authorize_allowed", run_id=run_id, timestep=0,
            context=allowed, expect_sent=True,
        )
    )
    validity = transport.cached_key_validity()
    rounds.append(
        _live_send_round(
            transport, receiver(), action.executed_prefix,
            step="reuse_valid_key", run_id=run_id, timestep=1,
            context=allowed, expect_sent=True,
        )
    )
    rounds.append(
        _live_send_round(
            transport, receiver(), action.executed_prefix,
            step="security_window_denied_context", run_id=run_id, timestep=2,
            context=denied, expect_sent=True,
        )
    )

    rel_validity_ms = validity.get("rel_validity_ms")
    sleep_seconds = (
        (float(rel_validity_ms) / 1000.0 + expiry_margin_seconds)
        if rel_validity_ms
        else expiry_margin_seconds
    )
    time.sleep(sleep_seconds)

    rounds.append(
        _live_send_round(
            transport, receiver(), action.executed_prefix,
            step="denied_after_expiry", run_id=run_id, timestep=3,
            context=denied, expect_sent=False,
        )
    )
    rounds.append(
        _live_send_round(
            transport, receiver(), action.executed_prefix,
            step="reissue_allowed", run_id=run_id, timestep=4,
            context=allowed, expect_sent=True,
        )
    )

    security_window_sends = sum(
        1 for item in rounds if item.step == "security_window_denied_context" and item.sent
    )
    denied_after_expiry = sum(
        1 for item in rounds if item.step == "denied_after_expiry" and not item.authorized
    )
    sent_latencies = [item.authorization_latency_ms for item in rounds if item.authorized]
    trial = {
        "label": label,
        "run_id": run_id,
        "authorization_mode": LIVE_AUTHORIZATION_MODE,
        "relative_validity_ms": rel_validity_ms,
        "absolute_validity_ms": validity.get("abs_validity_ms"),
        "key_requests": transport.key_requests,
        "key_reuses": sum(1 for item in rounds if item.key_reused),
        "allowed_while_valid": sum(1 for item in rounds if item.sent),
        "security_window_sends": security_window_sends,
        "denied_after_expiry": denied_after_expiry,
        "reissued_after_expiry": any(
            item.step == "reissue_allowed" and item.sent for item in rounds
        ),
        "average_authorization_latency_ms": (
            float(np.mean([item.authorization_latency_ms for item in rounds]))
        ),
        "average_handshake_latency_ms": (
            float(np.mean([item.handshake_latency_ms for item in rounds if item.sent]))
            if any(item.sent for item in rounds) else 0.0
        ),
        "average_send_latency_ms": (
            float(np.mean(sent_latencies)) if sent_latencies else 0.0
        ),
        "rounds": [
            {
                "step": item.step,
                "authorized": item.authorized,
                "sent": item.sent,
                "key_reused": item.key_reused,
                "received": item.received,
                "reason": item.reason,
            }
            for item in rounds
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "live_trials.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trial, sort_keys=True) + "\n")

    expected = (
        security_window_sends == 1
        and denied_after_expiry == 1
        and trial["reissued_after_expiry"]
        and transport.key_requests >= 2
    )
    trial["expected"] = expected
    if not expected:
        raise ScenarioError(
            f"live key-validity trial {label} did not reach its expected sequence: {trial}"
        )
    return trial


def summarize_key_validity_live(output_dir: Path) -> dict[str, Any]:
    """Aggregate real-SST validity trials into CSV, JSON, and a plot."""

    jsonl_path = output_dir / "live_trials.jsonl"
    if not jsonl_path.is_file():
        raise ScenarioError(f"no live key-validity trials found at {jsonl_path}")
    trials = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not trials:
        raise ScenarioError("live key-validity trial file is empty")

    import csv

    fieldnames = [
        "label",
        "relative_validity_ms",
        "key_requests",
        "key_reuses",
        "security_window_sends",
        "denied_after_expiry",
        "reissued_after_expiry",
        "average_authorization_latency_ms",
        "average_handshake_latency_ms",
        "average_send_latency_ms",
    ]
    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "analysis": "session_key_validity",
        "authorization_mode": LIVE_AUTHORIZATION_MODE,
        "relative_validity_ms": [trial.get("relative_validity_ms") for trial in trials],
        "trials": trials,
        "security_window": (
            "A session key issued in the allowed context stays usable until it "
            "expires. Sends in a denied context still succeed while the key is "
            "valid because SST checks the context at key request time, not per "
            "message. Shorter relative validity shortens this window at the cost "
            "of more key requests and more authorization latency."
        ),
        "limitation": (
            "The comparison uses three separately generated Auth deployments, one "
            "per relative validity. Wall-clock expiry waits make the measured "
            "latency include scheduling overhead."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_validity(trials, output_dir / "key_validity.png")
    return {"csv": str(csv_path), "summary": str(summary_path), "trials": trials}


def _plot_validity(trials: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(trial.get("label")) for trial in trials]
    requests = [int(trial.get("key_requests", 0)) for trial in trials]
    windows = [int(trial.get("security_window_sends", 0)) for trial in trials]

    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(labels))
    axis.bar(positions - 0.2, requests, width=0.4, label="key requests")
    axis.bar(positions + 0.2, windows, width=0.4, label="security-window sends")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("relative validity")
    axis.set_ylabel("count")
    axis.set_title("Session-key requests and security window vs relative validity")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def attempt_plain_tcp_bypass(
    *,
    host: str = SST_ACTUATOR_HOST,
    port: int = SST_ACTUATOR_PORT,
    payload: bytes,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Connect without SST credentials and report what the attacker observed."""

    report: dict[str, Any] = {
        "connected": False,
        "bytes_sent": 0,
        "server_reply_bytes": 0,
        "client_error": None,
    }
    client: socket.socket | None = None
    try:
        client = socket.create_connection((host, port), timeout=timeout)
        report["connected"] = True
        client.sendall(payload)
        report["bytes_sent"] = len(payload)
        client.settimeout(timeout)
        try:
            reply = client.recv(4096)
            report["server_reply_bytes"] = len(reply)
        except (TimeoutError, OSError) as exc:
            report["client_error"] = type(exc).__name__
    except OSError as exc:
        report["client_error"] = type(exc).__name__
    finally:
        if client is not None:
            client.close()
    return report


def run_bypass(
    root: Path, *, threshold: float, offline_diagnostic: bool = False
) -> ScenarioOutcome:
    """Unauthenticated bypass attempts must be rejected at the actuator boundary."""

    scenario = "bypass"
    run_id = utc_run_id(scenario)
    directory = run_directory(root, scenario)
    action = load_scenario_action(root, scenario)

    monitor_started = time.perf_counter()
    analysis = analyze_prefix(action, threshold)
    monitoring_ms = (time.perf_counter() - monitor_started) * 1000.0
    if not analysis.high_risk:
        raise ScenarioError("bypass scenario requires the high-risk fixture")

    # In-process boundary: a raw, unprotected chunk sent straight to the
    # trusted actuator object must hold without any simulator execution.
    _, _, actuator_protector, channel = _offline_stack()
    simulator = RecordingSimulator()
    actuator = SafeActuator(
        simulator, high_risk_protector=actuator_protector, low_risk_channel=channel
    )
    actuation = actuator.actuate(action.executed_prefix, action.fixture.current_state)
    boundary_ok = (
        actuation.held
        and actuation.path == "bypass"
        and simulator.execute_count == 0
        and simulator.hold_count == 1
    )

    network_report: dict[str, Any] | None = None
    network_ok = True
    if not offline_diagnostic:
        _, actuator_config = require_sst_runtime(root)
        receiver_transport = SSTActuatorTransport(
            actuator_config,
            host=SST_ACTUATOR_HOST,
            port=SST_ACTUATOR_PORT,
            accept_timeout=10.0,
        )
        with _ActuatorReceiver(receiver_transport) as receiver:
            attacker_payload = json.dumps(
                {
                    "schema": "safe-agent-actuation/protected-action/v1",
                    "run_id": run_id,
                    "action_chunk": action.executed_prefix.tolist(),
                }
            ).encode("utf-8")
            network_report = attempt_plain_tcp_bypass(payload=attacker_payload)
            received = receiver.result(timeout=20.0)
        network_ok = (
            bool(network_report["connected"])
            and not received.authenticated
            and not received.received
            and received.payload is None
        )
        network_report["server_reason"] = received.reason
        network_report["handshake_authenticated"] = received.authenticated
        network_report["protected_action_accepted"] = received.received

    total_ms = (time.perf_counter() - monitor_started) * 1000.0
    expected = boundary_ok and network_ok and simulator.execute_count == 0

    record = _base_record(
        run_id=run_id,
        scenario=scenario,
        action=action,
        analysis=analysis,
        context=None,
        authorization_requested=False,
        authorization=AuthorizationDecision(
            AuthorizationOutcome.NOT_REQUESTED,
            "bypass attempt deliberately skipped authorization",
        ),
        latencies=LatencyRecord(
            monitoring_ms=monitoring_ms,
            total_monitor_to_actuator_ms=total_ms,
        ),
        actuation=actuation,
        simulator_outcome="held_in_recording_harness",
        metadata={
            "sst_required": not offline_diagnostic,
            "authorization_mode": (
                OFFLINE_AUTHORIZATION_MODE if offline_diagnostic else LIVE_AUTHORIZATION_MODE
            ),
            "in_process_rejection_reason": actuation.reason,
            "network_bypass": network_report,
        },
    )
    record_path = _write(directory, record, None)
    if not expected:
        raise ScenarioError(
            "bypass was not fully rejected: "
            f"boundary_ok={boundary_ok} network_ok={network_ok}"
        )
    return ScenarioOutcome(
        scenario=scenario,
        run_id=run_id,
        record_path=record_path,
        expected=True,
        summary={
            "in_process_rejection": actuation.reason,
            "actuator_decision": actuation.decision.value,
            "simulator_executions": simulator.execute_count,
            "network_bypass": network_report,
        },
    )

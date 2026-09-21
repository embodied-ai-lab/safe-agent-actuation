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
import os
import queue
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
GRAD_VALIDITY_PERIODS = (
    ("5*sec", 5_000),
    ("20*sec", 20_000),
    ("40*sec", 40_000),
)
GRAD_VALIDITY_STEPS = (
    "authorize_allowed",
    "reuse_valid_key",
    "security_window_denied_context",
    "denied_after_expiry",
    "reissue_allowed",
)
GRAD_VALIDITY_COUNTERS: dict[str, int | bool] = {
    "key_request_attempts": 3,
    "successful_key_requests": 2,
    "key_reuses": 2,
    "transfer_attempts": 4,
    "successful_transfers": 4,
    "allowed_while_valid": 3,
    "security_window_sends": 1,
    "denied_after_expiry": 1,
    "reissued_after_expiry": True,
}
GRAD_VALIDITY_TOTAL_FIELDS = tuple(GRAD_VALIDITY_COUNTERS)


class SSTMonitorSender(Protocol):
    """Structural send boundary used by the monitor and its test doubles."""

    def send(
        self,
        action_chunk: np.ndarray,
        *,
        run_id: str,
        timestep: int,
        context: Mapping[str, object],
    ) -> SSTSendResult: ...


class ScenarioError(RuntimeError):
    """Infrastructure or unexpected-outcome failure; the command must exit nonzero."""


def utc_run_id(scenario: str) -> str:
    stamp = _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario}-{stamp}"


def run_directory(root: Path, scenario: str) -> Path:
    directory = root / "results" / scenario
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _output_directory(root: Path, scenario: str, output_dir: Path | None) -> Path:
    """Resolve one result directory without conflating live and diagnostic output."""

    if output_dir is None:
        return run_directory(root, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    if metadata_horizon is None:
        horizon = int(fixture.action.shape[0])
    elif isinstance(metadata_horizon, bool) or not isinstance(metadata_horizon, int):
        raise ScenarioError(f"fixture {fixture_file} has a non-integer executed horizon")
    else:
        horizon = metadata_horizon
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


def run_low_risk(
    root: Path, *, threshold: float, output_dir: Path | None = None
) -> ScenarioOutcome:
    """Trusted low-risk direct path; SST must not be involved."""

    scenario = "low-risk"
    run_id = utc_run_id(scenario)
    directory = _output_directory(root, scenario, output_dir)
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


def _auth_process_is_running(pid: int) -> bool:
    """Check the owned Auth process without opening a protocol connection.

    A bare TCP connect is not a harmless liveness probe for the pinned Auth
    server: every accepted socket occupies a handler until its protocol read
    times out.  Repeated probes can therefore delay genuine session-key
    requests.  The supported launcher records Auth's PID, so validate that
    process and its command line instead.
    """

    try:
        os.kill(pid, 0)
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    return b"auth-server-jar-with-dependencies.jar" in command


def require_auth_running(root: Path) -> None:
    """Require the Auth process started by ``sst/scripts/auth.sh``."""

    pid_path = root / SST_RUNTIME_DIR / "auth.pid"
    try:
        raw_pid = pid_path.read_text(encoding="utf-8").strip()
        if not raw_pid.isdecimal():
            raise ValueError("Auth PID is not a positive integer")
        pid = int(raw_pid)
        if pid <= 1 or not _auth_process_is_running(pid):
            raise ValueError("recorded Auth process is not running")
    except (OSError, ValueError) as exc:
        raise ScenarioError(
            "Auth is not running under the course PID file. Run: make auth-start"
        ) from exc


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
    transport: SSTMonitorSender,
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
    #   the nonempty horizon does not exceed the configured executed prefix,
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
    elif send.authorization_status == "auth_rejected_request" and denied_expected:
        outcome = AuthorizationOutcome.DENIED
    else:
        # Protocol alert 1 is only a controlled policy denial when the fixed,
        # locally validated context is also known to be denied.  Every other
        # failed request is an infrastructure/protocol error, including an
        # alert returned for a context the registered policy should allow.
        outcome = AuthorizationOutcome.ERROR
    return AuthorizationDecision(
        outcome=outcome,
        reason=send.reason,
        key_id=send.key_id if send.authorized else None,
        session_key_reused=send.key_reused,
        latency_ms=send.authorization_latency_ms,
    )


@dataclass(frozen=True, slots=True)
class MonitoredTransactionResult:
    """One monitor-receive through confirmed actuator-decision transaction.

    ``total_monitor_to_actuator_ms`` is one direct monotonic measurement.  It
    deliberately is not reconstructed by adding component measurements.
    """

    analysis: MotionAnalysis
    authorization_requested: bool
    authorization: AuthorizationDecision
    send: SSTSendResult | None
    received: SSTReceiveResult | None
    actuation: ActuationResult
    monitoring_ms: float
    actuator_check_ms: float
    total_monitor_to_actuator_ms: float
    expected_authorization_allowed: bool | None
    success: bool
    simulator_execute_count: int
    simulator_hold_count: int
    failure_stage: str | None = None
    failure_reason: str | None = None


def run_monitored_transaction_live(
    *,
    action_chunk: np.ndarray,
    current_state: np.ndarray,
    limits: PhysicalLimits,
    threshold: float,
    context: AuthorizationContext,
    run_id: str,
    timestep: int,
    monitor_transport: SSTMonitorSender | None,
    actuator_config: Path | None,
    actuator_transport: SSTActuatorTransport | None = None,
) -> MonitoredTransactionResult:
    """Run one multi-threshold-evaluation transaction through real SST when required.

    Low-risk actions take the trusted direct path.  High-risk actions request
    or reuse a key and traverse SST.  A locally denied fixed context is a
    successful fail-closed trial only when Auth returns its rejection signal;
    connection, protocol, transfer, and receive faults remain errors.
    """

    # Listener construction and bind are experiment initialization, not part
    # of the monitor-to-actuator interval.  Prepare the endpoint before the
    # Monitor receives the action, then start its receive thread only if the
    # action is actually classified high risk.  A denied fixed context has no
    # listener so unexpected authorization cannot masquerade as a valid send.
    expected_allowed = ContextPolicy.course_default().evaluate(context).allowed
    prepared_receiver: SSTActuatorTransport | None = None
    if expected_allowed and monitor_transport is not None:
        prepared_receiver = actuator_transport
        if prepared_receiver is None and actuator_config is not None:
            prepared_receiver = SSTActuatorTransport(
                actuator_config,
                host=SST_ACTUATOR_HOST,
                port=SST_ACTUATOR_PORT,
                accept_timeout=10.0,
            )
    if prepared_receiver is not None:
        prepared_receiver.listen()

    transaction_started_ns = time.perf_counter_ns()
    analysis = analyze_motion(
        action_chunk=np.asarray(action_chunk, dtype=np.float64),
        current_state=np.asarray(current_state, dtype=np.float64),
        lower_limits=limits.lower,
        upper_limits=limits.upper,
        threshold=float(threshold),
        dimension_names=limits.dimension_names,
    )
    monitoring_ms = (time.perf_counter_ns() - transaction_started_ns) / 1_000_000.0
    simulator = RecordingSimulator()

    if not analysis.valid:
        if prepared_receiver is not None:
            prepared_receiver.close()
        actuation = enforce_received_action(
            None,
            expected_run_id=run_id,
            current_state=np.asarray(current_state, dtype=np.float64),
            limits=limits,
            simulator=simulator,
        )
        total_ms = (time.perf_counter_ns() - transaction_started_ns) / 1_000_000.0
        authorization = AuthorizationDecision(
            AuthorizationOutcome.NOT_REQUESTED,
            "invalid motion was held before authorization",
        )
        return MonitoredTransactionResult(
            analysis=analysis,
            authorization_requested=False,
            authorization=authorization,
            send=None,
            received=None,
            actuation=actuation,
            monitoring_ms=monitoring_ms,
            actuator_check_ms=actuation.total_latency_ms,
            total_monitor_to_actuator_ms=total_ms,
            expected_authorization_allowed=None,
            success=True,
            simulator_execute_count=simulator.execute_count,
            simulator_hold_count=simulator.hold_count,
        )

    if analysis.is_low_risk:
        if prepared_receiver is not None:
            prepared_receiver.close()
        _, _, actuator_protector, channel = _offline_stack()
        actuator = SafeActuator(
            simulator,
            high_risk_protector=actuator_protector,
            low_risk_channel=channel,
        )
        envelope = channel.issue(np.asarray(action_chunk, dtype=np.float64), analysis)
        actuation = actuator.actuate(envelope, np.asarray(current_state, dtype=np.float64))
        total_ms = (time.perf_counter_ns() - transaction_started_ns) / 1_000_000.0
        success = (
            actuation.executed
            and simulator.execute_count == 1
            and simulator.hold_count == 0
        )
        authorization = AuthorizationDecision(
            AuthorizationOutcome.NOT_REQUESTED,
            "low-risk classification used the trusted direct path; SST was not involved",
        )
        return MonitoredTransactionResult(
            analysis=analysis,
            authorization_requested=False,
            authorization=authorization,
            send=None,
            received=None,
            actuation=actuation,
            monitoring_ms=monitoring_ms,
            actuator_check_ms=actuation.total_latency_ms,
            total_monitor_to_actuator_ms=total_ms,
            expected_authorization_allowed=None,
            success=success,
            simulator_execute_count=simulator.execute_count,
            simulator_hold_count=simulator.hold_count,
            failure_stage=None if success else "actuator",
            failure_reason=None if success else actuation.reason,
        )

    send: SSTSendResult | None = None
    received: SSTReceiveResult | None = None
    try:
        if monitor_transport is None:
            raise ScenarioError("high-risk transaction has no SST monitor transport")
        if expected_allowed:
            if prepared_receiver is None:
                raise ScenarioError("high-risk transaction has no SST actuator configuration")
            with _ActuatorReceiver(prepared_receiver) as receiver:
                send = send_high_risk_action_live(
                    monitor_transport,
                    np.asarray(action_chunk, dtype=np.float64),
                    run_id=run_id,
                    timestep=timestep,
                    context=context,
                )
                if send.sent:
                    received = receiver.result(timeout=20.0)
                else:
                    received = SSTReceiveResult(
                        authenticated=False,
                        received=False,
                        receive_decryption_latency_ms=0.0,
                        reason="not received because the SST send did not complete",
                    )
        else:
            # A controlled denial must happen before the transfer boundary.
            # No listener is opened, making unexpected progression observable.
            send = send_high_risk_action_live(
                monitor_transport,
                np.asarray(action_chunk, dtype=np.float64),
                run_id=run_id,
                timestep=timestep,
                context=context,
            )
            received = SSTReceiveResult(
                authenticated=False,
                received=False,
                receive_decryption_latency_ms=0.0,
                reason="not started: fixed context is expected to be denied by Auth",
            )
    except Exception as exc:  # noqa: BLE001 - preserve a failed measured trial.
        if prepared_receiver is not None:
            prepared_receiver.close()
        actuation = enforce_received_action(
            None,
            expected_run_id=run_id,
            current_state=np.asarray(current_state, dtype=np.float64),
            limits=limits,
            simulator=simulator,
        )
        total_ms = (time.perf_counter_ns() - transaction_started_ns) / 1_000_000.0
        authorization = AuthorizationDecision(
            AuthorizationOutcome.ERROR,
            f"SST transaction infrastructure failure: {type(exc).__name__}: {exc}",
        )
        return MonitoredTransactionResult(
            analysis=analysis,
            authorization_requested=True,
            authorization=authorization,
            send=send,
            received=received,
            actuation=actuation,
            monitoring_ms=monitoring_ms,
            actuator_check_ms=actuation.total_latency_ms,
            total_monitor_to_actuator_ms=total_ms,
            expected_authorization_allowed=expected_allowed,
            success=False,
            simulator_execute_count=simulator.execute_count,
            simulator_hold_count=simulator.hold_count,
            failure_stage="infrastructure",
            failure_reason=authorization.reason,
        )

    assert send is not None
    assert received is not None
    authorization = _authorization_from_send(
        send, denied_expected=not expected_allowed
    )
    actuation = enforce_received_action(
        received.payload if received.authenticated and received.received else None,
        expected_run_id=run_id,
        current_state=np.asarray(current_state, dtype=np.float64),
        limits=limits,
        simulator=simulator,
    )
    total_ms = (time.perf_counter_ns() - transaction_started_ns) / 1_000_000.0

    if expected_allowed:
        success = (
            authorization.outcome is AuthorizationOutcome.ALLOWED
            and send.sent
            and received.authenticated
            and received.received
            and actuation.executed
            and simulator.execute_count == 1
            and simulator.hold_count == 0
        )
    else:
        success = (
            authorization.outcome is AuthorizationOutcome.DENIED
            and not send.sent
            and not received.received
            and actuation.held
            and simulator.execute_count == 0
            and simulator.hold_count == 1
        )

    failure_stage: str | None = None
    failure_reason: str | None = None
    if not success:
        if authorization.outcome is AuthorizationOutcome.ERROR:
            failure_stage = "authorization"
            failure_reason = authorization.reason
        elif send.transfer_attempted and not send.sent:
            failure_stage = "protected_transfer"
            failure_reason = send.reason
        elif send.sent and not (received.authenticated and received.received):
            failure_stage = "protected_receive"
            failure_reason = received.reason
        else:
            failure_stage = "actuator"
            failure_reason = actuation.reason

    return MonitoredTransactionResult(
        analysis=analysis,
        authorization_requested=True,
        authorization=authorization,
        send=send,
        received=received,
        actuation=actuation,
        monitoring_ms=monitoring_ms,
        actuator_check_ms=actuation.total_latency_ms,
        total_monitor_to_actuator_ms=total_ms,
        expected_authorization_allowed=expected_allowed,
        success=success,
        simulator_execute_count=simulator.execute_count,
        simulator_hold_count=simulator.hold_count,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
    )


def run_high_risk_live(
    root: Path,
    *,
    scenario: str,
    threshold: float,
    context_name: str,
    expect_allowed: bool,
    output_dir: Path | None = None,
) -> ScenarioOutcome:
    """Graded high-risk path through the real pinned SST deployment."""

    run_id = utc_run_id(scenario)
    directory = _output_directory(root, scenario, output_dir)
    action = load_scenario_action(root, scenario)
    context = load_context(root, context_name)
    monitor_config, actuator_config = require_sst_runtime(root)
    require_auth_running(root)

    transport = SSTMonitorTransport(
        monitor_config, host=SST_ACTUATOR_HOST, port=SST_ACTUATOR_PORT
    )
    try:
        transaction = run_monitored_transaction_live(
            action_chunk=action.executed_prefix,
            current_state=np.asarray(action.fixture.current_state, dtype=np.float64),
            limits=action.limits,
            threshold=threshold,
            context=context,
            run_id=run_id,
            timestep=0,
            monitor_transport=transport,
            actuator_config=actuator_config,
        )
    except Exception as exc:  # Endpoint initialization failed before receipt.
        raise ScenarioError(
            f"SST transaction initialization failed: {type(exc).__name__}: {exc}"
        ) from exc

    analysis = transaction.analysis
    if not analysis.high_risk:
        raise ScenarioError(
            f"{scenario} requires an at-or-above-threshold fixture; got "
            f"score={analysis.motion_score} threshold={threshold}"
        )
    if transaction.expected_authorization_allowed is not expect_allowed:
        raise ScenarioError(
            f"{scenario} expected-policy configuration disagrees with context {context_name!r}"
        )

    send = transaction.send
    received = transaction.received
    authorization = transaction.authorization
    actuation = transaction.actuation
    failure_reason = transaction.failure_reason
    expected = transaction.success

    record = _base_record(
        run_id=run_id,
        scenario=scenario,
        action=action,
        analysis=analysis,
        context=context,
        authorization_requested=True,
        authorization=authorization,
        latencies=LatencyRecord(
            monitoring_ms=transaction.monitoring_ms,
            authorization_ms=(send.authorization_latency_ms if send is not None else None),
            encryption_ms=(send.encryption_send_latency_ms if send is not None else None),
            decryption_ms=(
                received.receive_decryption_latency_ms
                if received is not None and received.received
                else None
            ),
            total_monitor_to_actuator_ms=transaction.total_monitor_to_actuator_ms,
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
            "handshake_latency_ms": send.handshake_latency_ms if send is not None else None,
            "send_latency_ms": (
                send.encryption_send_latency_ms if send is not None else None
            ),
            "sst_send_reason": send.reason if send is not None else None,
            "sst_receive_reason": received.reason if received is not None else None,
            "session_key_reused": send.key_reused if send is not None else False,
            "key_request_attempts": send.key_request_attempts if send is not None else 0,
            "successful_key_requests": (
                send.successful_key_requests if send is not None else 0
            ),
            "authorization_status": (
                send.authorization_status if send is not None else "infrastructure_error"
            ),
            "transfer_attempted": send.transfer_attempted if send is not None else False,
            "transfer_succeeded": send.sent if send is not None else False,
            "failure_stage": transaction.failure_stage,
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
            "session_key_reused": send.key_reused if send is not None else False,
            "actuator_decision": actuation.decision.value,
            "simulator_executions": transaction.simulator_execute_count,
            "authorization_latency_ms": (
                send.authorization_latency_ms if send is not None else 0.0
            ),
            "handshake_latency_ms": send.handshake_latency_ms if send is not None else 0.0,
        },
    )


def run_high_risk_offline_diagnostic(
    root: Path,
    *,
    scenario: str,
    threshold: float,
    context_name: str,
    expect_allowed: bool,
    output_dir: Path | None = None,
) -> ScenarioOutcome:
    """Explicitly labeled offline diagnostic; never a live SST claim."""

    run_id = utc_run_id(scenario)
    directory = _output_directory(root, f"{scenario}-offline", output_dir)
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
    authorization_status: str
    key_request_attempts: int
    successful_key_requests: int
    transfer_attempted: bool
    authorization_latency_ms: float
    handshake_latency_ms: float
    send_latency_ms: float
    total_transport_latency_ms: float
    reason: str


def _round_value(round_: _LiveSendRound | Mapping[str, Any], field: str) -> Any:
    if isinstance(round_, Mapping):
        return round_.get(field)
    return getattr(round_, field)


def _validity_round_counters(
    rounds: Sequence[_LiveSendRound | Mapping[str, Any]],
) -> dict[str, int | bool]:
    """Recompute all trial counters from the five recorded transport rounds."""

    def completed(round_: _LiveSendRound | Mapping[str, Any]) -> bool:
        return (
            _round_value(round_, "sent") is True
            and _round_value(round_, "received") is True
        )

    by_step = {str(_round_value(round_, "step")): round_ for round_ in rounds}
    security_window = by_step.get("security_window_denied_context")
    post_expiry = by_step.get("denied_after_expiry")
    reissue = by_step.get("reissue_allowed")
    allowed_steps = {
        "authorize_allowed",
        "reuse_valid_key",
        "security_window_denied_context",
    }
    return {
        "key_request_attempts": sum(
            int(_round_value(round_, "key_request_attempts")) for round_ in rounds
        ),
        "successful_key_requests": sum(
            int(_round_value(round_, "successful_key_requests")) for round_ in rounds
        ),
        "key_reuses": sum(
            _round_value(round_, "key_reused") is True for round_ in rounds
        ),
        "transfer_attempts": sum(
            _round_value(round_, "transfer_attempted") is True for round_ in rounds
        ),
        "successful_transfers": sum(completed(round_) for round_ in rounds),
        "allowed_while_valid": sum(
            str(_round_value(round_, "step")) in allowed_steps and completed(round_)
            for round_ in rounds
        ),
        "security_window_sends": int(
            security_window is not None and completed(security_window)
        ),
        "denied_after_expiry": int(
            post_expiry is not None
            and _round_value(post_expiry, "authorized") is False
            and _round_value(post_expiry, "authorization_status")
            == "auth_rejected_request"
            and _round_value(post_expiry, "sent") is False
            and _round_value(post_expiry, "received") is False
            and _round_value(post_expiry, "transfer_attempted") is False
        ),
        "reissued_after_expiry": bool(
            reissue is not None
            and completed(reissue)
            and _round_value(reissue, "key_reused") is False
        ),
    }


def _validate_live_validity_trial(trial: Mapping[str, Any]) -> dict[str, int | bool]:
    """Validate period identity and round-derived counters before publication."""

    expected_ms = dict(GRAD_VALIDITY_PERIODS).get(str(trial.get("label")))
    if expected_ms is None or trial.get("relative_validity_ms") != expected_ms:
        raise ScenarioError(
            "live key-validity trial does not match a documented period: "
            f"label={trial.get('label')!r} relative_validity_ms="
            f"{trial.get('relative_validity_ms')!r}"
        )
    rounds = trial.get("rounds")
    if not isinstance(rounds, list) or not all(
        isinstance(round_, Mapping) for round_ in rounds
    ):
        raise ScenarioError("live key-validity trial rounds are missing or malformed")
    if [round_.get("step") for round_ in rounds] != list(GRAD_VALIDITY_STEPS):
        raise ScenarioError("live key-validity trial has an invalid controlled sequence")
    expected_rounds: dict[str, dict[str, Any]] = {
        "authorize_allowed": {
            "authorized": True,
            "authorization_status": "allowed",
            "sent": True,
            "received": True,
            "key_reused": False,
            "key_request_attempts": 1,
            "successful_key_requests": 1,
            "transfer_attempted": True,
        },
        "reuse_valid_key": {
            "authorized": True,
            "authorization_status": "allowed",
            "sent": True,
            "received": True,
            "key_reused": True,
            "key_request_attempts": 0,
            "successful_key_requests": 0,
            "transfer_attempted": True,
        },
        "security_window_denied_context": {
            "authorized": True,
            "authorization_status": "allowed",
            "sent": True,
            "received": True,
            "key_reused": True,
            "key_request_attempts": 0,
            "successful_key_requests": 0,
            "transfer_attempted": True,
        },
        "denied_after_expiry": {
            "authorized": False,
            "authorization_status": "auth_rejected_request",
            "sent": False,
            "received": False,
            "key_reused": False,
            "key_request_attempts": 1,
            "successful_key_requests": 0,
            "transfer_attempted": False,
        },
        "reissue_allowed": {
            "authorized": True,
            "authorization_status": "allowed",
            "sent": True,
            "received": True,
            "key_reused": False,
            "key_request_attempts": 1,
            "successful_key_requests": 1,
            "transfer_attempted": True,
        },
    }
    for round_ in rounds:
        step = str(round_["step"])
        mismatches = [
            field
            for field, expected in expected_rounds[step].items()
            if round_.get(field) != expected
        ]
        if mismatches:
            raise ScenarioError(
                f"live key-validity trial step {step} violates the controlled "
                f"sequence fields: {mismatches}"
            )
    counters = _validity_round_counters(rounds)
    for field, expected in counters.items():
        if trial.get(field) != expected:
            raise ScenarioError(
                f"live key-validity trial field {field} does not recompute from rounds"
            )
    if trial.get("key_requests") != counters["successful_key_requests"]:
        raise ScenarioError(
            "live key-validity trial key_requests alias is inconsistent with "
            "successful_key_requests"
        )
    if counters != GRAD_VALIDITY_COUNTERS:
        raise ScenarioError(
            f"live key-validity trial counters violate the controlled baseline: {counters}"
        )
    if trial.get("expected") is not True:
        raise ScenarioError("live key-validity trial is not marked expected")
    return counters


def _live_send_round(
    transport: SSTMonitorSender,
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
        authorization_status=send.authorization_status,
        key_request_attempts=send.key_request_attempts,
        successful_key_requests=send.successful_key_requests,
        transfer_attempted=send.transfer_attempted,
        authorization_latency_ms=send.authorization_latency_ms,
        handshake_latency_ms=send.handshake_latency_ms,
        send_latency_ms=send.encryption_send_latency_ms,
        total_transport_latency_ms=send.total_latency_ms,
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

    expected_relative_validity_ms = dict(GRAD_VALIDITY_PERIODS).get(label)
    if expected_relative_validity_ms is None:
        raise ScenarioError(
            f"unsupported graded validity label {label!r}; expected one of "
            f"{[item[0] for item in GRAD_VALIDITY_PERIODS]}"
        )

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

    # Keep the actuator entity context for the complete controlled sequence.
    # Repeated monitor connections may reuse the same key; rebuilding the
    # actuator context would discard its cached copy and make it ask Auth to
    # register an owner that is already present.  The denied post-expiry round
    # receives this object but never opens its listener.
    receiver_transport = SSTActuatorTransport(
        actuator_config,
        host=SST_ACTUATOR_HOST,
        port=SST_ACTUATOR_PORT,
        accept_timeout=10.0,
    )

    rounds: list[_LiveSendRound] = []
    rounds.append(
        _live_send_round(
            transport, receiver_transport, action.executed_prefix,
            step="authorize_allowed", run_id=run_id, timestep=0,
            context=allowed, expect_sent=True,
        )
    )
    validity = transport.cached_key_validity()
    first_use_ms = validity.get("first_use_ms")
    rel_validity_ms = validity.get("rel_validity_ms")
    abs_validity_ms = validity.get("abs_validity_ms")
    if rel_validity_ms != expected_relative_validity_ms:
        raise ScenarioError(
            "Auth returned a relative validity that does not match the generated "
            f"graded period: label={label!r}, expected_ms="
            f"{expected_relative_validity_ms}, actual_ms={rel_validity_ms!r}"
        )
    rounds.append(
        _live_send_round(
            transport, receiver_transport, action.executed_prefix,
            step="reuse_valid_key", run_id=run_id, timestep=1,
            context=allowed, expect_sent=True,
        )
    )

    # Record the actual wall-clock validity remaining when the policy context
    # changes.  The live experiment intentionally does not reauthorize each
    # action: an already-issued key remains usable until its SST validity ends.
    context_change_ms = int(time.time() * 1000)
    expiry_candidates: list[int] = []
    if first_use_ms is not None and rel_validity_ms is not None:
        expiry_candidates.append(int(first_use_ms) + int(rel_validity_ms))
    if abs_validity_ms is not None:
        expiry_candidates.append(int(abs_validity_ms))
    effective_expiry_ms = min(expiry_candidates) if expiry_candidates else None
    remaining_validity_ms = (
        max(0, effective_expiry_ms - context_change_ms)
        if effective_expiry_ms is not None
        else None
    )
    rounds.append(
        _live_send_round(
            transport, receiver_transport, action.executed_prefix,
            step="security_window_denied_context", run_id=run_id, timestep=2,
            context=denied, expect_sent=True,
        )
    )

    # Wait from the current wall clock to the effective SST deadline, rather
    # than sleeping for a second full relative-validity period after the first
    # three messages.  This remains a real-time live test; no manual clock is
    # involved.
    sleep_seconds = expiry_margin_seconds
    if effective_expiry_ms is not None:
        sleep_seconds += max(
            0.0,
            (effective_expiry_ms - int(time.time() * 1000)) / 1000.0,
        )
    time.sleep(sleep_seconds)

    rounds.append(
        _live_send_round(
            transport, receiver_transport, action.executed_prefix,
            step="denied_after_expiry", run_id=run_id, timestep=3,
            context=denied, expect_sent=False,
        )
    )
    rounds.append(
        _live_send_round(
            transport, receiver_transport, action.executed_prefix,
            step="reissue_allowed", run_id=run_id, timestep=4,
            context=allowed, expect_sent=True,
        )
    )

    round_rows = [
        {
            "step": item.step,
            "authorized": item.authorized,
            "sent": item.sent,
            "key_reused": item.key_reused,
            "received": item.received,
            "authorization_status": item.authorization_status,
            "key_request_attempts": item.key_request_attempts,
            "successful_key_requests": item.successful_key_requests,
            "transfer_attempted": item.transfer_attempted,
            "authorization_latency_ms": item.authorization_latency_ms,
            "handshake_latency_ms": item.handshake_latency_ms,
            "send_latency_ms": item.send_latency_ms,
            "total_transport_latency_ms": item.total_transport_latency_ms,
            "reason": item.reason,
        }
        for item in rounds
    ]
    counters = _validity_round_counters(round_rows)
    successful_transfers = [item for item in rounds if item.sent and item.received]
    trial = {
        "label": label,
        "run_id": run_id,
        "authorization_mode": LIVE_AUTHORIZATION_MODE,
        "relative_validity_ms": rel_validity_ms,
        "absolute_validity_ms": abs_validity_ms,
        "first_use_ms": first_use_ms,
        "context_change_ms": context_change_ms,
        "effective_expiry_ms": effective_expiry_ms,
        "remaining_validity_ms_at_context_change": remaining_validity_ms,
        **counters,
        # Backward-compatible alias, explicitly defined as successful requests.
        "key_requests": counters["successful_key_requests"],
        "average_authorization_latency_ms": (
            float(np.mean([item.authorization_latency_ms for item in rounds]))
        ),
        "average_handshake_latency_ms": (
            float(np.mean([item.handshake_latency_ms for item in rounds if item.sent]))
            if any(item.sent for item in rounds) else 0.0
        ),
        "average_send_latency_ms": (
            float(np.mean([item.send_latency_ms for item in successful_transfers]))
            if successful_transfers
            else 0.0
        ),
        "average_protected_transfer_latency_ms": (
            float(
                np.mean(
                    [
                        item.handshake_latency_ms + item.send_latency_ms
                        for item in successful_transfers
                    ]
                )
            )
            if successful_transfers
            else 0.0
        ),
        "rounds": round_rows,
    }

    expected = (
        counters == GRAD_VALIDITY_COUNTERS
        and counters["key_request_attempts"] == transport.key_request_attempts
        and counters["successful_key_requests"] == transport.key_requests
    )
    trial["expected"] = expected

    # Retain failed live trials for diagnosis; the explicit expected flag keeps
    # them out of any accidental "all passed" interpretation.
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "live_trials.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trial, sort_keys=True) + "\n")

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
    if not all(isinstance(trial, Mapping) for trial in trials):
        raise ScenarioError("live key-validity trial file contains a non-object row")
    observed_periods = [
        (str(trial.get("label")), trial.get("relative_validity_ms"))
        for trial in trials
    ]
    if observed_periods != list(GRAD_VALIDITY_PERIODS):
        raise ScenarioError(
            "graded live key-validity results must contain exactly the ordered "
            f"5/20/40-second periods; observed {observed_periods}"
        )
    recomputed = [_validate_live_validity_trial(trial) for trial in trials]

    import csv

    fieldnames = [
        "label",
        "relative_validity_ms",
        "absolute_validity_ms",
        "first_use_ms",
        "context_change_ms",
        "effective_expiry_ms",
        "remaining_validity_ms_at_context_change",
        "key_request_attempts",
        "successful_key_requests",
        "key_requests",
        "key_reuses",
        "transfer_attempts",
        "successful_transfers",
        "allowed_while_valid",
        "security_window_sends",
        "denied_after_expiry",
        "reissued_after_expiry",
        "expected",
        "average_authorization_latency_ms",
        "average_handshake_latency_ms",
        "average_send_latency_ms",
        "average_protected_transfer_latency_ms",
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
        "all_trials_expected": all(trial.get("expected") is True for trial in trials),
        "totals": {
            field: sum(int(counters[field]) for counters in recomputed)
            for field in GRAD_VALIDITY_TOTAL_FIELDS
        },
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
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    labels = [str(trial.get("label")) for trial in trials]
    attempts = [int(trial.get("key_request_attempts", 0)) for trial in trials]
    successful = [int(trial.get("successful_key_requests", 0)) for trial in trials]
    reuses = [int(trial.get("key_reuses", 0)) for trial in trials]
    windows = [
        float(trial.get("remaining_validity_ms_at_context_change") or 0.0) / 1000.0
        for trial in trials
    ]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(labels))
    width = 0.24
    axis.bar(positions - width, attempts, width=width, label="request attempts")
    axis.bar(positions, successful, width=width, label="successful requests")
    axis.bar(positions + width, reuses, width=width, label="key reuses")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("relative validity")
    axis.set_ylabel("count")
    window_axis = axis.twinx()
    window_axis.plot(
        positions,
        windows,
        color="black",
        marker="o",
        label="remaining validity at context change",
    )
    window_axis.set_ylabel("observed remaining validity (s)")
    axis.set_title("Session-key activity and context-change window")
    handles, legend_labels = axis.get_legend_handles_labels()
    window_handles, window_labels = window_axis.get_legend_handles_labels()
    axis.legend(handles + window_handles, legend_labels + window_labels, loc="best")
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
    root: Path,
    *,
    threshold: float,
    offline_diagnostic: bool = False,
    output_dir: Path | None = None,
) -> ScenarioOutcome:
    """Unauthenticated bypass attempts must be rejected at the actuator boundary."""

    scenario = "bypass"
    run_id = utc_run_id(scenario)
    result_name = f"{scenario}-offline" if offline_diagnostic else scenario
    directory = _output_directory(root, result_name, output_dir)
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

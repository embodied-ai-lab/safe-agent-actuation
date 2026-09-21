"""Multi-threshold evaluation and session-key validity analysis over fixed action sets.

Neither analysis performs new model inference. The controlled multi-threshold
evaluation reuses one committed fixture set for every threshold.  The key-validity diagnostic is a
deterministic offline model of SST reuse and expiration used for tests and the
labeled offline command; the graded extension drives real SST separately.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .context import AuthorizationContext, ContextPolicy
from .course_config import load_context, load_physical_limits, scenario_settings
from .course_fixtures import course_fixture_specs
from .fixtures import ActionFixture
from .motion import analyze_motion
from .protection import (
    AESGCMProtector,
    InMemorySessionKeyStore,
    OfflineContextAuthorizer,
    ProtectionError,
)
from .scenarios import (
    LIVE_AUTHORIZATION_MODE,
    MonitoredTransactionResult,
    ScenarioError,
    require_auth_running,
    require_sst_runtime,
    run_monitored_transaction_live,
)
from .sst_adapter import SSTActuatorTransport, SSTMonitorTransport
from .types import AuthorizationOutcome

# The multi-threshold evaluation uses the whole committed fixture set, not just the scenario
# fixtures, so the low-risk and high-risk rates change meaningfully with tau.
SWEEP_FIXTURE_NAMES = ("calm", "low_risk", "moderate", "boundary", "high_risk")


@dataclass(frozen=True, slots=True)
class ThresholdTrial:
    threshold: float
    fixture_count: int
    low_risk: int
    high_risk: int
    invalid: int
    authorization_active_rate: float
    allowed: int
    denied: int
    average_monitoring_ms: float
    average_authorization_ms: float
    average_monitor_to_actuator_ms: float
    observed_max_monitor_to_actuator_ms: float

    def to_row(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "fixture_count": self.fixture_count,
            "low_risk_count": self.low_risk,
            "high_risk_count": self.high_risk,
            "invalid_count": self.invalid,
            "authorization_active_rate": round(self.authorization_active_rate, 6),
            "allowed_count": self.allowed,
            "denied_count": self.denied,
            "average_monitoring_latency_ms": round(self.average_monitoring_ms, 6),
            "average_authorization_latency_ms": round(self.average_authorization_ms, 6),
            "average_monitor_to_actuator_latency_ms": round(
                self.average_monitor_to_actuator_ms, 6
            ),
            "observed_max_monitor_to_actuator_latency_ms": round(
                self.observed_max_monitor_to_actuator_ms, 6
            ),
        }


def _sweep_fixtures(
    root: Path, names: Sequence[str] = SWEEP_FIXTURE_NAMES
) -> list[ActionFixture]:
    limits = load_physical_limits(root)
    specs = {spec.name: spec for spec in course_fixture_specs(limits)}
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"threshold-sweep references unknown fixtures: {unknown}")
    return [specs[name].fixture for name in names]


def evaluate_threshold(
    fixtures: Sequence[ActionFixture],
    threshold: float,
    *,
    allowed_context: Any,
    denied_context: Any,
    context_names: Sequence[str],
) -> ThresholdTrial:
    """Classify fixed fixture/context pairs and simulate authorization.

    ``context_names`` is paired by input position before classification.  It
    therefore cannot change merely because a different number of earlier
    fixtures crosses a threshold.
    """

    if len(fixtures) != len(context_names):
        raise ValueError("threshold-sweep fixtures and contexts must have equal lengths")
    if any(name not in {"allowed", "denied"} for name in context_names):
        raise ValueError("threshold-sweep context names must be 'allowed' or 'denied'")

    monitor_store = InMemorySessionKeyStore()
    actuator_store = InMemorySessionKeyStore()
    authorizer = OfflineContextAuthorizer(
        ContextPolicy.course_default(), monitor_store, actuator_store
    )
    protector = AESGCMProtector(monitor_store)

    low = high = invalid = allowed = denied = 0
    monitoring: list[float] = []
    authorization: list[float] = []
    totals: list[float] = []
    for fixture, context_name in zip(fixtures, context_names, strict=True):
        started = time.perf_counter()
        analysis = analyze_motion(
            action_chunk=fixture.action_chunk,
            current_state=fixture.current_state,
            lower_limits=fixture.limits.lower,
            upper_limits=fixture.limits.upper,
            threshold=threshold,
            dimension_names=fixture.dimension_names,
        )
        monitor_ms = (time.perf_counter() - started) * 1000.0
        monitoring.append(monitor_ms)
        if not analysis.valid:
            invalid += 1
            totals.append(monitor_ms)
            continue
        if analysis.is_low_risk:
            low += 1
            totals.append(monitor_ms)
            continue
        high += 1
        # Classification controls whether authorization is active, never which
        # context belongs to this fixed input pair.
        context = allowed_context if context_name == "allowed" else denied_context
        decision = authorizer.authorize(context)
        authorization.append(decision.latency_ms)
        total_ms = monitor_ms + decision.latency_ms
        if decision.outcome is AuthorizationOutcome.ALLOWED:
            allowed += 1
            assert decision.key_id is not None
            try:
                protector.protect(fixture.action_chunk, decision.key_id)
            except ProtectionError:
                pass
        else:
            denied += 1
        totals.append(total_ms)

    count = len(fixtures)
    return ThresholdTrial(
        threshold=float(threshold),
        fixture_count=count,
        low_risk=low,
        high_risk=high,
        invalid=invalid,
        authorization_active_rate=(high / count) if count else 0.0,
        allowed=allowed,
        denied=denied,
        average_monitoring_ms=float(np.mean(monitoring)) if monitoring else 0.0,
        average_authorization_ms=float(np.mean(authorization)) if authorization else 0.0,
        average_monitor_to_actuator_ms=float(np.mean(totals)) if totals else 0.0,
        observed_max_monitor_to_actuator_ms=float(np.max(totals)) if totals else 0.0,
    )


def run_threshold_sweep_offline_diagnostic(
    root: Path, output_dir: Path
) -> dict[str, Any]:
    """Run the deterministic policy-model diagnostic (never a live SST claim)."""

    settings = scenario_settings(root, "threshold-sweep")
    thresholds = [float(value) for value in settings.get("thresholds", [])]
    if not thresholds:
        raise ValueError("threshold-sweep scenario declares no thresholds")
    fixed_cases = _live_sweep_cases(settings.get("cases"))
    fixtures = _sweep_fixtures(root, [case.fixture_name for case in fixed_cases])
    allowed_context = load_context(root, "allowed")
    denied_context = load_context(root, "denied")
    trials = [
        evaluate_threshold(
            fixtures,
            threshold,
            allowed_context=allowed_context,
            denied_context=denied_context,
            context_names=[case.context_name for case in fixed_cases],
        )
        for threshold in thresholds
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [trial.to_row() for trial in trials]
    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "analysis": "threshold_sweep",
        "authorization_mode": (
            "offline_diagnostic (OfflineContextAuthorizer, not live SST)"
        ),
        "action_source": "diagnostic_synthetic (committed fixtures, no model inference)",
        "fixture_count": len(fixtures),
        "fixtures": [case.fixture_name for case in fixed_cases],
        "fixed_cases": [
            {
                "id": case.case_id,
                "fixture": case.fixture_name,
                "context": case.context_name,
            }
            for case in fixed_cases
        ],
        "thresholds": thresholds,
        "high_risk_comparison": ">=",
        "trials": rows,
        "note": (
            "Motion classification, policy authorization, and physical safety are "
            "distinct. A high-risk classification only means the motion score met the "
            "threshold; authorization is a separate policy decision; neither proves "
            "the action is physically safe."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_path = output_dir / "threshold_sweep.png"
    _plot_threshold_sweep(trials, plot_path)
    return {
        "csv": str(csv_path),
        "summary": str(summary_path),
        "plot": str(plot_path),
        "trials": rows,
    }


def _plot_threshold_sweep(trials: Sequence[ThresholdTrial], path: Path) -> None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    thresholds = [trial.threshold for trial in trials]
    high = [trial.high_risk for trial in trials]
    low = [trial.low_risk for trial in trials]
    observed_max = [trial.observed_max_monitor_to_actuator_ms for trial in trials]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    top.plot(thresholds, high, marker="o", label="high risk")
    top.plot(thresholds, low, marker="s", label="low risk")
    top.set_ylabel("fixtures")
    top.set_title("Risk classification vs threshold (fixed fixture set)")
    top.legend()
    top.grid(True, alpha=0.3)

    bottom.plot(thresholds, observed_max, marker="^", color="tab:red")
    bottom.set_ylabel("observed max diagnostic latency (ms)")
    bottom.set_xlabel("threshold")
    bottom.set_title("Observed maximum diagnostic latency vs threshold")
    bottom.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Graded live-SST multi-threshold evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepCase:
    """A fixed fixture/context pair reused unchanged at every threshold."""

    case_id: str
    fixture_name: str
    context_name: str


_DEFAULT_LIVE_CASES = (
    SweepCase("calm-allowed", "calm", "allowed"),
    SweepCase("low-risk-denied", "low_risk", "denied"),
    SweepCase("moderate-allowed", "moderate", "allowed"),
    SweepCase("boundary-denied", "boundary", "denied"),
    SweepCase("high-risk-allowed", "high_risk", "allowed"),
    SweepCase("high-risk-denied", "high_risk", "denied"),
)

_LIVE_TRIAL_FIELDS = (
    "trial_id",
    "phase",
    "threshold",
    "case_id",
    "fixture",
    "context_name",
    "repetition",
    "motion_score",
    "risk_classification",
    "analysis_valid",
    "authorization_path_active",
    "expected_policy_outcome",
    "authorization_outcome",
    "authorization_status",
    "key_request_attempts",
    "successful_key_requests",
    "key_reused",
    "transfer_attempted",
    "transfer_succeeded",
    "authenticated_received",
    "actuator_decision",
    "simulator_execute_count",
    "simulator_hold_count",
    "trial_status",
    "controlled_policy_denial",
    "failure_stage",
    "failure_reason",
    "auth_error_type",
    "auth_alert_code",
    "transfer_error_type",
    "receive_error_type",
    "monitoring_ms",
    "authorization_ms",
    "handshake_ms",
    "protected_send_ms",
    "receive_decryption_ms",
    "actuator_check_ms",
    "monitor_to_actuator_ms",
    "elapsed_until_failure_ms",
)


def _live_sweep_cases(raw: object) -> tuple[SweepCase, ...]:
    if raw is None:
        return _DEFAULT_LIVE_CASES
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("threshold-sweep cases must be a sequence")
    cases: list[SweepCase] = []
    for index, item in enumerate(raw):
        if isinstance(item, SweepCase):
            case = item
        elif isinstance(item, Mapping):
            try:
                case = SweepCase(
                    case_id=str(item["id"]),
                    fixture_name=str(item["fixture"]),
                    context_name=str(item["context"]),
                )
            except KeyError as exc:
                raise ValueError(f"threshold-sweep case {index} is missing {exc.args[0]}") from exc
        else:
            raise ValueError(f"threshold-sweep case {index} must be an object")
        if not case.case_id or not case.fixture_name:
            raise ValueError("threshold-sweep case IDs and fixture names must be non-empty")
        if case.context_name not in {"allowed", "denied"}:
            raise ValueError(
                f"threshold-sweep case {case.case_id!r} has unknown context "
                f"{case.context_name!r}"
            )
        cases.append(case)
    if not cases:
        raise ValueError("threshold-sweep declares no fixed cases")
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("threshold-sweep case IDs must be unique")
    return tuple(cases)


def _executed_fixture(fixture: ActionFixture) -> np.ndarray:
    configured = fixture.metadata.get("executed_horizon", fixture.action.shape[0])
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise ValueError("multi-threshold evaluation fixture has an invalid executed_horizon")
    horizon = configured
    if not 1 <= horizon <= fixture.action.shape[0]:
        raise ValueError(
            "multi-threshold evaluation fixture has an out-of-range executed_horizon"
        )
    return np.array(fixture.action[:horizon], dtype=np.float64, copy=True)


def _row_from_transaction(
    result: MonitoredTransactionResult,
    *,
    trial_id: str,
    phase: str,
    threshold: float,
    case: SweepCase,
    repetition: int,
    expected_policy_outcome: str,
) -> dict[str, Any]:
    send = result.send
    received = result.received
    authorization_outcome = result.authorization.outcome.value
    controlled_denial = (
        result.success
        and authorization_outcome == AuthorizationOutcome.DENIED.value
        and expected_policy_outcome == AuthorizationOutcome.DENIED.value
    )
    return {
        "trial_id": trial_id,
        "phase": phase,
        "threshold": float(threshold),
        "case_id": case.case_id,
        "fixture": case.fixture_name,
        "context_name": case.context_name,
        "repetition": repetition,
        "motion_score": result.analysis.motion_score,
        "risk_classification": result.analysis.risk_classification,
        "analysis_valid": result.analysis.valid,
        "authorization_path_active": result.authorization_requested,
        "expected_policy_outcome": expected_policy_outcome,
        "authorization_outcome": authorization_outcome,
        "authorization_status": (
            send.authorization_status
            if send is not None
            else (
                "infrastructure_error"
                if result.authorization_requested
                else "not_requested"
            )
        ),
        "key_request_attempts": send.key_request_attempts if send is not None else 0,
        "successful_key_requests": (
            send.successful_key_requests if send is not None else 0
        ),
        "key_reused": send.key_reused if send is not None else False,
        "transfer_attempted": send.transfer_attempted if send is not None else False,
        "transfer_succeeded": send.sent if send is not None else False,
        "authenticated_received": (
            bool(received.authenticated and received.received)
            if received is not None
            else False
        ),
        "actuator_decision": result.actuation.decision.value,
        "simulator_execute_count": result.simulator_execute_count,
        "simulator_hold_count": result.simulator_hold_count,
        "trial_status": "success" if result.success else "failure",
        "controlled_policy_denial": controlled_denial,
        "failure_stage": result.failure_stage,
        "failure_reason": result.failure_reason,
        "auth_error_type": send.auth_error_type if send is not None else None,
        "auth_alert_code": send.auth_alert_code if send is not None else None,
        "transfer_error_type": send.transfer_error_type if send is not None else None,
        "receive_error_type": (
            received.receive_error_type if received is not None else None
        ),
        "monitoring_ms": result.monitoring_ms,
        "authorization_ms": result.authorization.latency_ms,
        "handshake_ms": send.handshake_latency_ms if send is not None else 0.0,
        "protected_send_ms": (
            send.encryption_send_latency_ms if send is not None else 0.0
        ),
        "receive_decryption_ms": (
            received.receive_decryption_latency_ms if received is not None else 0.0
        ),
        "actuator_check_ms": result.actuator_check_ms,
        "monitor_to_actuator_ms": result.total_monitor_to_actuator_ms,
        "elapsed_until_failure_ms": None,
    }


def _row_from_runner_exception(
    exc: BaseException,
    *,
    elapsed_ms: float,
    fixture: ActionFixture,
    action_chunk: np.ndarray,
    threshold: float,
    case: SweepCase,
    trial_id: str,
    phase: str,
    repetition: int,
    expected_policy_outcome: str,
) -> dict[str, Any]:
    analysis = analyze_motion(
        action_chunk=action_chunk,
        current_state=fixture.current_state,
        lower_limits=fixture.limits.lower,
        upper_limits=fixture.limits.upper,
        threshold=threshold,
        dimension_names=fixture.dimension_names,
    )
    return {
        "trial_id": trial_id,
        "phase": phase,
        "threshold": float(threshold),
        "case_id": case.case_id,
        "fixture": case.fixture_name,
        "context_name": case.context_name,
        "repetition": repetition,
        "motion_score": analysis.motion_score,
        "risk_classification": analysis.risk_classification,
        "analysis_valid": analysis.valid,
        "authorization_path_active": analysis.high_risk,
        "expected_policy_outcome": expected_policy_outcome,
        "authorization_outcome": AuthorizationOutcome.ERROR.value,
        "authorization_status": "runner_exception",
        "key_request_attempts": 0,
        "successful_key_requests": 0,
        "key_reused": False,
        "transfer_attempted": False,
        "transfer_succeeded": False,
        "authenticated_received": False,
        "actuator_decision": "not_confirmed",
        "simulator_execute_count": 0,
        "simulator_hold_count": 0,
        "trial_status": "failure",
        "controlled_policy_denial": False,
        "failure_stage": "runner_exception",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "auth_error_type": type(exc).__name__,
        "auth_alert_code": None,
        "transfer_error_type": None,
        "receive_error_type": None,
        "monitoring_ms": None,
        "authorization_ms": None,
        "handshake_ms": None,
        "protected_send_ms": None,
        "receive_decryption_ms": None,
        "actuator_check_ms": None,
        # There was no confirmed actuator decision, so this is deliberately
        # null rather than a misleading end-to-end latency.
        "monitor_to_actuator_ms": None,
        "elapsed_until_failure_ms": elapsed_ms,
    }


def _aggregate_live_trials(
    rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [row for row in rows if float(row["threshold"]) == float(threshold)]
        valid = [row for row in selected if bool(row["analysis_valid"])]
        high = [row for row in valid if row["risk_classification"] == "high"]
        low = [row for row in valid if row["risk_classification"] == "low"]
        invalid = [row for row in selected if row["risk_classification"] == "invalid"]
        active = [row for row in selected if bool(row["authorization_path_active"])]
        transfers = [row for row in selected if bool(row["transfer_attempted"])]
        latency_rows = [
            float(row["monitor_to_actuator_ms"])
            for row in selected
            if isinstance(row["monitor_to_actuator_ms"], (int, float))
        ]
        successes = sum(row["trial_status"] == "success" for row in selected)
        aggregates.append(
            {
                "threshold": float(threshold),
                "measured_trial_count": len(selected),
                "classified_trial_count": len(valid),
                "low_risk_count": len(low),
                "high_risk_count": len(high),
                "invalid_count": len(invalid),
                "high_risk_rate": len(high) / len(valid) if valid else 0.0,
                "authorization_activity_count": len(active),
                "authorization_activity_rate": len(active) / len(valid) if valid else 0.0,
                "key_request_attempts": sum(
                    int(row["key_request_attempts"]) for row in selected
                ),
                "successful_key_requests": sum(
                    int(row["successful_key_requests"]) for row in selected
                ),
                "key_reuses": sum(bool(row["key_reused"]) for row in selected),
                "authorization_allowed_count": sum(
                    row["authorization_outcome"] == "allowed" for row in selected
                ),
                "authorization_denied_count": sum(
                    row["authorization_outcome"] == "denied" for row in selected
                ),
                "authorization_error_count": sum(
                    row["authorization_outcome"] == "error" for row in selected
                ),
                "authorization_not_requested_count": sum(
                    row["authorization_outcome"] == "not_requested" for row in selected
                ),
                "controlled_policy_denial_count": sum(
                    bool(row["controlled_policy_denial"]) for row in selected
                ),
                "transfer_attempts": len(transfers),
                "transfer_successes": sum(
                    bool(row["transfer_succeeded"]) for row in transfers
                ),
                "transfer_failures": sum(
                    not bool(row["transfer_succeeded"]) for row in transfers
                ),
                "actuator_execute_count": sum(
                    row["actuator_decision"] == "execute" for row in selected
                ),
                "actuator_hold_count": sum(
                    row["actuator_decision"] == "hold" for row in selected
                ),
                "actuator_not_confirmed_count": sum(
                    row["actuator_decision"] == "not_confirmed" for row in selected
                ),
                "successful_trial_count": successes,
                "failed_trial_count": len(selected) - successes,
                "successful_trial_rate": successes / len(selected) if selected else 0.0,
                "latency_sample_count": len(latency_rows),
                "mean_monitor_to_actuator_ms": (
                    float(np.mean(latency_rows)) if latency_rows else None
                ),
                "observed_max_monitor_to_actuator_ms": (
                    float(np.max(latency_rows)) if latency_rows else None
                ),
            }
        )
    return aggregates


def _plot_live_threshold_sweep(
    aggregates: Sequence[Mapping[str, Any]], path: Path
) -> None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    thresholds = [float(item["threshold"]) for item in aggregates]
    high = [int(item["high_risk_count"]) for item in aggregates]
    low = [int(item["low_risk_count"]) for item in aggregates]
    authorization_active = [
        int(item["authorization_activity_count"]) for item in aggregates
    ]
    key_requests = [int(item["key_request_attempts"]) for item in aggregates]
    key_reuses = [int(item["key_reuses"]) for item in aggregates]
    transfer_attempts = [int(item["transfer_attempts"]) for item in aggregates]
    transfer_successes = [int(item["transfer_successes"]) for item in aggregates]

    def numeric_or_nan(value: object) -> float:
        return float(value) if isinstance(value, (int, float)) else float("nan")

    means = [numeric_or_nan(item["mean_monitor_to_actuator_ms"]) for item in aggregates]
    maxima = [
        numeric_or_nan(item["observed_max_monitor_to_actuator_ms"])
        for item in aggregates
    ]

    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    top.plot(thresholds, high, marker="o", label="high risk")
    top.plot(thresholds, low, marker="s", label="low risk")
    top.set_ylabel("measured trials")
    top.set_title("Risk classification vs threshold (fixed fixture/context pairs)")
    top.legend()
    top.grid(True, alpha=0.3)

    middle.plot(
        thresholds, authorization_active, marker="o", label="authorization active"
    )
    middle.plot(thresholds, key_requests, marker="s", label="key requests")
    middle.plot(thresholds, key_reuses, marker="^", label="key reuses")
    middle.plot(thresholds, transfer_attempts, marker="D", label="transfer attempts")
    middle.plot(thresholds, transfer_successes, marker="x", label="transfer successes")
    middle.set_ylabel("measured events")
    middle.set_title("Authorization, key-cache, and protected-transfer activity")
    middle.legend(ncol=2)
    middle.grid(True, alpha=0.3)

    bottom.plot(thresholds, means, marker="o", label="mean")
    bottom.plot(thresholds, maxima, marker="^", label="observed maximum")
    bottom.set_ylabel("monitor-to-actuator (ms)")
    bottom.set_xlabel("threshold")
    bottom.set_title("Measured monotonic interval through confirmed actuator decision")
    bottom.legend()
    bottom.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_threshold_sweep(
    root: Path,
    output_dir: Path,
    *,
    transaction_runner: Callable[..., MonitoredTransactionResult] | None = None,
    transport_factory: Callable[[], Any] | None = None,
    thresholds: Sequence[float] | None = None,
    cases: Sequence[SweepCase | Mapping[str, object]] | None = None,
    repetitions: int | None = None,
    warmup_repetitions: int | None = None,
) -> dict[str, Any]:
    """Run the default graded multi-threshold evaluation through live SST.

    ``transaction_runner`` and ``transport_factory`` are dependency-injection
    seams for unit tests.  With their defaults, runtime configuration and a
    reachable pinned Auth process are mandatory.  Each cache scope owns both
    its monitor transport and its actuator transport so both entities retain
    the same session key across repeated protected connections.
    """

    settings = scenario_settings(root, "threshold-sweep")
    configured_thresholds = (
        thresholds if thresholds is not None else settings.get("thresholds", [])
    )
    selected_thresholds = [float(value) for value in configured_thresholds]
    if not selected_thresholds:
        raise ValueError("threshold-sweep scenario declares no thresholds")
    if any(not np.isfinite(value) or value < 0 for value in selected_thresholds):
        raise ValueError("threshold-sweep thresholds must be finite and non-negative")
    if len(set(selected_thresholds)) != len(selected_thresholds):
        raise ValueError("threshold-sweep thresholds must be unique")
    selected_cases = _live_sweep_cases(
        cases if cases is not None else settings.get("cases")
    )
    measured_repetitions = int(
        repetitions if repetitions is not None else settings.get("repetitions", 3)
    )
    warmups = int(
        warmup_repetitions
        if warmup_repetitions is not None
        else settings.get("warmup_repetitions", 1)
    )
    if measured_repetitions < 1:
        raise ValueError("threshold-sweep repetitions must be at least one")
    if warmups < 0:
        raise ValueError("threshold-sweep warmup_repetitions must not be negative")

    limits = load_physical_limits(root)
    fixture_map = {spec.name: spec.fixture for spec in course_fixture_specs(limits)}
    unknown = sorted({case.fixture_name for case in selected_cases} - set(fixture_map))
    if unknown:
        raise ValueError(f"threshold-sweep references unknown fixtures: {unknown}")
    context_map: dict[str, AuthorizationContext] = {
        "allowed": load_context(root, "allowed"),
        "denied": load_context(root, "denied"),
    }

    injected = transaction_runner is not None
    runner = transaction_runner or run_monitored_transaction_live
    actuator_config: Path | None = None
    actuator_factory: Callable[[], Any]
    if not injected:
        monitor_config, live_actuator_config = require_sst_runtime(root)
        actuator_config = live_actuator_config
        require_auth_running(root)
        monitor_factory = transport_factory or (
            lambda: SSTMonitorTransport(
                monitor_config,
                host="127.0.0.1",
                port=23100,
                timeout=10.0,
            )
        )
        def new_live_actuator_transport() -> SSTActuatorTransport:
            return SSTActuatorTransport(
                live_actuator_config,
                host="127.0.0.1",
                port=23100,
                accept_timeout=15.0,
            )

        actuator_factory = new_live_actuator_transport
    else:
        # A fresh opaque token per cache scope lets fakes verify isolation
        # without importing or configuring IoTAuth.
        monitor_factory = transport_factory or (lambda: object())
        actuator_factory = object

    warmup_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    # Retain every cache object for the duration of the experiment.  Besides
    # making the cache scope explicit, this prevents opaque test doubles from
    # having object identities recycled between fixed pairs.
    transport_handles: list[tuple[Any, Any]] = []

    def new_transport_pair() -> tuple[Any, Any]:
        pair = (monitor_factory(), actuator_factory())
        transport_handles.append(pair)
        return pair

    def execute(
        *,
        threshold: float,
        case: SweepCase,
        repetition: int,
        phase: str,
        monitor_transport: Any,
        actuator_transport: Any,
    ) -> dict[str, Any]:
        fixture = fixture_map[case.fixture_name]
        action_chunk = _executed_fixture(fixture)
        context = context_map[case.context_name]
        expected_policy_outcome = (
            AuthorizationOutcome.ALLOWED.value
            if ContextPolicy.course_default().evaluate(context).allowed
            else AuthorizationOutcome.DENIED.value
        )
        trial_id = (
            f"tau-{threshold:g}__{case.case_id}__{phase}-{repetition:03d}"
        )
        started_ns = time.perf_counter_ns()
        try:
            result = runner(
                action_chunk=action_chunk,
                current_state=np.asarray(fixture.current_state, dtype=np.float64),
                limits=fixture.limits,
                threshold=threshold,
                context=context,
                run_id=trial_id,
                timestep=repetition,
                monitor_transport=monitor_transport,
                actuator_config=actuator_config,
                actuator_transport=actuator_transport,
            )
            if not isinstance(result, MonitoredTransactionResult):
                raise TypeError("transaction runner returned the wrong result type")
        except Exception as exc:  # noqa: BLE001 - failures remain explicit trial rows.
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            return _row_from_runner_exception(
                exc,
                elapsed_ms=elapsed_ms,
                fixture=fixture,
                action_chunk=action_chunk,
                threshold=threshold,
                case=case,
                trial_id=trial_id,
                phase=phase,
                repetition=repetition,
                expected_policy_outcome=expected_policy_outcome,
            )
        return _row_from_transaction(
            result,
            trial_id=trial_id,
            phase=phase,
            threshold=threshold,
            case=case,
            repetition=repetition,
            expected_policy_outcome=expected_policy_outcome,
        )

    for threshold in selected_thresholds:
        for case in selected_cases:
            # Warmups have their own transport/cache and are never included in
            # measured rows or aggregate denominators.
            warmup_monitor, warmup_actuator = new_transport_pair()
            for repetition in range(warmups):
                warmup_rows.append(
                    execute(
                        threshold=threshold,
                        case=case,
                        repetition=repetition,
                        phase="warmup",
                        monitor_transport=warmup_monitor,
                        actuator_transport=warmup_actuator,
                    )
                )
            # Exactly one transport/cache is reused by the measured repetitions
            # of this fixed (threshold, case) pair.  No cache crosses a context,
            # fixture, or threshold boundary.
            measured_monitor, measured_actuator = new_transport_pair()
            for repetition in range(measured_repetitions):
                measurement_rows.append(
                    execute(
                        threshold=threshold,
                        case=case,
                        repetition=repetition,
                        phase="measurement",
                        monitor_transport=measured_monitor,
                        actuator_transport=measured_actuator,
                    )
                )

    aggregates = _aggregate_live_trials(measurement_rows, selected_thresholds)
    warmup_failures = sum(row["trial_status"] == "failure" for row in warmup_rows)
    measurement_failures = sum(
        row["trial_status"] == "failure" for row in measurement_rows
    )
    failure_rows = [
        row
        for row in [*warmup_rows, *measurement_rows]
        if row["trial_status"] == "failure"
    ]
    failure_diagnostics = [
        {
            "trial_id": row["trial_id"],
            "phase": row["phase"],
            "threshold": row["threshold"],
            "case_id": row["case_id"],
            "repetition": row["repetition"],
            "failure_stage": row["failure_stage"],
            "failure_reason": row["failure_reason"],
            "auth_error_type": row["auth_error_type"],
            "transfer_error_type": row["transfer_error_type"],
            "receive_error_type": row["receive_error_type"],
        }
        for row in failure_rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(
            handle, fieldnames=list(_LIVE_TRIAL_FIELDS)
        )
        writer.writeheader()
        writer.writerows(measurement_rows)

    trials_json_path = output_dir / "trials.json"
    trials_json_path.write_text(
        json.dumps(
            {
                "analysis": "threshold_sweep",
                "measurement_trials": measurement_rows,
                "warmup_trials": warmup_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "analysis": "threshold_sweep",
        "authorization_mode": (
            "injected_transaction_runner_test_double" if injected else LIVE_AUTHORIZATION_MODE
        ),
        "action_source": "diagnostic_synthetic (fixed fixtures, no model inference)",
        "thresholds": selected_thresholds,
        "high_risk_comparison": ">=",
        "fixed_cases": [
            {
                "id": case.case_id,
                "fixture": case.fixture_name,
                "context": case.context_name,
            }
            for case in selected_cases
        ],
        "measured_repetitions_per_case_threshold": measured_repetitions,
        "warmup_repetitions_per_case_threshold": warmups,
        "warmup_trial_count": len(warmup_rows),
        "warmup_failure_count": warmup_failures,
        "failure_diagnostics": failure_diagnostics,
        "cache_policy": (
            "one isolated monitor/actuator cache pair per fixed case and threshold "
            "for measured repetitions; warmups use a separate throwaway pair"
        ),
        "authorization_timeout_policy": (
            "each default live monitor session-key request has a 10-second timeout "
            "and is attempted exactly once; requests are not retried because key "
            "issuance is not idempotent"
        ),
        "rate_denominators": {
            "high_risk_rate": "valid measured trials at the same threshold",
            "authorization_activity_rate": "valid measured trials at the same threshold",
            "successful_trial_rate": "all measured trials at the same threshold",
        },
        "latency_definition": (
            "one monotonic interval from monitor receipt before motion analysis through "
            "the confirmed actuator EXECUTE or HOLD decision; component timings are not summed"
        ),
        "latency_qualification": (
            "observed measurements only; observed maximum is not a worst-case bound"
        ),
        "failure_latency_policy": (
            "confirmed EXECUTE/HOLD intervals remain in latency aggregates even when the "
            "trial failed; failures before a confirmed actuator decision have a null "
            "monitor_to_actuator_ms and retain elapsed_until_failure_ms separately"
        ),
        "threshold_aggregates": aggregates,
        "measured_trial_count": len(measurement_rows),
        "failed_trial_count": measurement_failures,
        "note": (
            "Motion classification, policy authorization, protected transfer, and physical "
            "safety are distinct. A high-risk classification activates authorization; it "
            "does not itself allow or prove the physical safety of an action."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_path = output_dir / "threshold_sweep.png"
    _plot_live_threshold_sweep(aggregates, plot_path)
    result = {
        "mode": "injected_test_double" if injected else "live_sst",
        "csv": str(csv_path),
        "json": str(trials_json_path),
        "summary": str(summary_path),
        "plot": str(plot_path),
        "trials": measurement_rows,
        "warmup_trials": warmup_rows,
        "threshold_aggregates": aggregates,
        "failed_trial_count": measurement_failures,
        "warmup_failure_count": warmup_failures,
    }
    if not injected and (measurement_failures or warmup_failures):
        preview = " | ".join(
            f"{item['trial_id']} [{item['failure_stage']}]: {item['failure_reason']}"
            for item in failure_diagnostics[:3]
        )
        raise ScenarioError(
            "live multi-threshold evaluation recorded "
            f"{measurement_failures} measured and {warmup_failures} warmup failure(s); "
            f"artifacts were preserved under {output_dir}; first failure(s): {preview}"
        )
    return result


# ---------------------------------------------------------------------------
# Offline session-key validity diagnostic
# ---------------------------------------------------------------------------


class ManualClock:
    """A monotonic manual clock so validity behavior is exactly reproducible."""

    def __init__(self, value: float = 1_000.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


@dataclass(frozen=True, slots=True)
class ValidityTrial:
    validity_seconds: float
    key_requests: int
    key_reuses: int
    allowed_while_valid: int
    denied_after_context_change: int
    security_window_sends: int
    denied_after_expiry: int
    reissued_after_expiry: bool


def run_validity_diagnostic(root: Path, output_dir: Path) -> dict[str, Any]:
    """Model reuse, a context change, and expiration for several validity periods.

    This is a deterministic offline diagnostic.  It is explicitly not a live SST
    result and is labeled as such in the record.  It demonstrates the security
    window created when a session key remains valid after the authorizing
    context has changed to a denied state.
    """

    settings = scenario_settings(root, "grad-key-validity")
    periods = [float(value) for value in settings.get("validity_seconds", [])]
    if not periods:
        raise ValueError("grad-key-validity scenario declares no validity periods")
    allowed_context = load_context(root, "allowed")
    denied_context = load_context(root, "denied")

    trials: list[ValidityTrial] = []
    for validity in periods:
        clock = ManualClock()
        monitor_store = InMemorySessionKeyStore(clock=clock)
        actuator_store = InMemorySessionKeyStore(clock=clock)
        authorizer = OfflineContextAuthorizer(
            ContextPolicy.course_default(),
            monitor_store,
            actuator_store,
            validity_seconds=validity,
            clock=clock,
        )

        # 1. Authorize in the allowed context and issue a key.
        first = authorizer.authorize(allowed_context)
        assert first.allowed and not first.session_key_reused
        # 2. Reuse the still-valid key for a second allowed request.
        clock.advance(validity * 0.25)
        reuse = authorizer.authorize(allowed_context)
        assert reuse.allowed and reuse.session_key_reused
        # 3. The context changes to a denied state.  A fresh authorization is
        #    correctly denied, but the previously issued key is still valid.
        clock.advance(validity * 0.25)
        denied = authorizer.authorize(denied_context)
        # 4. Security window: the still-valid key can still protect an action
        #    even though the current context would be denied.
        security_window = monitor_store.has_valid(first.key_id or "")
        # 5. After expiration the key is gone and a denied context stays denied.
        clock.advance(validity * 2.0)
        expired_valid = monitor_store.has_valid(first.key_id or "")
        post_expiry = authorizer.authorize(denied_context)
        # 6. A later allowed context reissues a new key.
        reissue = authorizer.authorize(allowed_context)

        trials.append(
            ValidityTrial(
                validity_seconds=validity,
                key_requests=2 if reissue.allowed else 1,
                key_reuses=1,
                allowed_while_valid=2,
                denied_after_context_change=1 if not denied.allowed else 0,
                security_window_sends=1 if security_window else 0,
                denied_after_expiry=1 if not post_expiry.allowed else 0,
                reissued_after_expiry=bool(reissue.allowed and not expired_valid),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "validity_seconds": trial.validity_seconds,
            "key_requests": trial.key_requests,
            "key_reuses": trial.key_reuses,
            "allowed_while_valid": trial.allowed_while_valid,
            "denied_after_context_change": trial.denied_after_context_change,
            "security_window_sends": trial.security_window_sends,
            "denied_after_expiry": trial.denied_after_expiry,
            "reissued_after_expiry": trial.reissued_after_expiry,
        }
        for trial in trials
    ]
    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "analysis": "session_key_validity",
        "authorization_mode": "offline_diagnostic (OfflineContextAuthorizer, not live SST)",
        "validity_periods_seconds": periods,
        "trials": rows,
        "security_window": (
            "A session key issued in an allowed context stays usable until it "
            "expires. If the context changes to a denied state while the key is "
            "valid, protected actions can still be delivered. Shorter relative "
            "validity narrows this window at the cost of more key requests."
        ),
        "limitation": (
            "This offline diagnostic models SST reuse and expiration on a manual "
            "clock. The graded extension repeats the controlled sequence against "
            "the real pinned Auth deployment for three configured relative "
            "validity periods."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "summary": str(summary_path), "trials": rows}

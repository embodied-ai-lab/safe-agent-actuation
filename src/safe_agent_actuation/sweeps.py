"""Threshold sweep and session-key validity analysis over fixed action sets.

Neither analysis performs new model inference.  The threshold sweep reuses one
committed fixture set for every threshold.  The key-validity diagnostic is a
deterministic offline model of SST reuse and expiration used for tests and the
labeled offline command; the graded extension drives real SST separately.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .context import ContextPolicy
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
from .types import AuthorizationOutcome

# The sweep evaluates the whole committed fixture set, not just the scenario
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
    worst_monitor_to_actuator_ms: float

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
            "worst_monitor_to_actuator_latency_ms": round(
                self.worst_monitor_to_actuator_ms, 6
            ),
        }


def _sweep_fixtures(root: Path) -> list[ActionFixture]:
    limits = load_physical_limits(root)
    specs = {spec.name: spec for spec in course_fixture_specs(limits)}
    return [specs[name].fixture for name in SWEEP_FIXTURE_NAMES]


def evaluate_threshold(
    fixtures: Sequence[ActionFixture],
    threshold: float,
    *,
    allowed_context: Any,
    denied_context: Any,
) -> ThresholdTrial:
    """Classify every fixture at one threshold and simulate authorization.

    The allowed context stands in for a high-risk action that is authorized;
    the denied context stands in for one that is refused.  This isolates the
    policy layer from motion classification: the same high-risk motion is
    allowed or denied purely by context.
    """

    import time

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
    for fixture in fixtures:
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
        # A high-risk action requests authorization; alternate the two
        # committed contexts so the sweep exercises both policy outcomes.
        context = allowed_context if (high % 2 == 1) else denied_context
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
        worst_monitor_to_actuator_ms=float(np.max(totals)) if totals else 0.0,
    )


def run_threshold_sweep(root: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate the committed thresholds and write CSV, JSON, and a plot."""

    settings = scenario_settings(root, "threshold-sweep")
    thresholds = [float(value) for value in settings.get("thresholds", [])]
    if not thresholds:
        raise ValueError("threshold-sweep scenario declares no thresholds")
    fixtures = _sweep_fixtures(root)
    allowed_context = load_context(root, "allowed")
    denied_context = load_context(root, "denied")
    trials = [
        evaluate_threshold(
            fixtures, threshold, allowed_context=allowed_context, denied_context=denied_context
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
        "action_source": "diagnostic_synthetic (committed fixtures, no model inference)",
        "fixture_count": len(fixtures),
        "fixtures": list(SWEEP_FIXTURE_NAMES),
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [trial.threshold for trial in trials]
    high = [trial.high_risk for trial in trials]
    low = [trial.low_risk for trial in trials]
    worst = [trial.worst_monitor_to_actuator_ms for trial in trials]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    top.plot(thresholds, high, marker="o", label="high risk")
    top.plot(thresholds, low, marker="s", label="low risk")
    top.set_ylabel("fixtures")
    top.set_title("Risk classification vs threshold (fixed fixture set)")
    top.legend()
    top.grid(True, alpha=0.3)

    bottom.plot(thresholds, worst, marker="^", color="tab:red")
    bottom.set_ylabel("worst monitor-to-actuator (ms)")
    bottom.set_xlabel("threshold")
    bottom.set_title("Worst-case latency vs threshold")
    bottom.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


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

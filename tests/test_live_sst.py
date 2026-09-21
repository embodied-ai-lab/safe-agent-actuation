"""Live SST integration tests.

These require generated SST runtime state and a running Auth process, so they
carry the ``live_sst`` marker and are excluded from the default offline run.
Run them with generated state and Auth started:

    make build-auth && make generate-sst && make auth-start
    make test-sst
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_agent_actuation.course_config import (
    load_context,
    load_physical_limits,
    repository_root,
)
from safe_agent_actuation.scenarios import (
    require_sst_runtime,
    run_bypass,
    run_high_risk_live,
)
from safe_agent_actuation.sweeps import SweepCase, run_threshold_sweep

pytestmark = pytest.mark.live_sst


def root():
    return repository_root()


def test_runtime_state_present():
    # Fails with a clear message if generate-sst has not been run.
    monitor_config, actuator_config = require_sst_runtime(root())
    assert monitor_config.is_file()
    assert actuator_config.is_file()


def test_high_risk_allow_completes_through_real_sst():
    outcome = run_high_risk_live(
        root(),
        scenario="high-risk-allow",
        threshold=0.1,
        context_name="allowed",
        expect_allowed=True,
    )
    assert outcome.expected
    assert outcome.summary["authorization"] == "allowed"
    assert outcome.summary["actuator_decision"] == "execute"
    assert outcome.summary["simulator_executions"] == 1


def test_high_risk_deny_holds_through_real_sst():
    outcome = run_high_risk_live(
        root(),
        scenario="high-risk-deny",
        threshold=0.1,
        context_name="denied",
        expect_allowed=False,
    )
    assert outcome.expected
    assert outcome.summary["authorization"] == "denied"
    assert outcome.summary["actuator_decision"] == "hold"
    assert outcome.summary["simulator_executions"] == 0


def test_plain_tcp_bypass_is_rejected_by_real_sst():
    outcome = run_bypass(root(), threshold=0.1, offline_diagnostic=False)
    assert outcome.expected
    network = outcome.summary["network_bypass"]
    assert network["connected"] is True
    assert network["handshake_authenticated"] is False
    assert network["protected_action_accepted"] is False
    assert outcome.summary["simulator_executions"] == 0


def test_context_mapping_matches_policy():
    # The allowed context must map to the exact SST policy keys.
    from safe_agent_actuation.sst_adapter import context_to_sst

    allowed = load_context(root(), "allowed")
    mapped = context_to_sst(allowed.to_dict())
    assert set(mapped) == {
        "Location",
        "Number of People",
        "Time of Day",
        "Task Name",
        "Robot ID",
    }
    limits = load_physical_limits(root())
    assert limits.dimensions == 14
    assert np.asarray(limits.lower).shape == (14,)


def test_small_threshold_sweep_uses_real_sst_and_measured_reuse(tmp_path):
    result = run_threshold_sweep(
        root(),
        tmp_path,
        thresholds=(0.1,),
        cases=(
            SweepCase("high-risk-allowed", "high_risk", "allowed"),
            SweepCase("high-risk-denied", "high_risk", "denied"),
        ),
        repetitions=2,
        warmup_repetitions=1,
    )

    assert result["mode"] == "live_sst"
    assert len(result["warmup_trials"]) == 2
    assert len(result["trials"]) == 4
    aggregate = result["threshold_aggregates"][0]
    assert aggregate["high_risk_count"] == 4
    assert aggregate["authorization_activity_count"] == 4
    assert aggregate["key_request_attempts"] == 3
    assert aggregate["successful_key_requests"] == 1
    assert aggregate["key_reuses"] == 1
    assert aggregate["authorization_allowed_count"] == 2
    assert aggregate["authorization_denied_count"] == 2
    assert aggregate["controlled_policy_denial_count"] == 2
    assert aggregate["transfer_attempts"] == 2
    assert aggregate["transfer_successes"] == 2
    assert aggregate["failed_trial_count"] == 0
    assert aggregate["latency_sample_count"] == 4
    assert aggregate["observed_max_monitor_to_actuator_ms"] >= 0.0

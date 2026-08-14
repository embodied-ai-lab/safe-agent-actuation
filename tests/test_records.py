from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _analysis import high_risk_analysis

from safe_agent_actuation.context import DeterministicContextProvider
from safe_agent_actuation.records import ActionRecord, LatencyRecord, write_action_record
from safe_agent_actuation.types import (
    ActuationResult,
    ActuatorDecision,
    AuthorizationDecision,
    AuthorizationOutcome,
)


def complete_record() -> ActionRecord:
    motion = high_risk_analysis(("joint", "gripper"), motion_score=0.25, threshold=0.2)
    authorization = AuthorizationDecision(
        AuthorizationOutcome.ALLOWED,
        "context authorized",
        key_id="public-key-id",
        session_key_reused=False,
        latency_ms=2.0,
    )
    actuation = ActuationResult(
        ActuatorDecision.EXECUTE,
        "verified action forwarded to the simulator",
        "sst_protected_high_risk",
        (2, 2),
        decryption_latency_ms=0.2,
        total_latency_ms=3.2,
    )
    return ActionRecord(
        run_id="run-001",
        timestep=3,
        task_prompt="transfer the cube",
        model_configuration="pi0_fast_aloha_sim",
        action_source="diagnostic_synthetic",
        raw_fast_token_ids=(12, 34, 56),
        decoded_action_shape=(2, 2),
        executed_horizon=1,
        current_observed_joint_state=(0.0, 0.0),
        physical_limits_source="verified simulator limits",
        motion_analysis=motion,
        context=DeterministicContextProvider.allowed().build(),
        authorization_requested=True,
        authorization=authorization,
        latencies=LatencyRecord(
            inference_ms=100.0,
            decode_ms=5.0,
            monitoring_ms=0.1,
            authorization_ms=2.0,
            encryption_ms=0.3,
            decryption_ms=0.2,
            total_monitor_to_actuator_ms=3.2,
        ),
        actuation=actuation,
        simulator_outcome="executed one command in headless simulator",
        metadata={"openpi_commit": "abc123"},
    )


def test_latency_record_schema_has_requested_fields() -> None:
    latency = LatencyRecord(
        monitoring_ms=1,
        authorization_ms=2,
        encryption_ms=3,
        decryption_ms=4,
        total_monitor_to_actuator_ms=10,
    )
    assert latency.to_dict() == {
        "inference_ms": None,
        "decode_ms": None,
        "monitoring_ms": 1.0,
        "authorization_ms": 2.0,
        "encryption_ms": 3.0,
        "decryption_ms": 4.0,
        "total_monitor_to_actuator_ms": 10.0,
    }
    assert latency.total_monitor_to_actuator_latency_ms == 10.0


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_latency_record_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        LatencyRecord(monitoring_ms=value)


def test_action_record_to_dict_and_json_are_concise_and_structured() -> None:
    record = complete_record()
    data = record.to_dict()
    assert data["token_count"] == 3
    assert data["raw_fast_token_ids"] == [12, 34, 56]
    assert data["motion_analysis"]["max_dimension"] == "joint"
    assert data["authorization"]["outcome"] == "allowed"
    assert data["actuation"]["decision"] == "execute"
    assert data["decoded_action_shape"] == [2, 2]
    assert "action_chunk" not in data
    assert json.loads(record.to_json())["run_id"] == "run-001"


def test_action_record_rejects_secret_bearing_metadata() -> None:
    with pytest.raises(ValueError, match="secret-bearing"):
        ActionRecord(
            "run",
            0,
            "task",
            "model",
            "fixture",
            metadata={"session": {"cipher_key": "do-not-log"}},
        )


def test_write_action_record_writes_json_and_compact_npz(tmp_path: Path) -> None:
    record = complete_record()
    output = write_action_record(
        record,
        tmp_path / "record.json",
        action_chunk=[[0.0, 0.0], [0.5, 0.0]],
        npz_path=tmp_path / "arrays" / "action.npz",
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["action_npz_path"] == "arrays/action.npz"
    with np.load(tmp_path / data["action_npz_path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["action_chunk"], [[0.0, 0.0], [0.5, 0.0]])


def test_record_requires_authorization_result_when_requested() -> None:
    with pytest.raises(ValueError, match="requires"):
        ActionRecord(
            "run",
            0,
            "task",
            "model",
            "fixture",
            authorization_requested=True,
        )

"""Tests for the Part 1 pi0-FAST response record."""

from __future__ import annotations

import json

import numpy as np
import pytest

from safe_agent_actuation.captured import (
    CHUNK_NAME,
    RECORD_NAME,
    CapturedRecordError,
    captured_record_available,
    configuration_report,
    inspect_response,
    load_captured_response,
    sha256_file,
)
from safe_agent_actuation.course_config import repository_root


def _write_record(root, *, source_type="captured_live_pi0_fast", shape=(10, 8)):
    directory = root / "data" / "captured"
    directory.mkdir(parents=True, exist_ok=True)
    chunk = np.linspace(-0.5, 0.5, shape[0] * shape[1]).reshape(shape)
    chunk_path = directory / CHUNK_NAME
    np.savez_compressed(chunk_path, action_chunk=np.asarray(chunk, dtype="<f8"))
    metadata = {
        "source_type": source_type,
        "openpi_url": "https://github.com/Physical-Intelligence/openpi.git",
        "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "model_family": "pi0-FAST",
        "model_configuration": "pi0_fast_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/pi0_fast_droid",
        "fast_tokenizer_repository": ("https://huggingface.co/physical-intelligence/fast"),
        "fast_tokenizer_revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
        "fast_tokenizer_code_sha256": (
            "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20"
        ),
        "embodiment": "DROID Franka",
        "task_prompt": "pick up the bowl",
        "input_origin": "openpi.policies.droid_policy.make_droid_example",
        "input_description": "seeded offline OpenPI example observation",
        "input_seed": 0,
        "input_shapes": {
            "observation/exterior_image_1_left": [224, 224, 3],
            "observation/gripper_position": [1],
            "observation/joint_position": [7],
            "observation/wrist_image_left": [224, 224, 3],
            "prompt": "text",
        },
        "input_observation_sha256": "a" * 64,
        "tokens_available": False,
        "token_count": None,
        "token_availability_reason": "Policy.infer does not expose generated tokens.",
        "decode_ok": True,
        "decode_boundary_status": "verified_action_payload",
        "action_marker_present": True,
        "decoded_action_shape_before_unnormalization": [10, 8],
        "decode_observer": "capture-only tokenizer before unnormalization",
        "decode_validation": "action marker and decoded boundary shape verified",
        "decoded_model_action_shape": [10, 8],
        "decoded_model_action_shape_source": "pinned OpenPI model configuration",
        "final_action_shape": list(shape),
        "response_action_dtype": "float32",
        "configured_action_horizon": 10,
        "returned_action_horizon": shape[0],
        "actions_executed": False,
        "executed_horizon": 0,
        "inference_ms": 15.0,
        "end_to_end_infer_ms": 20.0,
        "decode_ms": None,
        "response_keys": ["actions", "policy_timing", "state"],
        "capture_date": "2026-08-13",
        "capture_timestamp_utc": "2026-08-13T12:00:00Z",
        "capture_environment": "ASU Sol Slurm GPU allocation",
        "action_chunk_sha256": sha256_file(chunk_path),
        "limitation": "different embodiment; actions were not executed",
    }
    (directory / RECORD_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def test_missing_record_reports_clearly(tmp_path):
    assert captured_record_available(tmp_path) is False
    with pytest.raises(CapturedRecordError, match="missing"):
        load_captured_response(tmp_path)


def test_valid_record_loads_and_summarizes(tmp_path):
    _write_record(tmp_path)
    response = load_captured_response(tmp_path)
    summary = response.summary()
    assert summary["source_type"] == "captured_live_pi0_fast"
    assert summary["final_action_shape"] == [10, 8]
    assert summary["returned_action_horizon"] == 10
    assert summary["actions_executed"] is False
    assert summary["executed_horizon"] == 0
    assert summary["action_chunk_sha256"] == sha256_file(response.chunk_path)
    assert response.executed_prefix().shape == (0, 8)


def test_synthetic_label_is_rejected(tmp_path):
    _write_record(tmp_path, source_type="diagnostic_synthetic")
    with pytest.raises(CapturedRecordError, match="source_type"):
        load_captured_response(tmp_path)


def test_tampered_chunk_is_rejected(tmp_path):
    directory = _write_record(tmp_path)
    np.savez_compressed(directory / CHUNK_NAME, action_chunk=np.zeros((10, 8), dtype="<f8"))
    with pytest.raises(CapturedRecordError, match="SHA-256"):
        load_captured_response(tmp_path)


def test_shape_mismatch_is_rejected(tmp_path):
    directory = _write_record(tmp_path)
    record = json.loads((directory / RECORD_NAME).read_text())
    record["final_action_shape"] = [5, 8]
    (directory / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="action shapes"):
        load_captured_response(tmp_path)


def test_unverified_checkpoint_label_is_rejected(tmp_path):
    directory = _write_record(tmp_path)
    record_path = directory / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["checkpoint_uri"] = "gs://openpi-assets/checkpoints/pi0_fast_libero"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="verified Part 1 provenance"):
        load_captured_response(tmp_path)


def test_checksum_is_required(tmp_path):
    directory = _write_record(tmp_path)
    record = json.loads((directory / RECORD_NAME).read_text())
    record.pop("action_chunk_sha256")
    (directory / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="missing fields.*action_chunk_sha256"):
        load_captured_response(tmp_path)


def test_corrupt_npz_is_rejected_after_checksum_validation(tmp_path):
    directory = _write_record(tmp_path)
    chunk_path = directory / CHUNK_NAME
    chunk_path.write_bytes(b"not an npz archive")
    record_path = directory / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["action_chunk_sha256"] = sha256_file(chunk_path)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="malformed"):
        load_captured_response(tmp_path)


def test_decode_failure_sentinel_is_rejected(tmp_path):
    directory = _write_record(tmp_path)
    chunk_path = directory / CHUNK_NAME
    np.savez_compressed(chunk_path, action_chunk=np.zeros((10, 8), dtype="<f8"))
    record_path = directory / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["action_chunk_sha256"] = sha256_file(chunk_path)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="all zero"):
        load_captured_response(tmp_path)


def test_capture_cannot_claim_execution(tmp_path):
    directory = _write_record(tmp_path)
    record_path = directory / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["actions_executed"] = True
    record["executed_horizon"] = 10
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="must not claim execution"):
        load_captured_response(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decode_ok", False),
        ("decode_boundary_status", "failed_missing_action_marker"),
        ("action_marker_present", False),
        ("decoded_action_shape_before_unnormalization", [1, 8]),
    ],
)
def test_missing_positive_decode_boundary_evidence_is_rejected(tmp_path, field, value):
    directory = _write_record(tmp_path)
    record_path = directory / RECORD_NAME
    record = json.loads(record_path.read_text())
    record[field] = value
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CapturedRecordError, match="decode|action shapes"):
        load_captured_response(tmp_path)


def test_configuration_report_is_honest_about_tokens():
    report = configuration_report(repository_root())
    assert report["action_source"] == "model_configuration_only"
    assert report["response_available"] is False
    assert report["tokens_available"] is False
    assert report["token_count"] is None
    assert "not available" in report["token_availability_reason"].lower()
    assert report["final_action_shape"] == [32, 14]
    assert report["configured_execution_horizon"] == 10
    assert report["actions_executed"] is False
    assert report["executed_horizon"] == 0


def test_inspect_response_prefers_a_captured_record(tmp_path):
    _write_record(tmp_path)
    summary = inspect_response(tmp_path)
    assert summary["action_source"] == "captured_live_pi0_fast"
    assert summary["response_available"] is True


def test_inspect_response_requires_capture(tmp_path):
    with pytest.raises(CapturedRecordError, match="missing"):
        inspect_response(tmp_path)


def test_configuration_only_diagnostic_is_explicit():
    summary = inspect_response(repository_root(), configuration_only=True)
    assert summary["action_source"] == "model_configuration_only"
    assert summary["response_available"] is False
    assert "final_action_shape" in summary

#!/usr/bin/env python3
"""Validate final common and graduate experiment artifacts.

The final ``results/`` check validates action records, the genuine Part 1
capture summary, per-trial multi-threshold evaluation CSV/JSON agreement,
independently recomputed evaluation aggregates, and (when present or requested) the real-SST
graduate validity sequence. Individual action-record files and scenario
directories remain valid focused inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

REQUIRED_TOP = (
    "run_id",
    "timestep",
    "action_source",
    "model_configuration",
    "motion_analysis",
    "authorization_requested",
    "actuation",
    "simulator_outcome",
)
REQUIRED_MOTION = ("valid", "classification", "motion_score", "threshold")
KNOWN_SOURCES = {
    "captured_live_pi0_fast",
    "diagnostic_synthetic",
    "generated_boundary_test",
}
HELD_DECISIONS = {"hold", "blocked"}
EXECUTED_OUTCOMES = {"executed", "executed_in_recording_harness"}
LIVE_MODE_PREFIX = "live_sst"
COMMON_SCENARIOS = ("low-risk", "high-risk-allow", "high-risk-deny", "bypass")
GRAD_STEPS = (
    "authorize_allowed",
    "reuse_valid_key",
    "security_window_denied_context",
    "denied_after_expiry",
    "reissue_allowed",
)
GRAD_PERIODS = (
    ("5*sec", 5_000),
    ("20*sec", 20_000),
    ("40*sec", 40_000),
)
GRAD_COUNTERS: dict[str, int | bool] = {
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
GRAD_TOTAL_FIELDS = tuple(GRAD_COUNTERS)
GRAD_CSV_FIELDS = (
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
)
SWEEP_FAILURE_DIAGNOSTIC_FIELDS = (
    "trial_id",
    "phase",
    "threshold",
    "case_id",
    "repetition",
    "failure_stage",
    "failure_reason",
    "auth_error_type",
    "transfer_error_type",
    "receive_error_type",
)
GRAD_ROUND_EXPECTATIONS: dict[str, dict[str, object]] = {
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
CAPTURE_PROVENANCE = {
    "openpi_url": "https://github.com/Physical-Intelligence/openpi.git",
    "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "model_configuration": "pi0_fast_droid",
    "checkpoint_uri": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "input_origin": "openpi.policies.droid_policy.make_droid_example",
    "fast_tokenizer_repository": "https://huggingface.co/physical-intelligence/fast",
    "fast_tokenizer_revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
    "fast_tokenizer_code_sha256": (
        "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20"
    ),
}
CAPTURE_ACTION_SHAPE = [10, 8]


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _nonnegative(value: object) -> TypeGuard[int | float]:
    return _finite_number(value) and float(value) >= 0.0


def _contains_secret_key(value: object) -> bool:
    sensitive = {
        "cipher_key",
        "distribution_key",
        "mac_key",
        "password",
        "private_key",
        "session_key",
        "secret",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in sensitive or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _load_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path}: must be a JSON object")
        return None
    if _contains_secret_key(payload):
        errors.append(f"{path}: contains a secret-bearing JSON key")
    return payload


def validate_record(path: Path, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TOP:
        if field not in record:
            errors.append(f"{path}: missing {field}")

    source = record.get("action_source")
    if source is not None and source not in KNOWN_SOURCES:
        errors.append(f"{path}: unknown action_source {source!r}")

    motion = record.get("motion_analysis")
    if isinstance(motion, dict):
        for field in REQUIRED_MOTION:
            if field not in motion:
                errors.append(f"{path}: motion_analysis missing {field}")
        score = motion.get("motion_score")
        threshold = motion.get("threshold")
        if not _finite_number(score):
            errors.append(f"{path}: motion_score must be finite")
        if not _finite_number(threshold):
            errors.append(f"{path}: threshold must be finite")
        classification = motion.get("classification")
        if motion.get("valid") and _finite_number(score) and _finite_number(threshold):
            expected = "high" if float(score) >= float(threshold) else "low"
            if classification != expected:
                errors.append(
                    f"{path}: classification {classification!r} does not match the "
                    f">= threshold rule (expected {expected!r})"
                )
    elif motion is not None:
        errors.append(f"{path}: motion_analysis must be an object")

    actuation = record.get("actuation")
    decision = actuation.get("decision") if isinstance(actuation, dict) else None
    outcome = record.get("simulator_outcome")
    if decision in HELD_DECISIONS and outcome in EXECUTED_OUTCOMES:
        errors.append(f"{path}: a {decision} action was marked executed ({outcome})")

    if _contains_secret_key(record):
        errors.append(f"{path}: record contains a secret-bearing JSON key")
    text = json.dumps(record).lower()
    for fragment in ("cipher_key", "mac_key", "private_key", "-----begin"):
        if fragment in text:
            errors.append(f"{path}: record contains secret-bearing text {fragment!r}")
    return errors


def validate_common_scenario(
    path: Path, scenario: str, record: Mapping[str, Any]
) -> list[str]:
    """Validate graded semantics when a record occupies a canonical path.

    Focused validation of one action record deliberately accepts explicitly
    labeled offline diagnostics. A final ``results/`` tree is stricter: the
    canonical Part 3 paths must contain real-SST evidence.
    """

    errors: list[str] = []
    prefix = f"{path}: {scenario}"

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{prefix}: {message}")

    motion = record.get("motion_analysis")
    authorization = record.get("authorization")
    actuation = record.get("actuation")
    metadata = record.get("metadata")
    motion_map = motion if isinstance(motion, Mapping) else {}
    authorization_map = authorization if isinstance(authorization, Mapping) else {}
    actuation_map = actuation if isinstance(actuation, Mapping) else {}
    metadata_map = metadata if isinstance(metadata, Mapping) else {}

    require(
        record.get("action_source") == "diagnostic_synthetic",
        "must use a synthetic fixture",
    )
    require(
        metadata_map.get("scenario") == scenario,
        "metadata.scenario does not match its path",
    )
    require(motion_map.get("valid") is True, "motion analysis must be valid")

    if scenario == "low-risk":
        require(
            motion_map.get("classification") == "low",
            "must be classified low risk",
        )
        require(
            record.get("authorization_requested") is False,
            "must not request authorization",
        )
        require(
            authorization_map.get("outcome") == "not_requested",
            "authorization outcome must be not_requested",
        )
        require(
            metadata_map.get("authorization_mode") == "not_requested",
            "must not use SST",
        )
        require(
            metadata_map.get("sst_required") is False,
            "metadata must state SST is not required",
        )
        require(actuation_map.get("decision") == "execute", "must record EXECUTE")
        require(
            actuation_map.get("path") == "trusted_low_risk",
            "must use the trusted low-risk path",
        )
        require(
            record.get("simulator_outcome") == "executed_in_recording_harness",
            "must record execution in the recording harness",
        )
        return errors

    # All remaining common scenarios are graded live-SST evidence.
    require(
        motion_map.get("classification") == "high",
        "must be classified high risk",
    )
    require(
        str(metadata_map.get("authorization_mode", "")).startswith(
            LIVE_MODE_PREFIX
        ),
        "must be labeled live SST, not an offline diagnostic",
    )
    require(
        metadata_map.get("sst_required") is True,
        "metadata must state SST is required",
    )

    if scenario == "high-risk-allow":
        require(
            record.get("authorization_requested") is True,
            "must request authorization",
        )
        require(
            authorization_map.get("outcome") == "allowed",
            "Auth outcome must be allowed",
        )
        require(
            metadata_map.get("context_name") == "allowed",
            "must use the allowed context",
        )
        require(
            metadata_map.get("authorization_status") == "allowed",
            "must record Auth allow status",
        )
        require(
            metadata_map.get("key_request_attempts") == 1,
            "must record one key-request attempt",
        )
        require(
            metadata_map.get("successful_key_requests") == 1,
            "must record one successful key request",
        )
        require(
            metadata_map.get("session_key_reused") is False,
            "standalone allow must use a fresh key",
        )
        require(
            metadata_map.get("transfer_attempted") is True,
            "must attempt protected transfer",
        )
        require(
            metadata_map.get("transfer_succeeded") is True,
            "protected transfer must succeed",
        )
        require(actuation_map.get("decision") == "execute", "must record EXECUTE")
        require(
            actuation_map.get("path") == "sst_protected_high_risk",
            "must use the SST-protected path",
        )
        require(
            record.get("simulator_outcome") == "executed_in_recording_harness",
            "must record execution in the recording harness",
        )
        return errors

    if scenario == "high-risk-deny":
        require(
            record.get("authorization_requested") is True,
            "must request authorization",
        )
        require(
            authorization_map.get("outcome") == "denied",
            "Auth outcome must be denied",
        )
        require(
            metadata_map.get("context_name") == "denied",
            "must use the denied context",
        )
        require(
            metadata_map.get("authorization_status") == "auth_rejected_request",
            "must record the controlled Auth rejection",
        )
        require(
            metadata_map.get("key_request_attempts") == 1,
            "must record one key-request attempt",
        )
        require(
            metadata_map.get("successful_key_requests") == 0,
            "denial must issue no key",
        )
        require(
            metadata_map.get("session_key_reused") is False,
            "denial must not reuse a key",
        )
        require(
            metadata_map.get("transfer_attempted") is False,
            "denial must withhold transfer",
        )
        require(
            metadata_map.get("transfer_succeeded") is False,
            "denial must not transfer",
        )
        require(actuation_map.get("decision") == "hold", "must record HOLD")
        require(
            actuation_map.get("path") == "sst_protected_high_risk",
            "must fail closed at the SST actuator path",
        )
        require(
            record.get("simulator_outcome") == "held_in_recording_harness",
            "must record HOLD in the recording harness",
        )
        return errors

    if scenario == "bypass":
        network = metadata_map.get("network_bypass")
        network_map = network if isinstance(network, Mapping) else {}
        bytes_sent = network_map.get("bytes_sent")
        require(
            record.get("authorization_requested") is False,
            "bypass must skip authorization",
        )
        require(
            authorization_map.get("outcome") == "not_requested",
            "authorization outcome must be not_requested",
        )
        require(actuation_map.get("decision") == "hold", "must record HOLD")
        require(
            actuation_map.get("path") == "bypass",
            "must use the bypass-rejection path",
        )
        require(
            record.get("simulator_outcome") == "held_in_recording_harness",
            "must record HOLD in the recording harness",
        )
        require(
            isinstance(network, Mapping),
            "must contain a live network_bypass result",
        )
        require(
            network_map.get("connected") is True,
            "plaintext client must reach a listening endpoint",
        )
        require(
            isinstance(bytes_sent, int)
            and not isinstance(bytes_sent, bool)
            and bytes_sent > 0,
            "plaintext bypass must send a nonempty payload",
        )
        require(
            network_map.get("handshake_authenticated") is False,
            "plaintext bypass must not authenticate",
        )
        require(
            network_map.get("protected_action_accepted") is False,
            "plaintext bypass must not be accepted",
        )
        server_reason = network_map.get("server_reason")
        require(
            isinstance(server_reason, str) and bool(server_reason.strip()),
            "must retain the actuator rejection reason",
        )
        return errors

    errors.append(f"{prefix}: unsupported common scenario")
    return errors


def discover(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("action_record.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"path not found: {path}")
    return found


def validate_capture_summary(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"{path}: missing required Part 1 result"]
    payload = _load_object(path, errors)
    if payload is None:
        return errors
    required = (
        "action_source",
        "response_available",
        "model_family",
        "model_configuration",
        "checkpoint_uri",
        "openpi_url",
        "openpi_commit",
        "fast_tokenizer_repository",
        "fast_tokenizer_revision",
        "fast_tokenizer_code_sha256",
        "input_origin",
        "input_observation_sha256",
        "tokens_available",
        "token_count",
        "token_availability_reason",
        "decode_ok",
        "decode_boundary_status",
        "action_marker_present",
        "decoded_action_shape_before_unnormalization",
        "decode_observer",
        "decode_validation",
        "decoded_model_action_shape",
        "final_action_shape",
        "loaded_action_shape",
        "configured_action_horizon",
        "returned_action_horizon",
        "actions_executed",
        "executed_horizon",
        "inference_ms",
        "end_to_end_infer_ms",
        "capture_environment",
        "capture_job_id",
        "action_chunk_sha256",
    )
    for field in required:
        if field not in payload:
            errors.append(f"{path}: missing {field}")
    if payload.get("action_source") != "captured_live_pi0_fast":
        errors.append(f"{path}: Part 1 must use captured_live_pi0_fast")
    if payload.get("response_available") is not True:
        errors.append(f"{path}: captured response must be available")
    if payload.get("model_family") != "pi0-FAST":
        errors.append(f"{path}: model_family must be pi0-FAST")
    for field, expected in CAPTURE_PROVENANCE.items():
        if payload.get(field) != expected:
            errors.append(
                f"{path}: {field} does not match the verified capture provenance"
            )
    if payload.get("decode_ok") is not True:
        errors.append(f"{path}: captured decode validation did not succeed")
    if (
        payload.get("decode_boundary_status") != "verified_action_payload"
        or payload.get("action_marker_present") is not True
    ):
        errors.append(f"{path}: positive FAST decode-boundary evidence is missing")
    if payload.get("decoded_action_shape_before_unnormalization") != CAPTURE_ACTION_SHAPE:
        errors.append(f"{path}: pre-unnormalization decoded shape must be [10, 8]")
    if payload.get("decoded_model_action_shape") != CAPTURE_ACTION_SHAPE:
        errors.append(f"{path}: decoded model action shape must be [10, 8]")
    for field in ("decode_observer", "decode_validation"):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{path}: {field} must explain decode-boundary validation")
    if payload.get("actions_executed") is not False or payload.get("executed_horizon") != 0:
        errors.append(f"{path}: offline response capture must not claim execution")
    if payload.get("tokens_available") is not False or payload.get("token_count") is not None:
        errors.append(f"{path}: capture must not invent token availability or a token count")
    if not str(payload.get("token_availability_reason") or "").strip():
        errors.append(f"{path}: unavailable tokens require a reason")
    final_shape = payload.get("final_action_shape")
    loaded_shape = payload.get("loaded_action_shape")
    if final_shape != loaded_shape:
        errors.append(f"{path}: loaded action shape does not match final action shape")
    if not (
        isinstance(final_shape, list)
        and len(final_shape) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in final_shape
        )
    ):
        errors.append(f"{path}: final action shape must be two positive integers")
    if isinstance(final_shape, list) and final_shape:
        if payload.get("returned_action_horizon") != final_shape[0]:
            errors.append(f"{path}: returned horizon does not match actual action shape")
    if final_shape != CAPTURE_ACTION_SHAPE:
        errors.append(f"{path}: final pi0-FAST-DROID action shape must be [10, 8]")
    if payload.get("configured_action_horizon") != 10:
        errors.append(f"{path}: configured action horizon must be 10")
    for field in ("inference_ms", "end_to_end_infer_ms"):
        value = payload.get(field)
        if not _finite_number(value) or float(value) <= 0.0:
            errors.append(f"{path}: {field} must be a positive measured duration")
    if not str(payload.get("capture_environment") or "").strip():
        errors.append(f"{path}: capture_environment is missing")
    if not str(payload.get("capture_job_id") or "").strip():
        errors.append(f"{path}: capture_job_id is missing")
    input_digest = payload.get("input_observation_sha256")
    if not (
        isinstance(input_digest, str)
        and len(input_digest) == 64
        and all(character in "0123456789abcdef" for character in input_digest)
    ):
        errors.append(f"{path}: input-observation checksum is not a lowercase SHA-256")
    digest = payload.get("action_chunk_sha256")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        errors.append(f"{path}: action checksum is not a lowercase SHA-256")
    return errors


def _csv_value(value: object) -> str:
    return "" if value is None else str(value)


def _rows_match_csv(
    csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            fields = reader.fieldnames or []
    except OSError as exc:
        return [f"{csv_path}: {exc}"]
    if len(csv_rows) != len(rows):
        errors.append(
            f"{csv_path}: has {len(csv_rows)} rows; JSON has {len(rows)} rows"
        )
        return errors
    if rows and set(fields) != set(rows[0]):
        errors.append(f"{csv_path}: header does not match JSON trial fields")
        return errors
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, rows, strict=True), start=2):
        for field in fields:
            if csv_row[field] != _csv_value(json_row.get(field)):
                errors.append(
                    f"{csv_path}:{index}: {field} disagrees with trials.json"
                )
    return errors


def _recompute_sweep(
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
            if _finite_number(row.get("monitor_to_actuator_ms"))
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
                    sum(latency_rows) / len(latency_rows) if latency_rows else None
                ),
                "observed_max_monitor_to_actuator_ms": (
                    max(latency_rows) if latency_rows else None
                ),
            }
        )
    return aggregates


def _equal_value(actual: object, expected: object) -> bool:
    if expected is None or isinstance(expected, (str, bool, int)):
        return actual == expected
    if isinstance(expected, float):
        return _finite_number(actual) and math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return actual == expected


def validate_sweep(directory: Path) -> list[str]:
    errors: list[str] = []
    csv_path = directory / "trials.csv"
    trials_path = directory / "trials.json"
    summary_path = directory / "summary.json"
    plot_path = directory / "threshold_sweep.png"
    for path in (csv_path, trials_path, summary_path, plot_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(
                f"{path}: missing or empty required multi-threshold evaluation artifact"
            )
    if errors:
        return errors
    trials_payload = _load_object(trials_path, errors)
    summary = _load_object(summary_path, errors)
    if trials_payload is None or summary is None:
        return errors
    rows = trials_payload.get("measurement_trials")
    warmups = trials_payload.get("warmup_trials")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return [*errors, f"{trials_path}: measurement_trials must be a list of objects"]
    if not isinstance(warmups, list) or not all(isinstance(row, dict) for row in warmups):
        errors.append(f"{trials_path}: warmup_trials must be a list of objects")
        return errors
    errors.extend(_rows_match_csv(csv_path, rows))

    for index, row in enumerate(rows):
        label = f"{trials_path}:measurement_trials[{index}]"
        score = row.get("motion_score")
        threshold = row.get("threshold")
        classification = row.get("risk_classification")
        if (
            row.get("analysis_valid") is True
            and _finite_number(score)
            and _finite_number(threshold)
        ):
            expected_class = "high" if float(score) >= float(threshold) else "low"
            if classification != expected_class:
                errors.append(f"{label}: classification violates the >= rule")
        if row.get("trial_status") not in {"success", "failure"}:
            errors.append(f"{label}: unknown trial_status")
        decision = row.get("actuator_decision")
        total = row.get("monitor_to_actuator_ms")
        failed_elapsed = row.get("elapsed_until_failure_ms")
        if decision in {"execute", "hold"} and not _nonnegative(total):
            errors.append(f"{label}: confirmed decision requires measured latency")
        if decision == "not_confirmed" and not _nonnegative(failed_elapsed):
            errors.append(f"{label}: unconfirmed failure requires elapsed failure time")
        if row.get("authorization_outcome") == "denied":
            if row.get("transfer_attempted") is not False or decision != "hold":
                errors.append(f"{label}: policy denial must withhold transfer and HOLD")
        if row.get("transfer_succeeded") is True and row.get("authenticated_received") is not True:
            errors.append(f"{label}: successful transfer was not authenticated and received")

    thresholds = summary.get("thresholds")
    observed = summary.get("threshold_aggregates")
    if not isinstance(thresholds, list) or not isinstance(observed, list):
        errors.append(f"{summary_path}: thresholds or threshold_aggregates are invalid")
        return errors
    numeric_thresholds: list[float] = []
    for value in thresholds:
        if not _finite_number(value):
            errors.append(f"{summary_path}: thresholds or threshold_aggregates are invalid")
            return errors
        numeric_thresholds.append(float(value))
    recomputed = _recompute_sweep(rows, numeric_thresholds)
    if len(observed) != len(recomputed):
        errors.append(f"{summary_path}: aggregate row count does not match thresholds")
    else:
        for index, (actual, expected) in enumerate(zip(observed, recomputed, strict=True)):
            if not isinstance(actual, Mapping):
                errors.append(f"{summary_path}: aggregate {index} must be an object")
                continue
            for field, value in expected.items():
                if not _equal_value(actual.get(field), value):
                    errors.append(
                        f"{summary_path}: aggregate {index} field {field} does not "
                        "recompute from measured trials"
                    )
    if not str(summary.get("authorization_mode", "")).startswith(LIVE_MODE_PREFIX):
        errors.append(
            f"{summary_path}: graded multi-threshold evaluation is not labeled live SST"
        )
    if summary.get("measured_trial_count") != len(rows):
        errors.append(f"{summary_path}: measured_trial_count is inconsistent")
    if summary.get("warmup_trial_count") != len(warmups):
        errors.append(f"{summary_path}: warmup_trial_count is inconsistent")
    failures = sum(row.get("trial_status") == "failure" for row in rows)
    if summary.get("failed_trial_count") != failures:
        errors.append(f"{summary_path}: failed_trial_count is inconsistent")
    if failures:
        errors.append(
            f"{summary_path}: graded multi-threshold evaluation contains "
            f"{failures} failed measured trial(s)"
        )
    warmup_failures = sum(row.get("trial_status") == "failure" for row in warmups)
    if summary.get("warmup_failure_count") != warmup_failures:
        errors.append(f"{summary_path}: warmup_failure_count is inconsistent")
    if warmup_failures:
        errors.append(
            f"{summary_path}: graded multi-threshold evaluation contains "
            f"{warmup_failures} failed warmup trial(s)"
        )
    expected_failure_diagnostics = [
        {field: row.get(field) for field in SWEEP_FAILURE_DIAGNOSTIC_FIELDS}
        for row in [*warmups, *rows]
        if row.get("trial_status") == "failure"
    ]
    if summary.get("failure_diagnostics") != expected_failure_diagnostics:
        errors.append(
            f"{summary_path}: failure_diagnostics does not recompute from trial rows"
        )
    if "observed maximum" not in str(summary.get("latency_qualification", "")).lower():
        errors.append(f"{summary_path}: latency maximum is not properly qualified")
    return errors


def _grad_round_counters(rounds: Sequence[Mapping[str, Any]]) -> dict[str, int | bool]:
    """Independently derive every published Part 5 counter from round rows."""

    def count(field: str) -> int:
        return sum(
            int(round_.get(field, 0))
            for round_ in rounds
            if isinstance(round_.get(field, 0), int)
            and not isinstance(round_.get(field, 0), bool)
        )

    def completed(round_: Mapping[str, Any]) -> bool:
        return round_.get("sent") is True and round_.get("received") is True

    by_step = {str(round_.get("step")): round_ for round_ in rounds}
    security_window = by_step.get("security_window_denied_context", {})
    post_expiry = by_step.get("denied_after_expiry", {})
    reissue = by_step.get("reissue_allowed", {})
    return {
        "key_request_attempts": count("key_request_attempts"),
        "successful_key_requests": count("successful_key_requests"),
        "key_reuses": sum(round_.get("key_reused") is True for round_ in rounds),
        "transfer_attempts": sum(
            round_.get("transfer_attempted") is True for round_ in rounds
        ),
        "successful_transfers": sum(completed(round_) for round_ in rounds),
        "allowed_while_valid": sum(
            str(round_.get("step")) in set(GRAD_STEPS[:3]) and completed(round_)
            for round_ in rounds
        ),
        "security_window_sends": int(completed(security_window)),
        "denied_after_expiry": int(
            post_expiry.get("authorized") is False
            and post_expiry.get("authorization_status") == "auth_rejected_request"
            and post_expiry.get("sent") is False
            and post_expiry.get("received") is False
            and post_expiry.get("transfer_attempted") is False
        ),
        "reissued_after_expiry": bool(
            completed(reissue) and reissue.get("key_reused") is False
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_grad(directory: Path) -> list[str]:
    errors: list[str] = []
    jsonl_path = directory / "live_trials.jsonl"
    csv_path = directory / "trials.csv"
    summary_path = directory / "summary.json"
    plot_path = directory / "key_validity.png"
    for path in (jsonl_path, csv_path, summary_path, plot_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{path}: missing or empty required graduate artifact")
    if errors:
        return errors
    try:
        trials = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{jsonl_path}: {exc}"]
    if len(trials) != len(GRAD_PERIODS) or not all(
        isinstance(trial, dict) for trial in trials
    ):
        errors.append(f"{jsonl_path}: expected exactly three live trial objects")
        return errors

    observed_periods = [
        (str(trial.get("label")), trial.get("relative_validity_ms"))
        for trial in trials
    ]
    if observed_periods != list(GRAD_PERIODS):
        errors.append(
            f"{jsonl_path}: expected exact ordered 5/20/40-second periods; "
            f"observed {observed_periods}"
        )

    summary = _load_object(summary_path, errors)
    if summary is None:
        return errors
    if summary.get("trials") != trials:
        errors.append(f"{summary_path}: trials do not match live_trials.jsonl")
    if summary.get("relative_validity_ms") != [period[1] for period in GRAD_PERIODS]:
        errors.append(f"{summary_path}: relative_validity_ms is not exact 5/20/40 seconds")
    if not str(summary.get("authorization_mode", "")).startswith(LIVE_MODE_PREFIX):
        errors.append(f"{summary_path}: graduate experiment is not labeled live SST")
    expected_flags = all(trial.get("expected") is True for trial in trials)
    if summary.get("all_trials_expected") is not expected_flags or not expected_flags:
        errors.append(f"{summary_path}: one or more graduate trials failed")

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            csv_fields = reader.fieldnames or []
    except OSError as exc:
        errors.append(f"{csv_path}: {exc}")
        csv_rows = []
        csv_fields = []
    missing_csv_fields = sorted(set(GRAD_CSV_FIELDS) - set(csv_fields))
    if missing_csv_fields:
        errors.append(f"{csv_path}: missing required columns {missing_csv_fields}")
    if len(csv_rows) != len(trials):
        errors.append(f"{csv_path}: row count does not match graduate trials")
    else:
        for index, (row, trial) in enumerate(zip(csv_rows, trials, strict=True), start=2):
            for field, value in row.items():
                if value != _csv_value(trial.get(field)):
                    errors.append(f"{csv_path}:{index}: {field} disagrees with JSONL")

    recomputed_trials: list[dict[str, int | bool]] = []
    for index, (trial, expected_period) in enumerate(
        zip(trials, GRAD_PERIODS, strict=True)
    ):
        label = f"{jsonl_path}:trial[{index}]"
        if (trial.get("label"), trial.get("relative_validity_ms")) != expected_period:
            errors.append(f"{label}: label and relative validity do not match")
        if trial.get("expected") is not True:
            errors.append(f"{label}: expected sequence did not complete")
        if not str(trial.get("authorization_mode", "")).startswith(LIVE_MODE_PREFIX):
            errors.append(f"{label}: trial is not live SST")

        for field in ("first_use_ms", "context_change_ms", "effective_expiry_ms"):
            if not _nonnegative(trial.get(field)):
                errors.append(f"{label}: {field} is missing or invalid")
        absolute = trial.get("absolute_validity_ms")
        if absolute is not None and not _nonnegative(absolute):
            errors.append(f"{label}: absolute_validity_ms is invalid")
        first_use = trial.get("first_use_ms")
        relative = trial.get("relative_validity_ms")
        context_change = trial.get("context_change_ms")
        if (
            _nonnegative(first_use)
            and _nonnegative(relative)
            and _nonnegative(context_change)
        ):
            expiry_candidates = [int(first_use) + int(relative)]
            if _nonnegative(absolute):
                expiry_candidates.append(int(absolute))
            effective_expiry = min(expiry_candidates)
            remaining = max(0, effective_expiry - int(context_change))
            if trial.get("effective_expiry_ms") != effective_expiry:
                errors.append(f"{label}: effective_expiry_ms does not recompute")
            if trial.get("remaining_validity_ms_at_context_change") != remaining:
                errors.append(
                    f"{label}: remaining validity at context change does not recompute"
                )

        rounds = trial.get("rounds")
        if not isinstance(rounds, list) or not all(
            isinstance(round_, Mapping) for round_ in rounds
        ):
            errors.append(f"{label}: controlled rounds are missing or malformed")
            continue
        if [round_.get("step") for round_ in rounds] != list(GRAD_STEPS):
            errors.append(f"{label}: controlled round sequence is invalid")
            continue
        for round_ in rounds:
            step = str(round_["step"])
            for field, expected_value in GRAD_ROUND_EXPECTATIONS[step].items():
                if round_.get(field) != expected_value:
                    if step == "denied_after_expiry" and field in {
                        "authorized",
                        "authorization_status",
                        "sent",
                        "received",
                        "transfer_attempted",
                    }:
                        errors.append(
                            f"{label}: post-expiry policy denial must be a "
                            "controlled Auth rejection with no transfer"
                        )
                    errors.append(
                        f"{label}: {step}.{field} violates the controlled sequence"
                    )
            for field in (
                "authorization_latency_ms",
                "handshake_latency_ms",
                "send_latency_ms",
                "total_transport_latency_ms",
            ):
                if not _nonnegative(round_.get(field)):
                    errors.append(f"{label}: {step}.{field} is missing or invalid")

        counters = _grad_round_counters(rounds)
        recomputed_trials.append(counters)
        for field, value in counters.items():
            if trial.get(field) != value:
                errors.append(f"{label}: {field} does not recompute from rounds")
        if trial.get("key_requests") != counters["successful_key_requests"]:
            errors.append(f"{label}: key_requests alias is inconsistent")
        if counters != GRAD_COUNTERS:
            errors.append(f"{label}: round-derived counters violate the baseline")

        completed = [
            round_
            for round_ in rounds
            if round_.get("sent") is True and round_.get("received") is True
        ]
        sent = [round_ for round_ in rounds if round_.get("sent") is True]
        # Missing or malformed per-round timings have already been reported
        # above.  Keep validation fail-closed without allowing an invalid
        # artifact to crash the validator with KeyError/TypeError.
        timings_valid = all(
            _nonnegative(round_.get(field))
            for round_ in rounds
            for field in (
                "authorization_latency_ms",
                "handshake_latency_ms",
                "send_latency_ms",
                "total_transport_latency_ms",
            )
        )
        latency_expectations: dict[str, float] = {}
        if timings_valid:
            latency_expectations = {
                "average_authorization_latency_ms": _mean(
                    [float(round_["authorization_latency_ms"]) for round_ in rounds]
                ),
                "average_handshake_latency_ms": _mean(
                    [float(round_["handshake_latency_ms"]) for round_ in sent]
                ),
                "average_send_latency_ms": _mean(
                    [float(round_["send_latency_ms"]) for round_ in completed]
                ),
                "average_protected_transfer_latency_ms": _mean(
                    [
                        float(round_["handshake_latency_ms"])
                        + float(round_["send_latency_ms"])
                        for round_ in completed
                    ]
                ),
            }
        for field, expected_value in latency_expectations.items():
            actual = trial.get(field)
            if not _finite_number(actual) or not math.isclose(
                float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append(f"{label}: {field} does not recompute from rounds")

    expected_totals = {
        field: sum(int(counters[field]) for counters in recomputed_trials)
        for field in GRAD_TOTAL_FIELDS
    }
    if summary.get("totals") != expected_totals:
        errors.append(f"{summary_path}: totals do not recompute from round rows")
    return errors


def validate_result_tree(root: Path, *, require_grad: bool) -> tuple[list[str], int]:
    errors: list[str] = []
    validated = 0
    for scenario in COMMON_SCENARIOS:
        path = root / scenario / "action_record.json"
        if not path.is_file():
            errors.append(f"{path}: missing required common scenario record")
            continue
        payload = _load_object(path, errors)
        if payload is not None:
            errors.extend(validate_record(path, payload))
            errors.extend(validate_common_scenario(path, scenario, payload))
            validated += 1
    errors.extend(validate_capture_summary(root / "inspect-actions" / "response_summary.json"))
    errors.extend(validate_sweep(root / "threshold-sweep"))
    validated += 2
    grad = root / "grad-key-validity"
    if require_grad or grad.exists():
        errors.extend(validate_grad(grad))
        validated += 1
    return errors, validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Results root, scenario directories, or action-record JSON files.",
    )
    parser.add_argument(
        "--require-grad",
        action="store_true",
        help="Require and validate all CSE 598 key-validity artifacts.",
    )
    args = parser.parse_args(argv)
    search = args.paths or [Path("results")]
    errors: list[str] = []
    validated = 0
    focused: list[Path] = []
    for path in search:
        if path.is_dir() and (
            path.name == "results"
            or (path / "inspect-actions").exists()
            or (path / "threshold-sweep").exists()
        ):
            tree_errors, count = validate_result_tree(path, require_grad=args.require_grad)
            errors.extend(tree_errors)
            validated += count
        else:
            focused.append(path)
    if focused:
        records = discover(focused)
        if not records:
            errors.append("no action records found")
        for path in records:
            payload = _load_object(path, errors)
            if payload is not None:
                errors.extend(validate_record(path, payload))
                validated += 1
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated {validated} result artifact group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

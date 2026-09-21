"""Load and summarize a captured pi0-FAST policy response.

The captured record is real model output recorded once from an officially
supported pi0-FAST configuration and committed so every student can inspect
the decoded response without a GPU or a checkpoint download.  The pinned
OpenPI response API does not expose generated token IDs; that limitation is
preserved in the record rather than filling in inferred or synthetic tokens.

A captured record is never a substitute for the deterministic ALOHA fixtures.
The fixtures drive the physical motion analysis because they are 14-dimensional
ALOHA commands with known physical limits. The captured record shows what a
real pi0-FAST response contains: whether tokens are exposed, how many there
are, the decoded model-space shape, the final action shape for its own
embodiment, and the measured inference and decode latency.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CAPTURED_DIR = Path("data/captured")
RECORD_NAME = "pi0_fast_action_record.json"
CHUNK_NAME = "pi0_fast_action_chunk.npz"
SOURCE_LABEL = "captured_live_pi0_fast"
PINNED_OPENPI_URL = "https://github.com/Physical-Intelligence/openpi.git"
PINNED_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CAPTURE_CONFIG_NAME = "pi0_fast_droid"
CAPTURE_CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi0_fast_droid"
CAPTURE_INPUT_ORIGIN = "openpi.policies.droid_policy.make_droid_example"
CAPTURE_ACTION_SHAPE = [10, 8]
FAST_TOKENIZER_REPOSITORY = "https://huggingface.co/physical-intelligence/fast"
FAST_TOKENIZER_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
FAST_TOKENIZER_CODE_SHA256 = "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20"

REQUIRED_FIELDS = (
    "source_type",
    "openpi_url",
    "openpi_commit",
    "model_family",
    "model_configuration",
    "checkpoint_uri",
    "fast_tokenizer_repository",
    "fast_tokenizer_revision",
    "fast_tokenizer_code_sha256",
    "embodiment",
    "task_prompt",
    "input_origin",
    "input_description",
    "input_seed",
    "input_shapes",
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
    "decoded_model_action_shape_source",
    "final_action_shape",
    "response_action_dtype",
    "configured_action_horizon",
    "returned_action_horizon",
    "actions_executed",
    "executed_horizon",
    "inference_ms",
    "end_to_end_infer_ms",
    "decode_ms",
    "response_keys",
    "capture_date",
    "capture_timestamp_utc",
    "capture_environment",
    "action_chunk_sha256",
    "limitation",
)


class CapturedRecordError(RuntimeError):
    """Raised when a captured record is missing, malformed, or mislabeled."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CapturedResponse:
    """A validated captured pi0-FAST response and its provenance."""

    metadata: Mapping[str, Any]
    action_chunk: np.ndarray
    record_path: Path
    chunk_path: Path

    @property
    def token_count(self) -> int | None:
        value = self.metadata.get("token_count")
        return int(value) if isinstance(value, int) else None

    @property
    def tokens_available(self) -> bool:
        return bool(self.metadata.get("tokens_available"))

    @property
    def executed_horizon(self) -> int:
        return int(self.metadata["executed_horizon"])

    def executed_prefix(self) -> np.ndarray:
        return np.array(self.action_chunk[: self.executed_horizon], copy=True)

    def summary(self) -> dict[str, Any]:
        """The concise, secret-free summary students report in Part 1."""

        return {
            "source_type": self.metadata["source_type"],
            "model_family": self.metadata["model_family"],
            "model_configuration": self.metadata["model_configuration"],
            "checkpoint_uri": self.metadata["checkpoint_uri"],
            "fast_tokenizer_repository": self.metadata["fast_tokenizer_repository"],
            "fast_tokenizer_revision": self.metadata["fast_tokenizer_revision"],
            "fast_tokenizer_code_sha256": self.metadata["fast_tokenizer_code_sha256"],
            "openpi_url": self.metadata["openpi_url"],
            "openpi_commit": self.metadata["openpi_commit"],
            "embodiment": self.metadata["embodiment"],
            "task_prompt": self.metadata["task_prompt"],
            "input_origin": self.metadata["input_origin"],
            "input_description": self.metadata["input_description"],
            "input_seed": self.metadata["input_seed"],
            "input_shapes": self.metadata["input_shapes"],
            "input_observation_sha256": self.metadata["input_observation_sha256"],
            "tokens_available": self.tokens_available,
            "token_count": self.token_count,
            "token_availability_reason": self.metadata["token_availability_reason"],
            "decode_ok": bool(self.metadata["decode_ok"]),
            "decode_boundary_status": self.metadata["decode_boundary_status"],
            "action_marker_present": self.metadata["action_marker_present"],
            "decoded_action_shape_before_unnormalization": list(
                self.metadata["decoded_action_shape_before_unnormalization"]
            ),
            "decode_observer": self.metadata["decode_observer"],
            "decode_validation": self.metadata["decode_validation"],
            "decoded_model_action_shape": list(self.metadata["decoded_model_action_shape"]),
            "decoded_model_action_shape_source": self.metadata["decoded_model_action_shape_source"],
            "final_action_shape": list(self.metadata["final_action_shape"]),
            "loaded_action_shape": list(self.action_chunk.shape),
            "response_action_dtype": self.metadata["response_action_dtype"],
            "configured_action_horizon": self.metadata["configured_action_horizon"],
            "returned_action_horizon": self.metadata["returned_action_horizon"],
            "actions_executed": self.metadata["actions_executed"],
            "executed_horizon": self.executed_horizon,
            "inference_ms": self.metadata.get("inference_ms"),
            "end_to_end_infer_ms": self.metadata.get("end_to_end_infer_ms"),
            "decode_ms": self.metadata.get("decode_ms"),
            "decode_timing_reason": self.metadata.get("decode_timing_reason"),
            "response_keys": list(self.metadata["response_keys"]),
            "capture_date": self.metadata["capture_date"],
            "capture_timestamp_utc": self.metadata["capture_timestamp_utc"],
            "capture_environment": self.metadata["capture_environment"],
            "capture_job_id": self.metadata.get("capture_job_id"),
            "action_chunk_sha256": self.metadata["action_chunk_sha256"],
            "limitation": self.metadata["limitation"],
        }


def captured_paths(root: Path) -> tuple[Path, Path]:
    return root / CAPTURED_DIR / RECORD_NAME, root / CAPTURED_DIR / CHUNK_NAME


def captured_record_available(root: Path) -> bool:
    record_path, chunk_path = captured_paths(root)
    return record_path.is_file() and chunk_path.is_file()


def configuration_report(root: Path) -> dict[str, Any]:
    """Describe the pi0-FAST response contract when no capture is available.

    This reports observed configuration facts and states plainly that no live
    or captured model response is present, so token availability is recorded as
    unavailable with a reason. It never invents token identifiers or actions.
    """

    from . import openpi_adapter as adapter

    model_config = json.loads(
        (root / "configs" / "model" / "pi0_fast_aloha_sim.json").read_text(encoding="utf-8")
    )
    return {
        "action_source": "model_configuration_only",
        "response_available": False,
        "model_family": adapter.MODEL_FAMILY,
        "model_configuration": adapter.COURSE_CONFIG_NAME,
        "checkpoint_uri": adapter.CHECKPOINT_URI,
        "openpi_commit": model_config["openpi_commit"],
        "tokens_available": False,
        "token_count": None,
        "token_availability_reason": (
            "No live or captured pi0-FAST response is present in this deployment. "
            "Upstream OpenPI policy responses return state, actions, and timing "
            "only, so action tokens are visible exclusively through the course "
            "adapter's observable FAST tokenizer during a live run. No official "
            "pi0-FAST ALOHA simulator checkpoint is published, so that live run "
            "is not available here."
        ),
        "decode_ok": None,
        "decoded_model_action_shape": [
            adapter.PREDICTED_ACTION_HORIZON,
            adapter.MODEL_ACTION_DIMENSION,
        ],
        "final_action_shape": [
            adapter.PREDICTED_ACTION_HORIZON,
            adapter.SIMULATOR_ACTION_DIMENSION,
        ],
        "configured_action_horizon": adapter.PREDICTED_ACTION_HORIZON,
        "returned_action_horizon": None,
        "configured_execution_horizon": adapter.DEFAULT_EXECUTED_ACTION_HORIZON,
        "actions_executed": False,
        "executed_horizon": 0,
        "action_chunk_sha256": None,
        "checkpoint_compatibility": adapter.CHECKPOINT_COMPATIBILITY,
        "limitation": (
            "The decoded and final shapes above are the configured contract, not "
            "a measurement. The physical motion analysis in this project runs on "
            "the committed 14-dimensional ALOHA fixtures, whose provenance is "
            "recorded in data/fixtures/*.json."
        ),
    }


def inspect_response(root: Path, *, configuration_only: bool = False) -> dict[str, Any]:
    """Return a validated response, or an explicitly requested config diagnostic.

    Missing or corrupt capture files are errors on the normal Part 1 path.  This
    prevents configuration-only dimensions from silently passing as inspection
    of real model output.
    """

    if configuration_only:
        return configuration_report(root)
    summary = load_captured_response(root).summary()
    summary["action_source"] = SOURCE_LABEL
    summary["response_available"] = True
    return summary


def _validated_shape(metadata: Mapping[str, Any], field: str) -> list[int]:
    value = metadata[field]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise CapturedRecordError(f"captured {field} must be two positive integers")
    return value


def _validate_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapturedRecordError(f"captured {field} must be a lowercase SHA-256 digest")
    return value


def load_captured_response(root: Path) -> CapturedResponse:
    """Load the committed captured record and verify its integrity."""

    record_path, chunk_path = captured_paths(root)
    if not record_path.is_file():
        raise CapturedRecordError(
            f"captured pi0-FAST record is missing: {record_path}. "
            "See data/captured/README.md for how it is produced."
        )
    try:
        metadata = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapturedRecordError(f"captured record is malformed: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise CapturedRecordError("captured record must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise CapturedRecordError(f"captured record is missing fields: {', '.join(missing)}")
    if metadata["source_type"] != SOURCE_LABEL:
        raise CapturedRecordError(
            f"captured record source_type must be {SOURCE_LABEL!r}; a synthetic fixture "
            "must never be labeled as captured model output"
        )
    if metadata["model_family"] != "pi0-FAST":
        raise CapturedRecordError("captured model_family must be 'pi0-FAST'")
    expected_provenance = {
        "openpi_url": PINNED_OPENPI_URL,
        "openpi_commit": PINNED_OPENPI_COMMIT,
        "model_configuration": CAPTURE_CONFIG_NAME,
        "checkpoint_uri": CAPTURE_CHECKPOINT_URI,
        "input_origin": CAPTURE_INPUT_ORIGIN,
        "fast_tokenizer_repository": FAST_TOKENIZER_REPOSITORY,
        "fast_tokenizer_revision": FAST_TOKENIZER_REVISION,
        "fast_tokenizer_code_sha256": FAST_TOKENIZER_CODE_SHA256,
    }
    for field, expected in expected_provenance.items():
        if metadata[field] != expected:
            raise CapturedRecordError(
                f"captured {field} must match the verified Part 1 provenance {expected!r}"
            )
    if metadata["tokens_available"] is not False or metadata["token_count"] is not None:
        raise CapturedRecordError(
            "captured token fields must state that the pinned response API exposes no token IDs"
        )
    if (
        not isinstance(metadata["token_availability_reason"], str)
        or not metadata["token_availability_reason"].strip()
    ):
        raise CapturedRecordError("captured token availability requires a reason")
    if metadata["decode_ok"] is not True:
        raise CapturedRecordError("captured response did not pass decode validation")
    if (
        metadata["decode_boundary_status"] != "verified_action_payload"
        or metadata["action_marker_present"] is not True
    ):
        raise CapturedRecordError("captured response lacks positive FAST decode-boundary evidence")
    if metadata["actions_executed"] is not False:
        raise CapturedRecordError("offline captured actions must not claim execution")

    decoded_shape = _validated_shape(metadata, "decoded_model_action_shape")
    boundary_shape = _validated_shape(metadata, "decoded_action_shape_before_unnormalization")
    final_shape = _validated_shape(metadata, "final_action_shape")
    if (
        decoded_shape != CAPTURE_ACTION_SHAPE
        or boundary_shape != CAPTURE_ACTION_SHAPE
        or final_shape != CAPTURE_ACTION_SHAPE
    ):
        raise CapturedRecordError(
            "captured action shapes do not match the pinned pi0-FAST-DROID response contract"
        )
    configured_horizon = metadata["configured_action_horizon"]
    returned_horizon = metadata["returned_action_horizon"]
    executed_horizon = metadata["executed_horizon"]
    for field, value in (
        ("configured_action_horizon", configured_horizon),
        ("returned_action_horizon", returned_horizon),
        ("executed_horizon", executed_horizon),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapturedRecordError(f"captured {field} must be a non-negative integer")
    if configured_horizon <= 0 or returned_horizon <= 0:
        raise CapturedRecordError("captured configured and returned horizons must be positive")
    if decoded_shape[0] != configured_horizon:
        raise CapturedRecordError(
            "captured configured_action_horizon does not match decoded_model_action_shape"
        )
    if final_shape[0] != returned_horizon:
        raise CapturedRecordError(
            "captured returned_action_horizon does not match final_action_shape"
        )
    if executed_horizon != 0:
        raise CapturedRecordError("offline captured response must have executed_horizon 0")
    response_keys = metadata["response_keys"]
    if not isinstance(response_keys, list) or not {"actions", "policy_timing"}.issubset(
        response_keys
    ):
        raise CapturedRecordError("captured response_keys do not describe a policy response")

    _validate_sha256(metadata["input_observation_sha256"], "input_observation_sha256")
    recorded_digest = _validate_sha256(metadata["action_chunk_sha256"], "action_chunk_sha256")

    if not chunk_path.is_file():
        raise CapturedRecordError(f"captured action chunk is missing: {chunk_path}")
    if sha256_file(chunk_path) != recorded_digest:
        raise CapturedRecordError("captured action chunk does not match its recorded SHA-256")

    try:
        with np.load(chunk_path, allow_pickle=False) as archive:
            if archive.files != ["action_chunk"]:
                raise CapturedRecordError(
                    "captured NPZ must contain exactly one action_chunk array"
                )
            chunk = np.array(archive["action_chunk"], dtype=np.float64)
    except CapturedRecordError:
        raise
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CapturedRecordError(f"captured action chunk is malformed: {exc}") from exc
    if chunk.ndim != 2 or 0 in chunk.shape:
        raise CapturedRecordError("captured action chunk must have non-empty shape [H, D]")
    if not np.all(np.isfinite(chunk)):
        raise CapturedRecordError("captured action chunk contains NaN or infinity")
    if not np.any(chunk):
        raise CapturedRecordError("captured final action chunk is unexpectedly all zero")
    if list(chunk.shape) != final_shape:
        raise CapturedRecordError(
            "captured action chunk shape does not match the recorded final_action_shape"
        )
    return CapturedResponse(
        metadata=dict(metadata),
        action_chunk=chunk,
        record_path=record_path,
        chunk_path=chunk_path,
    )

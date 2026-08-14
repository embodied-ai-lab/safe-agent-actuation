"""Load and summarize a captured pi0-FAST policy response.

The captured record is real model output recorded once from an officially
supported pi0-FAST configuration and committed so every student can inspect
token and decode behavior without a GPU or a checkpoint download.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CAPTURED_DIR = Path("data/captured")
RECORD_NAME = "pi0_fast_action_record.json"
CHUNK_NAME = "pi0_fast_action_chunk.npz"
SOURCE_LABEL = "captured_live_pi0_fast"

REQUIRED_FIELDS = (
    "source_type",
    "openpi_url",
    "openpi_commit",
    "model_family",
    "model_configuration",
    "checkpoint_uri",
    "embodiment",
    "task_prompt",
    "tokens_available",
    "decode_ok",
    "decoded_model_action_shape",
    "final_action_shape",
    "executed_horizon",
    "capture_date",
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
            "openpi_commit": self.metadata["openpi_commit"],
            "embodiment": self.metadata["embodiment"],
            "task_prompt": self.metadata["task_prompt"],
            "tokens_available": self.tokens_available,
            "token_count": self.token_count,
            "decode_ok": bool(self.metadata["decode_ok"]),
            "decoded_model_action_shape": list(self.metadata["decoded_model_action_shape"]),
            "final_action_shape": list(self.metadata["final_action_shape"]),
            "loaded_action_shape": list(self.action_chunk.shape),
            "executed_horizon": self.executed_horizon,
            "inference_ms": self.metadata.get("inference_ms"),
            "decode_ms": self.metadata.get("decode_ms"),
            "capture_date": self.metadata["capture_date"],
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
        "executed_horizon": adapter.DEFAULT_EXECUTED_ACTION_HORIZON,
        "checkpoint_compatibility": adapter.CHECKPOINT_COMPATIBILITY,
        "limitation": (
            "The decoded and final shapes above are the configured contract, not "
            "a measurement. The physical motion analysis in this project runs on "
            "the committed 14-dimensional ALOHA fixtures, whose provenance is "
            "recorded in data/fixtures/*.json."
        ),
    }


def inspect_response(root: Path) -> dict[str, Any]:
    """Return the best available pi0-FAST response summary, honestly labeled."""

    if captured_record_available(root):
        summary = load_captured_response(root).summary()
        summary["action_source"] = SOURCE_LABEL
        summary["response_available"] = True
        return summary
    return configuration_report(root)


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

    if not chunk_path.is_file():
        raise CapturedRecordError(f"captured action chunk is missing: {chunk_path}")
    recorded_digest = metadata.get("action_chunk_sha256")
    if recorded_digest and sha256_file(chunk_path) != recorded_digest:
        raise CapturedRecordError("captured action chunk does not match its recorded SHA-256")

    with np.load(chunk_path, allow_pickle=False) as archive:
        key = "action_chunk" if "action_chunk" in archive.files else archive.files[0]
        chunk = np.array(archive[key], dtype=np.float64)
    if chunk.ndim != 2 or 0 in chunk.shape:
        raise CapturedRecordError("captured action chunk must have non-empty shape [H, D]")
    if not np.all(np.isfinite(chunk)):
        raise CapturedRecordError("captured action chunk contains NaN or infinity")
    if list(chunk.shape) != list(metadata["final_action_shape"]):
        raise CapturedRecordError(
            "captured action chunk shape does not match the recorded final_action_shape"
        )
    horizon = int(metadata["executed_horizon"])
    if not 1 <= horizon <= chunk.shape[0]:
        raise CapturedRecordError("captured executed_horizon is outside the recorded chunk")
    return CapturedResponse(
        metadata=dict(metadata),
        action_chunk=chunk,
        record_path=record_path,
        chunk_path=chunk_path,
    )

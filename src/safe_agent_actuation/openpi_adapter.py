"""Course-owned OpenPI integration and pi0-FAST observability.

OpenPI remains an unmodified, pinned submodule.  This module creates the course
configuration at runtime and supplies the narrow adapter needed to preserve the
raw autoregressive output before OpenPI's FAST output transform consumes it.

The published ``pi0_fast_base`` checkpoint is a base checkpoint for fine-
tuning, not a validated ALOHA-simulator expert.  Its compatibility with the
course simulator is therefore deliberately and repeatedly marked *unresolved*.
No decode failure is converted into an action: an invalid response contains no
``actions`` field and has ``executable == False``.

OpenPI imports are intentionally local to factory functions.  Offline analysis
and unit tests can import this module without JAX, OpenPI, or a GPU.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, MutableMapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

COURSE_CONFIG_NAME = "pi0_fast_aloha_sim_course"
MODEL_FAMILY = "pi0-FAST"
CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi0_fast_base"
MODEL_ACTION_DIMENSION = 32
PREDICTED_ACTION_HORIZON = 32
SIMULATOR_ACTION_DIMENSION = 14
DEFAULT_EXECUTED_ACTION_HORIZON = 10
FAST_SKIP_TOKENS = 128

CHECKPOINT_COMPATIBILITY = (
    "UNRESOLVED: pi0_fast_base is a generic fine-tuning checkpoint, not a "
    "validated pi0-FAST ALOHA-simulator expert checkpoint"
)

# A fake or alternate policy may return a capture through this private field.
# The observable wrapper consumes it and never forwards it over the wire.
_CAPTURE_FIELD = "_safe_agent_actuation_fast_capture"
_ACTIVE_CAPTURE: ContextVar[MutableMapping[str, Any] | None] = ContextVar(
    "safe_agent_actuation_fast_capture", default=None
)


class FastDecodeError(ValueError):
    """Raised internally when FAST output cannot become a valid action chunk."""


class InvalidPolicyResponse(ValueError):
    """Raised when a policy response is not safe to pass toward a monitor."""


class PolicyLike(Protocol):
    """Dependency-light policy protocol used by the recorder and tests."""

    def infer(self, observation: Any, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RecordedPolicyResponse:
    """A validated client-side policy record.

    ``action_chunk`` is the final, physical ALOHA command returned after
    OpenPI's unnormalization and ALOHA output transforms.  It is not the
    normalized model-space decoder output.
    """

    record_index: int
    executable: bool
    action_chunk: NDArray[np.float64] | None
    observability: Mapping[str, Any]
    response: Mapping[str, Any]
    action_path: Path | None = None
    summary_path: Path | None = None


def _integer_token_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    """Validate a one-dimensional token sequence without silently truncating."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FastDecodeError(f"{field_name} must be a one-dimensional integer sequence") from exc
    if array.ndim != 1:
        raise FastDecodeError(f"{field_name} must have shape [T]")
    if array.size == 0:
        raise FastDecodeError(f"{field_name} must not be empty")
    if array.dtype.kind not in "iu":
        # A sequence such as [1.0] is not accepted as a token stream even if its
        # numerical value happens to be integral.
        raise FastDecodeError(f"{field_name} must contain integer token IDs")
    if np.any(array < 0):
        raise FastDecodeError(f"{field_name} must contain non-negative token IDs")
    return tuple(int(value) for value in array.tolist())


def extract_fast_action_tokens(
    raw_paligemma_token_ids: Sequence[int] | NDArray[np.integer[Any]],
    paligemma_tokenizer: Any,
    *,
    fast_skip_tokens: int = FAST_SKIP_TOKENS,
) -> tuple[int, ...]:
    """Extract the actual FAST payload using OpenPI's encoding convention.

    This deliberately does **not** subtract a vocabulary offset from every
    generated token.  The sequence also contains formatting, EOS, and padding
    tokens.  We first decode the generated text, select exactly the payload in
    ``"Action: <payload>|"``, re-encode that payload, and then apply the same
    involution as ``openpi.models.tokenizer.FASTTokenizer``.
    """

    raw = _integer_token_tuple(
        raw_paligemma_token_ids, field_name="raw_paligemma_token_ids"
    )
    if not isinstance(fast_skip_tokens, int) or fast_skip_tokens < 0:
        raise FastDecodeError("fast_skip_tokens must be a non-negative integer")

    try:
        decoded = paligemma_tokenizer.decode(list(raw))
    except Exception as exc:  # tokenizer implementations raise several exception types
        raise FastDecodeError("PaliGemma output could not be decoded") from exc
    if not isinstance(decoded, str):
        raise FastDecodeError("PaliGemma tokenizer returned non-text output")
    marker = "Action: "
    if marker not in decoded:
        raise FastDecodeError("generated output is missing the 'Action: ' marker")
    suffix = decoded.split(marker, 1)[1]
    if "|" not in suffix:
        raise FastDecodeError("generated action payload is missing the '|' terminator")
    payload = suffix.split("|", 1)[0].strip()
    if not payload:
        raise FastDecodeError("generated action payload is empty")

    try:
        encoded_payload = np.asarray(paligemma_tokenizer.encode(payload))
        vocab_size = int(paligemma_tokenizer.vocab_size())
    except Exception as exc:
        raise FastDecodeError("generated action payload could not be re-encoded") from exc
    if encoded_payload.ndim != 1 or encoded_payload.size == 0:
        raise FastDecodeError("generated action payload did not encode to tokens")
    if encoded_payload.dtype.kind not in "iu":
        raise FastDecodeError("encoded action payload contains non-integer token IDs")
    if vocab_size <= fast_skip_tokens:
        raise FastDecodeError("invalid PaliGemma vocabulary size")

    fast_ids = vocab_size - 1 - fast_skip_tokens - encoded_payload.astype(np.int64)
    if np.any(fast_ids < 0) or np.any(fast_ids >= vocab_size - fast_skip_tokens):
        raise FastDecodeError("action payload contains tokens outside the FAST token region")
    return tuple(int(value) for value in fast_ids.tolist())


def _strict_decoded_chunk(
    value: object,
    *,
    expected_shape: tuple[int, int],
    field_name: str,
) -> NDArray[np.float64]:
    try:
        chunk = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FastDecodeError(f"{field_name} must be a numeric action chunk") from exc
    if chunk.shape != expected_shape:
        raise FastDecodeError(
            f"{field_name} has shape {chunk.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(chunk)):
        raise FastDecodeError(f"{field_name} contains NaN or infinity")
    if not np.any(chunk):
        raise FastDecodeError(f"{field_name} is an all-zero decode fallback")
    return np.array(chunk, dtype=np.float64, copy=True)


def _capture_error(capture: MutableMapping[str, Any], exc: BaseException) -> None:
    capture["decode_ok"] = False
    capture["decode_error"] = str(exc) or exc.__class__.__name__
    capture["decoded_all_zero"] = "all-zero" in capture["decode_error"]


def _make_observable_tokenizer_class() -> type:
    """Build an OpenPI FASTTokenizer subclass after OpenPI is importable."""

    from openpi.models import tokenizer as openpi_tokenizer  # type: ignore[import-not-found]

    class ObservableFASTTokenizer(openpi_tokenizer.FASTTokenizer):
        """Strict FAST decoder that publishes one context-local observation."""

        def extract_actions(
            self, tokens: NDArray[np.integer[Any]], action_horizon: int, action_dim: int
        ) -> NDArray[np.floating[Any]]:
            capture = _ACTIVE_CAPTURE.get()
            local_capture: MutableMapping[str, Any] = capture if capture is not None else {}
            started = time.perf_counter()
            try:
                raw = _integer_token_tuple(tokens, field_name="raw_paligemma_token_ids")
                local_capture["raw_paligemma_token_ids"] = raw
                fast_ids = extract_fast_action_tokens(
                    raw,
                    self._paligemma_tokenizer,  # noqa: SLF001 - required external observation seam
                    fast_skip_tokens=int(self._fast_skip_tokens),  # noqa: SLF001
                )
                local_capture["fast_action_token_ids"] = fast_ids
                decoded = super().extract_actions(
                    np.asarray(raw, dtype=np.int32), action_horizon, action_dim
                )
                checked = _strict_decoded_chunk(
                    decoded,
                    expected_shape=(int(action_horizon), int(action_dim)),
                    field_name="decoded_model_actions",
                )
                local_capture.update(
                    {
                        "decode_ok": True,
                        "decode_error": None,
                        "decoded_all_zero": False,
                        "decoded_model_action_shape": tuple(checked.shape),
                    }
                )
                return checked.astype(np.float32, copy=False)
            except FastDecodeError as exc:
                _capture_error(local_capture, exc)
                raise
            except Exception as exc:
                wrapped = FastDecodeError(f"FAST decoder failed: {exc}")
                _capture_error(local_capture, wrapped)
                raise wrapped from exc
            finally:
                local_capture["decode_ms"] = (time.perf_counter() - started) * 1000.0

    ObservableFASTTokenizer.__name__ = "CourseObservableFASTTokenizer"
    ObservableFASTTokenizer.__qualname__ = "CourseObservableFASTTokenizer"
    return ObservableFASTTokenizer


def build_course_train_config(
    checkpoint_uri: str = CHECKPOINT_URI,
    *,
    tokenizer_class: type | None = None,
) -> Any:
    """Build ``pi0_fast_aloha_sim_course`` from upstream's ALOHA-sim config.

    The returned object is an OpenPI ``TrainConfig``.  It is constructed rather
    than registered in ``third_party/openpi``, keeping the submodule clean.
    Constructing this config does not download the checkpoint; policy creation
    does.  The base checkpoint's live ALOHA compatibility remains unresolved.
    """

    if not isinstance(checkpoint_uri, str) or not checkpoint_uri.strip():
        raise ValueError("checkpoint_uri must be a non-empty string")

    # Lazy imports preserve dependency-light offline tests.
    from openpi.models import pi0_fast  # type: ignore[import-not-found]
    from openpi.training import config as openpi_config  # type: ignore[import-not-found]
    from openpi.training import weight_loaders  # type: ignore[import-not-found]

    upstream = openpi_config.get_config("pi0_aloha_sim")
    observable_tokenizer = tokenizer_class or _make_observable_tokenizer_class()
    model = pi0_fast.Pi0FASTConfig(
        action_dim=MODEL_ACTION_DIMENSION,
        action_horizon=PREDICTED_ACTION_HORIZON,
        max_token_len=250,
        fast_model_tokenizer=observable_tokenizer,
    )
    data = replace(
        upstream.data,
        assets=openpi_config.AssetsConfig(
            assets_dir=f"{checkpoint_uri.rstrip('/')}/assets",
            asset_id="trossen",
        ),
        use_delta_joint_actions=False,
    )
    metadata = dict(upstream.policy_metadata or {})
    metadata.update(
        {
            "model_family": MODEL_FAMILY,
            "course_config": COURSE_CONFIG_NAME,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "predicted_action_horizon": PREDICTED_ACTION_HORIZON,
            "model_action_dimension": MODEL_ACTION_DIMENSION,
            "simulator_action_dimension": SIMULATOR_ACTION_DIMENSION,
        }
    )
    return replace(
        upstream,
        name=COURSE_CONFIG_NAME,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{checkpoint_uri.rstrip('/')}/params"
        ),
        policy_metadata=metadata,
    )


# A descriptive alias for callers that prefer a verb matching policy creation.
create_course_train_config = build_course_train_config


def _normalise_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    raw_value = capture.get("raw_paligemma_token_ids", ())
    fast_value = capture.get("fast_action_token_ids", ())
    try:
        raw = list(_integer_token_tuple(raw_value, field_name="raw_paligemma_token_ids"))
    except FastDecodeError:
        raw = []
    try:
        fast = list(_integer_token_tuple(fast_value, field_name="fast_action_token_ids"))
    except FastDecodeError:
        fast = []

    shape_value = capture.get("decoded_model_action_shape")
    try:
        model_shape = [int(value) for value in shape_value] if shape_value is not None else None
    except (TypeError, ValueError, OverflowError):
        model_shape = None
    decode_ms = capture.get("decode_ms")
    try:
        decode_ms_value = float(decode_ms) if decode_ms is not None else None
    except (TypeError, ValueError, OverflowError):
        decode_ms_value = None
    if decode_ms_value is not None and (
        not np.isfinite(decode_ms_value) or decode_ms_value < 0
    ):
        decode_ms_value = None

    return {
        "raw_paligemma_token_ids": raw,
        "raw_token_count": len(raw),
        "fast_action_token_ids": fast,
        "fast_token_count": len(fast),
        "decode_ok": capture.get("decode_ok") is True,
        "decode_error": capture.get("decode_error"),
        "decoded_all_zero": bool(capture.get("decoded_all_zero", False)),
        "decoded_model_action_shape": model_shape,
        "decode_ms": decode_ms_value,
    }


def _base_observability(
    capture: Mapping[str, Any],
    *,
    checkpoint_uri: str,
    executed_action_horizon: int,
    total_infer_ms: float,
) -> dict[str, Any]:
    metadata = _normalise_capture(capture)
    metadata.update(
        {
            "model_family": MODEL_FAMILY,
            "model_configuration": COURSE_CONFIG_NAME,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "model_action_dimension": MODEL_ACTION_DIMENSION,
            "simulator_action_dimension": SIMULATOR_ACTION_DIMENSION,
            "predicted_action_horizon": PREDICTED_ACTION_HORIZON,
            "executed_action_horizon": executed_action_horizon,
            "total_policy_adapter_ms": float(total_infer_ms),
        }
    )
    return metadata


class ObservableFastPolicy:
    """Wrap an OpenPI policy and return correlated token/decode/action data.

    The wrapper is also useful with fakes: a fake policy can provide a mapping
    under :data:`_CAPTURE_FIELD`.  Real policies publish that mapping through
    the context-local observable tokenizer created above.
    """

    def __init__(
        self,
        policy: PolicyLike,
        *,
        checkpoint_uri: str = CHECKPOINT_URI,
        executed_action_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    ) -> None:
        if not 1 <= int(executed_action_horizon) <= PREDICTED_ACTION_HORIZON:
            raise ValueError(
                f"executed_action_horizon must be between 1 and {PREDICTED_ACTION_HORIZON}"
            )
        self._policy = policy
        self.checkpoint_uri = checkpoint_uri
        self.executed_action_horizon = int(executed_action_horizon)

    @property
    def metadata(self) -> dict[str, Any]:
        underlying = getattr(self._policy, "metadata", {})
        metadata = dict(underlying if isinstance(underlying, Mapping) else {})
        metadata.update(
            {
                "model_family": MODEL_FAMILY,
                "course_config": COURSE_CONFIG_NAME,
                "checkpoint_uri": self.checkpoint_uri,
                "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
                "predicted_action_horizon": PREDICTED_ACTION_HORIZON,
                "executed_action_horizon": self.executed_action_horizon,
                "model_action_dimension": MODEL_ACTION_DIMENSION,
                "simulator_action_dimension": SIMULATOR_ACTION_DIMENSION,
            }
        )
        return metadata

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, observation: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        capture: MutableMapping[str, Any] = {}
        token = _ACTIVE_CAPTURE.set(capture)
        started = time.perf_counter()
        result: Mapping[str, Any] | None = None
        inference_error: BaseException | None = None
        try:
            candidate = self._policy.infer(dict(observation), **kwargs)
            if not isinstance(candidate, Mapping):
                raise InvalidPolicyResponse("wrapped policy returned a non-mapping response")
            result = candidate
            supplied_capture = candidate.get(_CAPTURE_FIELD)
            if isinstance(supplied_capture, Mapping):
                capture.update(supplied_capture)
        except Exception as exc:  # returned as a structured, non-executable response
            inference_error = exc
            if not capture.get("decode_error"):
                _capture_error(capture, exc)
        finally:
            total_ms = (time.perf_counter() - started) * 1000.0
            _ACTIVE_CAPTURE.reset(token)

        observability = _base_observability(
            capture,
            checkpoint_uri=self.checkpoint_uri,
            executed_action_horizon=self.executed_action_horizon,
            total_infer_ms=total_ms,
        )
        response: dict[str, Any] = {}
        if result is not None:
            response.update({key: value for key, value in result.items() if key != _CAPTURE_FIELD})

        try:
            if inference_error is not None:
                raise InvalidPolicyResponse(
                    str(inference_error) or inference_error.__class__.__name__
                )
            if not observability["decode_ok"]:
                raise InvalidPolicyResponse(
                    str(observability.get("decode_error") or "FAST decode status is missing")
                )
            if not observability["raw_paligemma_token_ids"]:
                raise InvalidPolicyResponse("raw PaliGemma token sequence is missing")
            if not observability["fast_action_token_ids"]:
                raise InvalidPolicyResponse("FAST action-token payload is missing")
            if observability["decoded_all_zero"]:
                raise InvalidPolicyResponse("FAST decoder returned an all-zero fallback")
            if observability["decoded_model_action_shape"] != [
                PREDICTED_ACTION_HORIZON,
                MODEL_ACTION_DIMENSION,
            ]:
                raise InvalidPolicyResponse(
                    "decoded model action shape is missing or malformed: expected [32, 32]"
                )
            physical = _strict_decoded_chunk(
                response.get("actions"),
                expected_shape=(PREDICTED_ACTION_HORIZON, SIMULATOR_ACTION_DIMENSION),
                field_name="decoded_physical_actions",
            )
        except (FastDecodeError, InvalidPolicyResponse) as exc:
            # It is essential that stale or fallback actions cannot survive this path.
            response.pop("actions", None)
            observability["decode_ok"] = False
            if not observability.get("decode_error"):
                observability["decode_error"] = str(exc)
            response["executable"] = False
            response["openpi_observability"] = observability
            return response

        response["actions"] = physical
        observability["decoded_physical_action_shape"] = list(physical.shape)
        response["executable"] = True
        response["openpi_observability"] = observability
        return response


# Alternative capitalization is convenient in teaching material that spells
# the family name as pi0-FAST.
ObservableFASTPolicy = ObservableFastPolicy


def create_observable_fast_policy(
    checkpoint_uri: str = CHECKPOINT_URI,
    *,
    executed_action_horizon: int = DEFAULT_EXECUTED_ACTION_HORIZON,
    default_prompt: str | None = None,
) -> ObservableFastPolicy:
    """Download/load the base checkpoint and return the strict outer policy.

    Calling this function can trigger a large first-run checkpoint download.
    It does not assert that ``pi0_fast_base`` is behaviorally compatible with
    ALOHA simulation. That unresolved compatibility limits this optional
    instructor probe and does not block the fixture-based graded release.
    """

    from openpi.policies import policy_config  # type: ignore[import-not-found]

    config = build_course_train_config(checkpoint_uri)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_uri,
        default_prompt=default_prompt,
    )
    return ObservableFastPolicy(
        policy,
        checkpoint_uri=checkpoint_uri,
        executed_action_horizon=executed_action_horizon,
    )


def require_executable_actions(
    response: Mapping[str, Any],
    *,
    executed_action_horizon: int | None = None,
) -> NDArray[np.float64]:
    """Return a defensive copy of a valid physical chunk or fail closed."""

    if not isinstance(response, Mapping):
        raise InvalidPolicyResponse("policy response must be a mapping")
    if response.get("executable") is not True:
        raise InvalidPolicyResponse("policy response is explicitly non-executable")
    metadata = response.get("openpi_observability")
    if not isinstance(metadata, Mapping) or metadata.get("decode_ok") is not True:
        raise InvalidPolicyResponse("policy response lacks a successful FAST decode record")
    if metadata.get("checkpoint_compatibility") != CHECKPOINT_COMPATIBILITY:
        raise InvalidPolicyResponse("policy response lacks the checkpoint compatibility disclosure")
    try:
        raw = _integer_token_tuple(
            metadata.get("raw_paligemma_token_ids"),
            field_name="raw_paligemma_token_ids",
        )
        fast = _integer_token_tuple(
            metadata.get("fast_action_token_ids"), field_name="fast_action_token_ids"
        )
    except FastDecodeError as exc:
        raise InvalidPolicyResponse(str(exc)) from exc
    if not raw or not fast:
        raise InvalidPolicyResponse("token observation is incomplete")
    if metadata.get("decoded_model_action_shape") != [
        PREDICTED_ACTION_HORIZON,
        MODEL_ACTION_DIMENSION,
    ]:
        raise InvalidPolicyResponse("decoded model action shape must be [32, 32]")
    if metadata.get("decoded_all_zero") is True:
        raise InvalidPolicyResponse("all-zero decode fallback is never executable")
    try:
        chunk = _strict_decoded_chunk(
            response.get("actions"),
            expected_shape=(PREDICTED_ACTION_HORIZON, SIMULATOR_ACTION_DIMENSION),
            field_name="decoded_physical_actions",
        )
    except FastDecodeError as exc:
        raise InvalidPolicyResponse(str(exc)) from exc

    if executed_action_horizon is None:
        return chunk
    horizon = int(executed_action_horizon)
    if not 1 <= horizon <= PREDICTED_ACTION_HORIZON:
        raise InvalidPolicyResponse(
            f"executed_action_horizon must be between 1 and {PREDICTED_ACTION_HORIZON}"
        )
    return np.array(chunk[:horizon], dtype=np.float64, copy=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class FastPolicyClientRecorder:
    """Record full policy responses before any action-chunk brokering.

    Large physical chunks are stored once in compressed NumPy files when a
    directory is configured.  JSON summaries contain shapes and paths, not a
    repeated action array.  Secret key material is neither expected nor logged.
    """

    def __init__(self, policy: PolicyLike, record_dir: str | Path | None = None) -> None:
        self._policy = policy
        self._record_dir = Path(record_dir) if record_dir is not None else None
        self.records: list[RecordedPolicyResponse] = []

    @property
    def metadata(self) -> Mapping[str, Any]:
        metadata = getattr(self._policy, "metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, observation: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        response = self._policy.infer(observation, **kwargs)
        self.record(response)
        return response

    def record(self, response: Mapping[str, Any]) -> RecordedPolicyResponse:
        if not isinstance(response, Mapping):
            raise InvalidPolicyResponse("policy response must be a mapping")
        metadata_value = response.get("openpi_observability")
        observability = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        executable = response.get("executable") is True
        action_chunk: NDArray[np.float64] | None
        try:
            action_chunk = require_executable_actions(response) if executable else None
        except InvalidPolicyResponse:
            # Never preserve an invalid action array as an executable client record.
            executable = False
            action_chunk = None

        index = len(self.records)
        action_path: Path | None = None
        summary_path: Path | None = None
        if self._record_dir is not None:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            if action_chunk is not None:
                action_path = self._record_dir / f"action_{index:06d}.npz"
                np.savez_compressed(action_path, actions=action_chunk)
            summary_path = self._record_dir / f"action_{index:06d}.json"
            summary = {
                "record_index": index,
                "executable": executable,
                "action_shape": list(action_chunk.shape) if action_chunk is not None else None,
                "action_path": action_path.name if action_path is not None else None,
                "observability": _json_safe(observability),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        record = RecordedPolicyResponse(
            record_index=index,
            executable=executable,
            action_chunk=action_chunk,
            observability=observability,
            response=dict(response),
            action_path=action_path,
            summary_path=summary_path,
        )
        self.records.append(record)
        return record


# Short alias used by the live client.
PolicyResponseRecorder = FastPolicyClientRecorder

from __future__ import annotations

import json

import numpy as np
import pytest

from safe_agent_actuation.openpi_adapter import (
    CHECKPOINT_COMPATIBILITY,
    CHECKPOINT_URI,
    FastDecodeError,
    FastPolicyClientRecorder,
    InvalidPolicyResponse,
    ObservableFastPolicy,
    extract_fast_action_tokens,
    require_executable_actions,
)

CAPTURE_FIELD = "_safe_agent_actuation_fast_capture"


class FakeSentencePiece:
    def decode(self, tokens: list[int]) -> str:
        assert tokens == [17, 900, 1, 0]
        return "prefix Action: compressed payload| ignored suffix"

    def encode(self, payload: str) -> list[int]:
        assert payload == "compressed payload"
        return [870, 865]

    def vocab_size(self) -> int:
        return 1000


def capture(*, decode_ok: bool = True, all_zero: bool = False) -> dict:
    return {
        "raw_paligemma_token_ids": [17, 900, 1, 0],
        "fast_action_token_ids": [3, 5, 8],
        "decode_ok": decode_ok,
        "decode_error": None if decode_ok else "decode failed",
        "decoded_all_zero": all_zero,
        "decoded_model_action_shape": [32, 32],
        "decode_ms": 1.25,
    }


def physical_actions() -> np.ndarray:
    actions = np.full((32, 14), 0.25, dtype=np.float64)
    actions[:, 0] = np.linspace(0.1, 0.2, 32)
    return actions


class FakePolicy:
    metadata = {"upstream": "fake"}

    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.observations: list[dict] = []

    def infer(self, observation: dict, **kwargs: object) -> dict:
        self.observations.append(observation)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return dict(self.response)


def valid_fake_policy() -> FakePolicy:
    return FakePolicy(
        {
            "actions": physical_actions(),
            CAPTURE_FIELD: capture(),
            "policy_timing": {"infer_ms": 12.0},
        }
    )


def test_extracts_only_the_fast_payload_not_all_generated_tokens() -> None:
    raw = [17, 900, 1, 0]
    fast = extract_fast_action_tokens(raw, FakeSentencePiece(), fast_skip_tokens=128)

    # 1000 - 1 - 128 - [870, 865] == [1, 6].  Formatting/EOS/padding
    # tokens from the raw sequence are deliberately absent.
    assert fast == (1, 6)
    assert tuple(raw) != fast


@pytest.mark.parametrize(
    "decoded",
    [
        "no action marker",
        "Action: payload without terminator",
        "Action: |",
    ],
)
def test_missing_or_malformed_fast_payload_is_rejected(decoded: str) -> None:
    tokenizer = FakeSentencePiece()
    tokenizer.decode = lambda _: decoded  # type: ignore[method-assign]
    with pytest.raises(FastDecodeError):
        extract_fast_action_tokens([2, 3], tokenizer, fast_skip_tokens=128)


def test_observable_policy_returns_tokens_and_final_physical_chunk() -> None:
    wrapped = ObservableFastPolicy(valid_fake_policy())
    response = wrapped.infer({"prompt": "Transfer cube"})

    assert response["executable"] is True
    assert response["actions"].shape == (32, 14)
    metadata = response["openpi_observability"]
    assert metadata["raw_paligemma_token_ids"] == [17, 900, 1, 0]
    assert metadata["fast_action_token_ids"] == [3, 5, 8]
    assert metadata["raw_paligemma_token_ids"] != metadata["fast_action_token_ids"]
    assert metadata["decoded_model_action_shape"] == [32, 32]
    assert metadata["decoded_physical_action_shape"] == [32, 14]
    assert metadata["predicted_action_horizon"] == 32
    assert metadata["executed_action_horizon"] == 10
    assert metadata["checkpoint_uri"] == CHECKPOINT_URI
    assert metadata["checkpoint_compatibility"] == CHECKPOINT_COMPATIBILITY
    assert metadata["decode_ms"] == pytest.approx(1.25)
    assert metadata["total_policy_adapter_ms"] >= 0


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"actions": physical_actions()}, "decode"),
        (
            {
                "actions": physical_actions(),
                CAPTURE_FIELD: capture(decode_ok=False),
            },
            "decode failed",
        ),
        (
            {
                "actions": physical_actions(),
                CAPTURE_FIELD: capture(all_zero=True),
            },
            "all-zero",
        ),
        (
            {
                "actions": np.ones((31, 14)),
                CAPTURE_FIELD: capture(),
            },
            "shape",
        ),
        (
            {
                "actions": np.full((32, 14), np.nan),
                CAPTURE_FIELD: capture(),
            },
            "NaN",
        ),
    ],
)
def test_invalid_decode_or_action_never_survives_as_executable(
    response: dict, message: str
) -> None:
    wrapped = ObservableFastPolicy(FakePolicy(response))
    result = wrapped.infer({"prompt": "Transfer cube"})

    assert result["executable"] is False
    assert "actions" not in result
    assert result["openpi_observability"]["decode_ok"] is False
    assert message.lower() in result["openpi_observability"]["decode_error"].lower()
    with pytest.raises(InvalidPolicyResponse):
        require_executable_actions(result)


def test_underlying_inference_error_becomes_structured_non_executable_response() -> None:
    wrapped = ObservableFastPolicy(FakePolicy(error=RuntimeError("model failure")))
    result = wrapped.infer({"prompt": "Transfer cube"})

    assert result["executable"] is False
    assert "actions" not in result
    assert "model failure" in result["openpi_observability"]["decode_error"]


def test_require_executable_actions_returns_only_requested_first_ten() -> None:
    response = ObservableFastPolicy(valid_fake_policy()).infer({"prompt": "Transfer cube"})
    executed = require_executable_actions(response, executed_action_horizon=10)

    assert executed.shape == (10, 14)
    np.testing.assert_allclose(executed, physical_actions()[:10])
    executed[0, 0] = 999
    assert response["actions"][0, 0] != 999


def test_client_recorder_stores_array_once_and_compact_json(tmp_path) -> None:
    policy = ObservableFastPolicy(valid_fake_policy())
    recorder = FastPolicyClientRecorder(policy, tmp_path)

    response = recorder.infer({"prompt": "Transfer cube"})
    record = recorder.records[0]

    assert response["executable"] is True
    assert record.executable is True
    assert record.action_chunk is not None and record.action_chunk.shape == (32, 14)
    assert record.action_path is not None and record.action_path.exists()
    assert record.summary_path is not None and record.summary_path.exists()
    saved = np.load(record.action_path)["actions"]
    np.testing.assert_allclose(saved, physical_actions())
    summary_text = record.summary_path.read_text()
    summary = json.loads(summary_text)
    assert summary["action_shape"] == [32, 14]
    assert "\"actions\"" not in summary_text
    assert summary["observability"]["raw_token_count"] == 4

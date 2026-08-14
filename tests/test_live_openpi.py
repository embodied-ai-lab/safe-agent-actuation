"""Live pi0-FAST decode test.

This is skipped until a validated pi0-FAST ALOHA checkpoint is available. No
official pi0-FAST ALOHA-simulator expert checkpoint exists at the pinned OpenPI
commit, so the live path is a documented release blocker. When a validated
checkpoint is set through ``SAFE_ACTUATION_VALIDATED_CHECKPOINT`` and a policy
server is running, this test asserts that one live response decodes to a finite,
correctly shaped ALOHA action.

Run only on a GPU node with the OpenPI environment:

    make test-openpi
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live_openpi

VALIDATED_CHECKPOINT_ENV = "SAFE_ACTUATION_VALIDATED_CHECKPOINT"


def _validated_checkpoint() -> str | None:
    value = os.environ.get(VALIDATED_CHECKPOINT_ENV)
    return value if value else None


def test_live_pi0_fast_decodes_valid_aloha_action():
    checkpoint = _validated_checkpoint()
    if checkpoint is None:
        pytest.skip(
            "no validated pi0-FAST ALOHA checkpoint is configured. The official "
            "pi0_fast_base checkpoint is a fine-tuning base whose zero-shot ALOHA "
            "compatibility is unresolved (release blocker). Set "
            f"{VALIDATED_CHECKPOINT_ENV} once a validated checkpoint exists."
        )

    # Only import OpenPI once a validated checkpoint is configured, so the course
    # environment without JAX or OpenPI can still collect this file.
    import numpy as np

    from safe_agent_actuation.openpi_adapter import (
        MODEL_ACTION_DIMENSION,
        PREDICTED_ACTION_HORIZON,
        SIMULATOR_ACTION_DIMENSION,
        create_observable_fast_policy,
        require_executable_actions,
    )

    policy = create_observable_fast_policy(checkpoint)
    observation = {
        "state": np.zeros(SIMULATOR_ACTION_DIMENSION, dtype=np.float64),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
        "prompt": "Transfer cube",
    }
    response = policy.infer(observation)
    assert response.get("executable") is True, "live response was not executable"
    observability = response["openpi_observability"]
    assert observability["decode_ok"] is True
    assert observability["decoded_model_action_shape"] == [
        PREDICTED_ACTION_HORIZON,
        MODEL_ACTION_DIMENSION,
    ]
    chunk = require_executable_actions(response, executed_action_horizon=10)
    assert chunk.shape == (10, SIMULATOR_ACTION_DIMENSION)
    assert np.all(np.isfinite(chunk))
    assert np.any(chunk), "decoded action must not be an all-zero fallback"

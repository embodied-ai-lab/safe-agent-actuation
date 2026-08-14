from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from safe_agent_actuation.fixtures import (
    ActionFixture,
    FixtureSource,
    load_action_fixture,
    save_action_fixture,
)
from safe_agent_actuation.types import PhysicalLimits


def make_fixture() -> ActionFixture:
    return ActionFixture(
        action_chunk=[[0.0, 0.0], [0.2, 0.5]],
        current_state=[0.0, 0.0],
        limits=PhysicalLimits(
            [-1.0, 0.0], [1.0, 1.0], ("joint", "gripper"), "fixture test limits"
        ),
        source=FixtureSource.DIAGNOSTIC_SYNTHETIC,
        description="small deterministic test action",
        raw_fast_token_ids=None,
        metadata={"generator": "unit test", "seed": 0},
    )


def test_fixture_round_trip_is_deterministic_and_labeled(tmp_path: Path) -> None:
    path = save_action_fixture(make_fixture(), tmp_path / "fixture.npz")
    first = load_action_fixture(path)
    second = load_action_fixture(path)
    assert first.source is FixtureSource.DIAGNOSTIC_SYNTHETIC
    assert first.source_label == "diagnostic_synthetic"
    assert first.description == second.description
    assert first.metadata == second.metadata == {"generator": "unit test", "seed": 0}
    assert first.dimension_names == second.dimension_names == ("joint", "gripper")
    np.testing.assert_array_equal(first.action_chunk, second.action_chunk)
    np.testing.assert_array_equal(first.current_state, second.current_state)
    assert not first.action_chunk.flags.writeable


def test_fixture_preserves_live_token_observability(tmp_path: Path) -> None:
    fixture = ActionFixture(
        action_chunk=[[0.0]],
        current_state=[0.0],
        limits=PhysicalLimits([-1.0], [1.0], ("joint",), "live config limits"),
        source=FixtureSource.CAPTURED_LIVE_PI0_FAST,
        description="captured from a live simulator run",
        raw_fast_token_ids=(1, 2, 3),
        metadata={"run_id": "live-001"},
    )
    loaded = load_action_fixture(save_action_fixture(fixture, tmp_path / "live.npz"))
    assert loaded.tokens == (1, 2, 3)
    assert loaded.source is FixtureSource.CAPTURED_LIVE_PI0_FAST


def test_fixture_requires_honest_source_label() -> None:
    with pytest.raises(ValueError, match="source"):
        ActionFixture(
            [[0.0]],
            [0.0],
            PhysicalLimits([-1.0], [1.0], ("joint",), "test"),
            "model_generated_maybe",
            "ambiguous source",
        )


def test_loader_rejects_missing_schema_fields(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, action_chunk=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="requires"):
        load_action_fixture(path)


def test_loader_disables_pickle(tmp_path: Path) -> None:
    path = tmp_path / "pickled.npz"
    np.savez_compressed(
        path,
        action_chunk=np.array([[{"unsafe": True}]], dtype=object),
        current_state=np.array([0.0]),
        lower_limits=np.array([-1.0]),
        upper_limits=np.array([1.0]),
        dimension_names=np.array(["joint"]),
        source_label=np.array("diagnostic_synthetic"),
        description=np.array("must not load"),
        physical_limits_source=np.array("test"),
    )
    with pytest.raises(ValueError):
        load_action_fixture(path)


def test_fixture_rejects_out_of_range_action() -> None:
    with pytest.raises(ValueError, match="outside"):
        ActionFixture(
            [[2.0]],
            [0.0],
            PhysicalLimits([-1.0], [1.0], ("joint",), "test"),
            FixtureSource.GENERATED_BOUNDARY_TEST,
            "invalid boundary fixture",
        )

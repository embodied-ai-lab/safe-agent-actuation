from __future__ import annotations

import numpy as np
import pytest

from safe_agent_actuation.motion import JointMotionAnalyzer, analyze_motion
from safe_agent_actuation.types import PhysicalLimits, RiskClassification

NAMES = ("joint", "gripper")


def analyze(action: object, state: object, *, threshold: float = 0.5):
    return analyze_motion(action, state, [0.0, 0.0], [10.0, 1.0], threshold, NAMES)


def test_all_zero_action_chunk_is_valid_low_risk() -> None:
    result = analyze(np.zeros((3, 2)), np.zeros(2), threshold=0.1)
    assert result.valid
    assert result.classification is RiskClassification.LOW
    assert result.motion_score == 0.0
    assert result.intra == (0.0, 0.0)
    assert result.inter == (0.0, 0.0)


def test_formula_records_named_per_dimension_scores() -> None:
    result = analyze([[2.0, 0.0], [4.0, 1.0]], [1.0, 0.0], threshold=0.6)
    assert result.intra == pytest.approx((0.2, 1.0))
    assert result.inter == pytest.approx((0.1, 0.0))
    assert result.motion_score == pytest.approx(1.0)
    assert result.max_dimension == "gripper"
    assert result.per_dimension()["joint"] == pytest.approx({"intra": 0.2, "inter": 0.1})


def test_intra_chunk_high_risk_movement() -> None:
    result = analyze([[1.0, 0.1], [8.0, 0.1]], [1.0, 0.1], threshold=0.7)
    assert result.high_risk
    assert result.max_dimension == "joint"
    assert result.motion_score == pytest.approx(0.7)


def test_inter_chunk_high_risk_jump() -> None:
    result = analyze([[8.0, 0.1], [8.0, 0.1]], [1.0, 0.1], threshold=0.7)
    assert result.high_risk
    assert result.inter == pytest.approx((0.7, 0.0))


def test_gripper_uses_its_own_physical_range() -> None:
    result = analyze([[0.0, 0.25], [0.0, 0.75]], [0.0, 0.25], threshold=0.5)
    assert result.motion_score == pytest.approx(0.5)
    assert result.max_dimension == "gripper"
    assert result.high_risk


def test_threshold_boundary_is_high_risk() -> None:
    result = analyze([[0.0, 0.0], [5.0, 0.0]], [0.0, 0.0], threshold=0.5)
    assert result.classification is RiskClassification.HIGH


@pytest.mark.parametrize(
    ("action", "state", "lower", "upper", "expected_reason"),
    [
        ([[0.0, 0.0]], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], "range"),
        ([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "dimension"),
        ([[0.0, 0.0]], [0.0], [0.0, 0.0], [1.0, 1.0], "current_state"),
        ([[np.nan, 0.0]], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "NaN"),
        ([[np.inf, 0.0]], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "infinity"),
        ([[2.0, 0.0]], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "outside"),
        (np.empty((0, 2)), [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "timestep"),
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "[H, D]"),
    ],
)
def test_invalid_inputs_fail_closed(
    action: object, state: object, lower: object, upper: object, expected_reason: str
) -> None:
    result = analyze_motion(action, state, lower, upper, 0.5)
    assert not result.valid
    assert result.blocked
    assert result.classification is RiskClassification.INVALID
    assert np.isinf(result.motion_score)
    assert expected_reason in (result.reason or "")


@pytest.mark.parametrize("threshold", [-0.1, np.nan, np.inf, "bad"])
def test_invalid_threshold_fails_closed(threshold: object) -> None:
    result = analyze_motion([[0.0]], [0.0], [-1.0], [1.0], threshold)  # type: ignore[arg-type]
    assert result.blocked
    assert "threshold" in (result.reason or "")


def test_configured_analyzer_uses_limit_names() -> None:
    limits = PhysicalLimits([-1.0, 0.0], [1.0, 1.0], NAMES, "verified test limits")
    result = JointMotionAnalyzer(limits, threshold=0.2).analyze([[0.0, 0.0]], [0.0, 0.0])
    assert result.valid
    assert result.dimension_names == NAMES


def test_physical_limits_reject_zero_range_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="range"):
        PhysicalLimits([0.0], [0.0], ("joint",), "test")
    with pytest.raises(ValueError, match="unique"):
        PhysicalLimits([0.0, 0.0], [1.0, 1.0], ("joint", "joint"), "test")

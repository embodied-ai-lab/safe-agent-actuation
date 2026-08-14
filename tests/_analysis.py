"""Shared test helpers that build MotionAnalysis values directly.

Provided-component tests use these so they do not depend on the graded motion
computation in ``motion.py``. Constructing the dataclass by hand exercises the
actuator, records, and protection code without revealing the intra, inter, or
threshold formulas that students implement.
"""

from __future__ import annotations

from collections.abc import Sequence

from safe_agent_actuation.types import MotionAnalysis, RiskClassification


def low_risk_analysis(
    dimension_names: Sequence[str] = ("dimension_0", "dimension_1"),
    *,
    motion_score: float = 0.0,
    threshold: float = 0.1,
) -> MotionAnalysis:
    names = tuple(dimension_names)
    count = len(names)
    return MotionAnalysis(
        valid=True,
        classification=RiskClassification.LOW,
        motion_score=motion_score,
        threshold=threshold,
        dimension_names=names,
        intra=(0.0,) * count,
        inter=(0.0,) * count,
        max_dimension_index=0,
        max_dimension=names[0],
    )


def high_risk_analysis(
    dimension_names: Sequence[str] = ("joint", "gripper"),
    *,
    motion_score: float = 0.25,
    threshold: float = 0.2,
) -> MotionAnalysis:
    names = tuple(dimension_names)
    count = len(names)
    intra = [0.0] * count
    intra[0] = motion_score
    return MotionAnalysis(
        valid=True,
        classification=RiskClassification.HIGH,
        motion_score=motion_score,
        threshold=threshold,
        dimension_names=names,
        intra=tuple(intra),
        inter=(0.0,) * count,
        max_dimension_index=0,
        max_dimension=names[0],
    )

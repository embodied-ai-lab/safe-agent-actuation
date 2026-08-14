"""Joint Motion Analyzer and Threshold Gate.

Risk is evaluated from a decoded continuous action chunk, never from raw FAST
token identifiers.  Invalid inputs produce an explicit blocked analysis rather
than being allowed to propagate toward the actuator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .types import MotionAnalysis, PhysicalLimits, RiskClassification


def _blocked(reason: str, *, threshold: float, names: Sequence[str] = ()) -> MotionAnalysis:
    """Construct the uniform fail-closed result used by every validation path."""

    return MotionAnalysis(
        valid=False,
        classification=RiskClassification.INVALID,
        motion_score=float("inf"),
        threshold=float(threshold),
        dimension_names=tuple(names),
        intra=(),
        inter=(),
        max_dimension_index=None,
        max_dimension=None,
        reason=reason,
    )


def _as_float_array(value: ArrayLike, name: str) -> tuple[np.ndarray | None, str | None]:
    try:
        return np.asarray(value, dtype=np.float64), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must contain numeric values"


def _intra_chunk_motion(action: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Normalized intra-chunk motion for every dimension.

    ``action`` has shape ``[H, D]`` and ``ranges`` has shape ``[D]`` with every
    entry positive.  The result has shape ``[D]``.
    """

    # TODO(student): compute the normalized intra-chunk motion.
    # Input: action [H, D] finite float array, ranges [D] positive float array.
    # Output: return a [D] array where entry j is
    #   (max over timesteps of action[:, j] - min over timesteps of action[:, j]) / ranges[j].
    # Invalid input is already rejected before this function is called.
    # Verify: pytest tests/test_motion.py -k intra ; make low-risk
    raise NotImplementedError("TODO(student): intra-chunk-motion")


def _inter_state_motion(
    action: np.ndarray, state: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    """Normalized motion from the observed state to the first action.

    ``action`` has shape ``[H, D]``, ``state`` and ``ranges`` have shape
    ``[D]``.  The result has shape ``[D]``.
    """

    # TODO(student): compute the normalized current-state-to-first-action motion.
    # Input: action [H, D], state [D] (the observed pose q_t), ranges [D] positive.
    # Output: return a [D] array where entry j is
    #   abs(action[0, j] - state[j]) / ranges[j].
    # Invalid input is already rejected before this function is called.
    # Verify: pytest tests/test_motion.py -k inter ; make low-risk
    raise NotImplementedError("TODO(student): inter-chunk-motion")


def _score_and_classify(
    intra_array: np.ndarray, inter_array: np.ndarray, threshold: float
) -> tuple[np.ndarray, int, float, RiskClassification]:
    """Combine per-dimension scores, find the maximum, and apply the gate.

    Returns the per-dimension maximum array, the index of the maximum
    dimension, the scalar motion score, and the risk classification.
    """

    # TODO(student): reduce to one score and classify it.
    # Input: intra_array [D], inter_array [D], threshold (finite, non-negative).
    # Output: return (per_dimension_max, max_index, score, classification) where
    #   per_dimension_max[j] = max(intra_array[j], inter_array[j]);
    #   max_index is the argmax dimension; score is per_dimension_max[max_index];
    #   classification is RiskClassification.HIGH when score >= threshold, else LOW.
    # Note: the comparison is >=, so a score equal to the threshold is high risk.
    # Verify: pytest tests/test_motion.py ; make low-risk ; make threshold-sweep
    raise NotImplementedError("TODO(student): max-score-and-classification")


def analyze_motion(
    action_chunk: ArrayLike,
    current_state: ArrayLike,
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    threshold: float,
    dimension_names: Sequence[str] | None = None,
) -> MotionAnalysis:
    """Compute the paper's normalized joint-motion score.

    Given an action chunk with shape ``[H, D]``, the per-dimension values are::

        intra[j] = (max(action[:, j]) - min(action[:, j])) / range[j]
        inter[j] = abs(action[0, j] - current[j]) / range[j]

    The score is ``max_j(max(intra[j], inter[j]))``.  A valid score equal to
    the threshold is high risk.  Every malformed or physically invalid input
    returns ``classification == "invalid"`` and must be blocked.
    """

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError):
        threshold_value = float("nan")
    if not np.isfinite(threshold_value) or threshold_value < 0:
        return _blocked("threshold must be finite and non-negative", threshold=threshold_value)

    action, error = _as_float_array(action_chunk, "action_chunk")
    if error:
        return _blocked(error, threshold=threshold_value)
    state, error = _as_float_array(current_state, "current_state")
    if error:
        return _blocked(error, threshold=threshold_value)
    lower, error = _as_float_array(lower_limits, "lower_limits")
    if error:
        return _blocked(error, threshold=threshold_value)
    upper, error = _as_float_array(upper_limits, "upper_limits")
    if error:
        return _blocked(error, threshold=threshold_value)
    assert action is not None and state is not None and lower is not None and upper is not None

    if action.ndim != 2:
        return _blocked("action_chunk must have shape [H, D]", threshold=threshold_value)
    horizon, dimensions = action.shape
    if horizon == 0:
        return _blocked(
            "action_chunk must contain at least one timestep", threshold=threshold_value
        )
    if dimensions == 0:
        return _blocked(
            "action_chunk must contain at least one dimension", threshold=threshold_value
        )
    if state.ndim != 1:
        return _blocked("current_state must have shape [D]", threshold=threshold_value)
    if lower.ndim != 1:
        return _blocked("lower_limits must have shape [D]", threshold=threshold_value)
    if upper.ndim != 1:
        return _blocked("upper_limits must have shape [D]", threshold=threshold_value)
    if state.shape != (dimensions,):
        return _blocked(
            f"current_state dimension mismatch: expected {dimensions}, got {state.size}",
            threshold=threshold_value,
        )
    if lower.shape != (dimensions,) or upper.shape != (dimensions,):
        return _blocked(
            f"physical-limit dimension mismatch: expected {dimensions}",
            threshold=threshold_value,
        )

    if dimension_names is None:
        names = tuple(f"dimension_{index}" for index in range(dimensions))
    else:
        try:
            names = tuple(dimension_names)
        except TypeError:
            return _blocked("dimension_names must be a sequence", threshold=threshold_value)
        if len(names) != dimensions:
            return _blocked(
                f"dimension_names mismatch: expected {dimensions}, got {len(names)}",
                threshold=threshold_value,
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            return _blocked("dimension_names must be non-empty strings", threshold=threshold_value)
        if len(set(names)) != len(names):
            return _blocked("dimension_names must be unique", threshold=threshold_value)

    for value, name in (
        (action, "action_chunk"),
        (state, "current_state"),
        (lower, "lower_limits"),
        (upper, "upper_limits"),
    ):
        if not np.all(np.isfinite(value)):
            return _blocked(
                f"{name} contains NaN or infinity", threshold=threshold_value, names=names
            )

    ranges = upper - lower
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        return _blocked(
            "every physical range must be finite and greater than zero",
            threshold=threshold_value,
            names=names,
        )
    if np.any(action < lower[np.newaxis, :]) or np.any(action > upper[np.newaxis, :]):
        return _blocked(
            "action_chunk contains a command outside configured physical limits",
            threshold=threshold_value,
            names=names,
        )
    if np.any(state < lower) or np.any(state > upper):
        return _blocked(
            "current_state is outside configured physical limits",
            threshold=threshold_value,
            names=names,
        )

    # The per-dimension scores are a direct transcription of the paper's
    # equations, computed by the graded helpers above.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        try:
            intra_array = _intra_chunk_motion(action, ranges)
            inter_array = _inter_state_motion(action, state, ranges)
            per_dimension_max, max_index, score, classification = _score_and_classify(
                intra_array, inter_array, threshold_value
            )
        except FloatingPointError:
            return _blocked(
                "motion score could not be computed as finite values",
                threshold=threshold_value,
                names=names,
            )

    if not (
        np.all(np.isfinite(intra_array))
        and np.all(np.isfinite(inter_array))
        and np.all(np.isfinite(per_dimension_max))
        and np.isfinite(score)
    ):
        return _blocked(
            "motion score could not be computed as finite values",
            threshold=threshold_value,
            names=names,
        )

    return MotionAnalysis(
        valid=True,
        classification=classification,
        motion_score=score,
        threshold=threshold_value,
        dimension_names=names,
        intra=tuple(float(value) for value in intra_array),
        inter=tuple(float(value) for value in inter_array),
        max_dimension_index=max_index,
        max_dimension=names[max_index],
    )


class JointMotionAnalyzer:
    """Configured Joint Motion Analyzer plus Threshold Gate."""

    def __init__(
        self,
        limits: PhysicalLimits | ArrayLike,
        upper_limits: ArrayLike | None = None,
        threshold: float = 0.05,
        dimension_names: Sequence[str] | None = None,
        limits_source: str = "configured physical limits",
    ) -> None:
        if isinstance(limits, PhysicalLimits):
            if upper_limits is not None:
                raise TypeError("upper_limits must not be provided with PhysicalLimits")
            self.lower_limits = limits.lower
            self.upper_limits = limits.upper
            self.dimension_names = limits.dimension_names
            self.limits_source = limits.source
        else:
            self.lower_limits = limits
            self.upper_limits = upper_limits
            self.dimension_names = tuple(dimension_names) if dimension_names is not None else None
            self.limits_source = limits_source
        self.threshold = threshold

    def analyze(self, action_chunk: ArrayLike, current_state: ArrayLike) -> MotionAnalysis:
        if self.upper_limits is None:
            try:
                threshold = float(self.threshold)
            except (TypeError, ValueError, OverflowError):
                threshold = float("nan")
            return _blocked("upper_limits are required", threshold=threshold)
        return analyze_motion(
            action_chunk=action_chunk,
            current_state=current_state,
            lower_limits=self.lower_limits,
            upper_limits=self.upper_limits,
            threshold=self.threshold,
            dimension_names=self.dimension_names,
        )

    __call__ = analyze


MotionAnalyzer = JointMotionAnalyzer


def classify_motion(*args: object, **kwargs: object) -> MotionAnalysis:
    """Compatibility alias for :func:`analyze_motion`."""

    return analyze_motion(*args, **kwargs)  # type: ignore[arg-type]

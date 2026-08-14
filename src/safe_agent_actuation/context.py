"""Deterministic context construction and a transparent course policy.

The SST mapping uses the exact key names and condition syntax implemented by
``ContextVerifier`` in the pinned IoTAuth source.  Snake-case names remain the
course package's stable Python/record schema.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time

LOCATION_KEY = "Location"
PEOPLE_KEY = "Number of People"
TIME_KEY = "Time of Day"
TASK_KEY = "Task Name"
ROBOT_KEY = "Robot ID"

_PYTHON_FIELDS = frozenset(
    {"location", "nearby_people", "permitted_time", "task_name", "robot_id"}
)
_SST_FIELDS = frozenset({LOCATION_KEY, PEOPLE_KEY, TIME_KEY, TASK_KEY, ROBOT_KEY})
_TIME_PATTERN = re.compile(r"^(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?$")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _valid_time(value: object, field_name: str = "permitted_time") -> str:
    if not isinstance(value, str) or _TIME_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use zero-padded HH:MM or HH:MM:SS")
    # The regular expression checks the range; parsing makes that invariant
    # explicit and guards against future changes to the expression.
    time.fromisoformat(value)
    return value


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Strict, deterministic high-risk authorization context."""

    location: str
    nearby_people: int
    permitted_time: str
    task_name: str
    robot_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", _required_text(self.location, "location"))
        if isinstance(self.nearby_people, bool) or not isinstance(self.nearby_people, int):
            raise ValueError("nearby_people must be a non-negative integer")
        if self.nearby_people < 0:
            raise ValueError("nearby_people must be a non-negative integer")
        object.__setattr__(self, "permitted_time", _valid_time(self.permitted_time))
        object.__setattr__(self, "task_name", _required_text(self.task_name, "task_name"))
        object.__setattr__(self, "robot_id", _required_text(self.robot_id, "robot_id"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AuthorizationContext:
        """Parse the record schema, rejecting missing and unknown fields."""

        if not isinstance(value, Mapping):
            raise ValueError("context must be a mapping")
        keys = set(value)
        missing = _PYTHON_FIELDS - keys
        extra = keys - _PYTHON_FIELDS
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unknown fields: {', '.join(sorted(str(key) for key in extra))}")
            raise ValueError("malformed context (" + "; ".join(details) + ")")
        return cls(
            location=value["location"],  # type: ignore[arg-type]
            nearby_people=value["nearby_people"],  # type: ignore[arg-type]
            permitted_time=value["permitted_time"],  # type: ignore[arg-type]
            task_name=value["task_name"],  # type: ignore[arg-type]
            robot_id=value["robot_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_sst_context(cls, value: Mapping[str, object]) -> AuthorizationContext:
        """Parse an SST request context with exact, case-sensitive keys."""

        if not isinstance(value, Mapping) or set(value) != _SST_FIELDS:
            raise ValueError("malformed SST context: exact supported fields are required")
        return cls(
            location=value[LOCATION_KEY],  # type: ignore[arg-type]
            nearby_people=value[PEOPLE_KEY],  # type: ignore[arg-type]
            permitted_time=value[TIME_KEY],  # type: ignore[arg-type]
            task_name=value[TASK_KEY],  # type: ignore[arg-type]
            robot_id=value[ROBOT_KEY],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "location": self.location,
            "nearby_people": self.nearby_people,
            "permitted_time": self.permitted_time,
            "task_name": self.task_name,
            "robot_id": self.robot_id,
        }

    def to_sst_context(self) -> dict[str, str | int]:
        """Return the request-context object understood by current SST."""

        return {
            LOCATION_KEY: self.location,
            PEOPLE_KEY: self.nearby_people,
            TIME_KEY: self.permitted_time,
            TASK_KEY: self.task_name,
            ROBOT_KEY: self.robot_id,
        }

    def to_sst_purpose(self, target_group: str = "Actuators") -> dict[str, object]:
        """Return an SST session-key request purpose with this context."""

        group = _required_text(target_group, "target_group")
        return {"group": group, "context": self.to_sst_context()}


ActionContext = AuthorizationContext


@dataclass(frozen=True, slots=True)
class ContextPolicyDecision:
    allowed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("policy decision requires a reason")


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Readable local mirror of the registered SST communication policy."""

    allowed_locations: tuple[str, ...]
    max_nearby_people: int
    start_time: str
    end_time: str
    allowed_tasks: tuple[str, ...]
    allowed_robot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        locations = self._allowlist(self.allowed_locations, "allowed_locations")
        tasks = self._allowlist(self.allowed_tasks, "allowed_tasks")
        robots = self._allowlist(self.allowed_robot_ids, "allowed_robot_ids")
        if isinstance(self.max_nearby_people, bool) or not isinstance(self.max_nearby_people, int):
            raise ValueError("max_nearby_people must be a non-negative integer")
        if self.max_nearby_people < 0:
            raise ValueError("max_nearby_people must be a non-negative integer")
        start = _valid_time(self.start_time, "start_time")
        end = _valid_time(self.end_time, "end_time")
        if time.fromisoformat(start) > time.fromisoformat(end):
            raise ValueError("start_time must not be after end_time for SST range syntax")
        object.__setattr__(self, "allowed_locations", locations)
        object.__setattr__(self, "allowed_tasks", tasks)
        object.__setattr__(self, "allowed_robot_ids", robots)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)

    @staticmethod
    def _allowlist(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a non-empty sequence") from exc
        if not items or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"{field_name} must contain non-empty strings")
        normalized = tuple(item.strip() for item in items)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{field_name} must not contain duplicates")
        return normalized

    @classmethod
    def course_default(cls) -> ContextPolicy:
        return cls(
            allowed_locations=("ASU Robotics Lab",),
            max_nearby_people=0,
            start_time="08:00",
            end_time="18:00",
            allowed_tasks=("transfer_cube",),
            allowed_robot_ids=("aloha-sim-01",),
        )

    def evaluate(
        self, context: AuthorizationContext | Mapping[str, object]
    ) -> ContextPolicyDecision:
        """Apply the same supported conditions locally and fail closed."""

        try:
            parsed = (
                context
                if isinstance(context, AuthorizationContext)
                else AuthorizationContext.from_mapping(context)
            )
        except (TypeError, ValueError) as exc:
            return ContextPolicyDecision(False, f"malformed context: {exc}")
        if parsed.location not in self.allowed_locations:
            return ContextPolicyDecision(False, "location is not approved")
        if parsed.nearby_people > self.max_nearby_people:
            return ContextPolicyDecision(False, "nearby_people exceeds the policy maximum")
        parsed_time = time.fromisoformat(parsed.permitted_time)
        if parsed_time < time.fromisoformat(self.start_time) or parsed_time > time.fromisoformat(
            self.end_time
        ):
            return ContextPolicyDecision(False, "permitted_time is outside approved hours")
        if parsed.task_name not in self.allowed_tasks:
            return ContextPolicyDecision(False, "task_name is not approved")
        if parsed.robot_id not in self.allowed_robot_ids:
            return ContextPolicyDecision(False, "robot_id is not registered")
        return ContextPolicyDecision(True, "context satisfies the registered policy")

    def allows(self, context: AuthorizationContext | Mapping[str, object]) -> bool:
        return self.evaluate(context).allowed

    def to_sst_policy_context(self) -> dict[str, dict[str, object]]:
        """Return exact ``ContextVerifier`` policy syntax for Auth registration."""

        return {
            PEOPLE_KEY: {"Max": self.max_nearby_people},
            LOCATION_KEY: {"Allowed": list(self.allowed_locations)},
            TIME_KEY: {"Min": self.start_time, "Max": self.end_time},
            TASK_KEY: {"Allowed": list(self.allowed_tasks)},
            ROBOT_KEY: {"Allowed": list(self.allowed_robot_ids)},
        }

    # A concise alias useful in configuration-generation code.
    to_sst = to_sst_policy_context


class DeterministicContextProvider:
    """Context source whose result changes only through an explicit replacement."""

    def __init__(self, context: AuthorizationContext | Mapping[str, object]) -> None:
        self._context = (
            context
            if isinstance(context, AuthorizationContext)
            else AuthorizationContext.from_mapping(context)
        )

    def build(self) -> AuthorizationContext:
        return self._context

    def get_context(self) -> AuthorizationContext:
        return self.build()

    __call__ = build

    def replace(self, context: AuthorizationContext | Mapping[str, object]) -> None:
        """Explicitly change scenario context; no ambient state is consulted."""

        self._context = (
            context
            if isinstance(context, AuthorizationContext)
            else AuthorizationContext.from_mapping(context)
        )

    @classmethod
    def allowed(cls) -> DeterministicContextProvider:
        return cls(
            AuthorizationContext(
                location="ASU Robotics Lab",
                nearby_people=0,
                permitted_time="14:00",
                task_name="transfer_cube",
                robot_id="aloha-sim-01",
            )
        )

    @classmethod
    def denied_nearby_person(cls) -> DeterministicContextProvider:
        return cls(
            AuthorizationContext(
                location="ASU Robotics Lab",
                nearby_people=1,
                permitted_time="14:00",
                task_name="transfer_cube",
                robot_id="aloha-sim-01",
            )
        )


ContextBuilder = DeterministicContextProvider

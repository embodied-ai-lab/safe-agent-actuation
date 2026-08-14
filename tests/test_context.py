from __future__ import annotations

import pytest

from safe_agent_actuation.context import (
    AuthorizationContext,
    ContextPolicy,
    DeterministicContextProvider,
)


def test_context_has_deterministic_record_and_exact_sst_fields() -> None:
    context = DeterministicContextProvider.allowed().build()
    assert context.to_dict() == {
        "location": "ASU Robotics Lab",
        "nearby_people": 0,
        "permitted_time": "14:00",
        "task_name": "transfer_cube",
        "robot_id": "aloha-sim-01",
    }
    assert context.to_sst_purpose() == {
        "group": "Actuators",
        "context": {
            "Location": "ASU Robotics Lab",
            "Number of People": 0,
            "Time of Day": "14:00",
            "Task Name": "transfer_cube",
            "Robot ID": "aloha-sim-01",
        },
    }


def test_allowed_and_denied_nearby_person_contexts() -> None:
    policy = ContextPolicy.course_default()
    assert policy.evaluate(DeterministicContextProvider.allowed().build()).allowed
    decision = policy.evaluate(DeterministicContextProvider.denied_nearby_person().build())
    assert not decision.allowed
    assert "nearby_people" in decision.reason


def test_policy_produces_current_sst_context_verifier_syntax() -> None:
    assert ContextPolicy.course_default().to_sst_policy_context() == {
        "Number of People": {"Max": 0},
        "Location": {"Allowed": ["ASU Robotics Lab"]},
        "Time of Day": {"Min": "08:00", "Max": "18:00"},
        "Task Name": {"Allowed": ["transfer_cube"]},
        "Robot ID": {"Allowed": ["aloha-sim-01"]},
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {
            "location": "ASU Robotics Lab",
            "nearby_people": 0,
            "permitted_time": "14:00",
            "task_name": "transfer_cube",
        },
        {
            "location": "ASU Robotics Lab",
            "nearby_people": 0,
            "permitted_time": "14:00",
            "task_name": "transfer_cube",
            "robot_id": "aloha-sim-01",
            "unexpected": True,
        },
    ],
)
def test_malformed_context_is_strictly_rejected(mapping: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="malformed"):
        AuthorizationContext.from_mapping(mapping)
    assert not ContextPolicy.course_default().evaluate(mapping).allowed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("location", ""),
        ("nearby_people", -1),
        ("nearby_people", True),
        ("nearby_people", 1.5),
        ("permitted_time", "2:00 PM"),
        ("task_name", " "),
        ("robot_id", ""),
    ],
)
def test_context_field_types_fail_closed(field: str, value: object) -> None:
    values: dict[str, object] = {
        "location": "ASU Robotics Lab",
        "nearby_people": 0,
        "permitted_time": "14:00",
        "task_name": "transfer_cube",
        "robot_id": "aloha-sim-01",
    }
    values[field] = value
    with pytest.raises(ValueError):
        AuthorizationContext.from_mapping(values)


def test_provider_changes_only_on_explicit_replace() -> None:
    provider = DeterministicContextProvider.allowed()
    first = provider.build()
    assert provider.build() is first
    provider.replace(DeterministicContextProvider.denied_nearby_person().build())
    assert provider.build().nearby_people == 1

"""Command-line entry point for the fixture-driven monitoring scenarios.

Every subcommand resolves the repository root, loads committed configuration,
runs one deterministic scenario, writes a structured record, and returns a
status code that distinguishes an expected policy denial from an infrastructure
failure.  Live model inference is never triggered here; that path has its own
GPU-only scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .captured import CapturedRecordError, inspect_response
from .course_config import CourseConfigError, default_threshold, repository_root
from .scenarios import (
    ScenarioError,
    ScenarioOutcome,
    run_bypass,
    run_high_risk_live,
    run_high_risk_offline_diagnostic,
    run_key_validity_live_trial,
    run_low_risk,
    summarize_key_validity_live,
)
from .sweeps import (
    run_threshold_sweep,
    run_threshold_sweep_offline_diagnostic,
    run_validity_diagnostic,
)


def _resolve_threshold(root: Path, value: float | None) -> float:
    return default_threshold(root) if value is None else float(value)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _outcome_payload(outcome: ScenarioOutcome) -> dict[str, Any]:
    return {
        "scenario": outcome.scenario,
        "run_id": outcome.run_id,
        "record": str(outcome.record_path),
        "expected_outcome": outcome.expected,
        "summary": outcome.summary,
    }


def _cmd_scenario(args: argparse.Namespace, root: Path) -> int:
    threshold = _resolve_threshold(root, args.threshold)
    name = args.name
    output_dir = Path(args.output) if args.output else None
    if name == "low-risk":
        outcome = run_low_risk(root, threshold=threshold, output_dir=output_dir)
    elif name in {"high-risk-allow", "high-risk-deny"}:
        context_name = "allowed" if name == "high-risk-allow" else "denied"
        expect_allowed = name == "high-risk-allow"
        if args.offline_diagnostic:
            outcome = run_high_risk_offline_diagnostic(
                root,
                scenario=name,
                threshold=threshold,
                context_name=context_name,
                expect_allowed=expect_allowed,
                output_dir=output_dir,
            )
        else:
            outcome = run_high_risk_live(
                root,
                scenario=name,
                threshold=threshold,
                context_name=context_name,
                expect_allowed=expect_allowed,
                output_dir=output_dir,
            )
    elif name == "bypass":
        outcome = run_bypass(
            root,
            threshold=threshold,
            offline_diagnostic=args.offline_diagnostic,
            output_dir=output_dir,
        )
    else:  # pragma: no cover - argparse restricts the choices.
        raise ScenarioError(f"unknown scenario {name!r}")
    _print(_outcome_payload(outcome))
    return 0


def _cmd_inspect_actions(args: argparse.Namespace, root: Path) -> int:
    result_name = (
        "inspect-actions-configuration-only"
        if args.configuration_only
        else "inspect-actions"
    )
    output_dir = Path(args.output) if args.output else root / "results" / result_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = inspect_response(root, configuration_only=args.configuration_only)
    path = output_dir / "response_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print({"command": "inspect-actions", "record": str(path), "summary": summary})
    return 0


def _cmd_threshold_sweep(args: argparse.Namespace, root: Path) -> int:
    result_name = "threshold-sweep-offline" if args.offline_diagnostic else "threshold-sweep"
    output_dir = Path(args.output) if args.output else root / "results" / result_name
    if args.offline_diagnostic:
        result = run_threshold_sweep_offline_diagnostic(root, output_dir)
        result = {"mode": "offline_diagnostic", **result}
    else:
        result = run_threshold_sweep(root, output_dir)
    _print({"scenario": "threshold-sweep", **result})
    if not args.offline_diagnostic and (
        int(result.get("failed_trial_count", 0)) > 0
        or int(result.get("warmup_failure_count", 0)) > 0
    ):
        return 1
    return 0


def _cmd_key_validity(args: argparse.Namespace, root: Path) -> int:
    live_mode = args.live_trial or args.summarize
    result_name = "grad-key-validity" if live_mode else "grad-key-validity-offline"
    output_dir = Path(args.output) if args.output else root / "results" / result_name
    if args.live_trial:
        trial = run_key_validity_live_trial(root, label=args.label, output_dir=output_dir)
        _print({"scenario": "grad-key-validity", "mode": "live_trial", "trial": trial})
    elif args.summarize:
        result = summarize_key_validity_live(output_dir)
        _print({"scenario": "grad-key-validity", "mode": "live_summary", **result})
    else:
        result = run_validity_diagnostic(root, output_dir)
        _print({"scenario": "grad-key-validity", "mode": "offline_diagnostic", **result})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safe-agent-actuation",
        description="Fixture-driven safe actuation monitoring scenarios.",
    )
    parser.add_argument(
        "--root",
        help="Repository root override; defaults to SAFE_ACTUATION_ROOT or the checkout.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scenario = subparsers.add_parser("scenario", help="Run one monitoring scenario.")
    scenario.add_argument(
        "name",
        choices=("low-risk", "high-risk-allow", "high-risk-deny", "bypass"),
    )
    scenario.add_argument("--threshold", type=float, default=None)
    scenario.add_argument(
        "--output",
        default=None,
        help="Override the result directory; diagnostic defaults remain separate.",
    )
    scenario.add_argument(
        "--offline-diagnostic",
        action="store_true",
        help=(
            "Use the labeled offline authorizer instead of live SST. Never a "
            "substitute for the graded live run."
        ),
    )
    scenario.set_defaults(func=_cmd_scenario)

    inspect = subparsers.add_parser(
        "inspect-actions",
        help="Validate and report the committed genuine pi0-FAST response.",
    )
    inspect.add_argument("--output", default=None)
    inspect.add_argument(
        "--configuration-only",
        action="store_true",
        help=(
            "Report only the configured response contract. This diagnostic does "
            "not inspect a model response and is not the Part 1 graded path."
        ),
    )
    inspect.set_defaults(func=_cmd_inspect_actions)

    sweep = subparsers.add_parser(
        "threshold-sweep",
        help="Run a fixed-case multi-threshold parameter evaluation.",
    )
    sweep.add_argument("--output", default=None)
    sweep.add_argument(
        "--offline-diagnostic",
        action="store_true",
        help=(
            "Use the labeled deterministic policy-model diagnostic instead of "
            "the default graded live-SST multi-threshold evaluation."
        ),
    )
    sweep.set_defaults(func=_cmd_threshold_sweep)

    validity = subparsers.add_parser("key-validity", help="Session-key validity analysis.")
    validity.add_argument("--output", default=None)
    group = validity.add_mutually_exclusive_group()
    group.add_argument(
        "--live-trial",
        action="store_true",
        help="Run one real-SST trial for the currently generated relative validity.",
    )
    group.add_argument(
        "--summarize",
        action="store_true",
        help="Aggregate the recorded real-SST trials into CSV, JSON, and a plot.",
    )
    validity.add_argument(
        "--label",
        default="default",
        help="Label for one live trial, usually the relative validity string.",
    )
    validity.set_defaults(func=_cmd_key_validity)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = repository_root(args.root)
        return int(args.func(args, root))
    except ScenarioError as exc:
        # An expected policy denial is reported inside the scenario record; a
        # ScenarioError is always an infrastructure or unexpected-state failure.
        print(f"scenario failed: {exc}", file=sys.stderr)
        return 1
    except CourseConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except CapturedRecordError as exc:
        print(f"captured response error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

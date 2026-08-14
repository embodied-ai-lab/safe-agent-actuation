#!/usr/bin/env python3
"""Render a runtime graph with a controlled relative session-key validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-validity", default="20*sec")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.relative_validity or any(
        token in args.relative_validity for token in ("/", "\\", "..", "\n", "\r")
    ):
        raise SystemExit("invalid relative validity")
    graph = json.loads(args.source.read_text(encoding="utf-8"))
    policies = graph.get("commPolicies")
    if not isinstance(policies, list) or len(policies) != 1:
        raise SystemExit("expected exactly one communication policy")
    policies[0]["RelativeValidity"] = args.relative_validity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


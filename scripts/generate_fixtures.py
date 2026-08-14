#!/usr/bin/env python3
"""Regenerate the committed deterministic action fixtures under data/fixtures/.

The output is fully determined by committed constants and the physical limits.
Running this script twice produces byte-identical NPZ payloads and JSON
metadata, so the committed fixtures can be verified in review and in tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    from safe_agent_actuation.course_config import load_physical_limits, repository_root
    from safe_agent_actuation.course_fixtures import write_course_fixtures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None, help="Defaults to <root>/data/fixtures")
    args = parser.parse_args(argv)

    root = repository_root(args.root)
    limits = load_physical_limits(root)
    output_dir = Path(args.output) if args.output else root / "data" / "fixtures"
    written = write_course_fixtures(limits, output_dir)
    for path in written:
        print(f"wrote {path.relative_to(root) if path.is_relative_to(root) else path}")
    print(f"generated {len(written)} files under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

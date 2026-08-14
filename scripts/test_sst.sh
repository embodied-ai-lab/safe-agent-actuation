#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PY:-${root}/.venv/bin/python3}"
test -x "${python}" || python="python3.11"

# The pinned IoTAuth tests import a `tests.helpers` package, so pytest must run
# with the entity/python directory as its working directory. Otherwise the
# course `tests/` package shadows it.
(
  cd "${root}/third_party/iotauth/entity/python"
  "${python}" -m pytest -q tests
)

# Course live SST integration tests need generated runtime state and Auth.
"${python}" -m pytest -q -m live_sst "${root}/tests"


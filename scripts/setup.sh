#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${UV_CACHE_DIR:-/tmp/safe-agent-actuation-uv-${USER:-user}}"
host_name="$(hostname -s)"
if [[ "${host_name,,}" == *login* ]]; then
  echo "Refusing environment setup on a login node. Start a CPU compute allocation." >&2
  exit 2
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif command -v python3.11 >/dev/null 2>&1; then
  python_bin="python3.11"
else
  python_bin="python3"
fi

test -f "${root}/third_party/iotauth/entity/python/pyproject.toml" || {
  echo "SST submodule is missing. Run make submodules." >&2
  exit 2
}
test -f "${root}/uv.lock" || {
  echo "The course uv.lock is missing; refusing an unlocked installation." >&2
  exit 2
}
command -v uv >/dev/null || {
  echo "uv is required. Use the Apptainer image or install uv first." >&2
  exit 2
}

mkdir -p "${cache}"
UV_CACHE_DIR="${cache}" UV_PROJECT_ENVIRONMENT="${root}/.venv" \
  uv sync --project "${root}" --locked --extra dev --no-install-project \
  --python "${python_bin}"

# Avoid an unlocked editable build (and its separately resolved build-system
# requirements).  Both course packages are pure Python, so a .pth file exposes
# the reviewed source trees while every third-party wheel still comes from
# uv.lock.
"${root}/.venv/bin/python3" -c '
import pathlib
import sysconfig
import sys

root = pathlib.Path(sys.argv[1]).resolve()
paths = [root / "src", root / "third_party" / "iotauth" / "entity" / "python"]
target = pathlib.Path(sysconfig.get_paths()["purelib"]) / "safe_agent_actuation_course.pth"
target.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
' "${root}"

echo "Course environment ready at ${root}/.venv"
echo "The CPU environment uses uv.lock and does not install OpenPI, JAX, or model weights."

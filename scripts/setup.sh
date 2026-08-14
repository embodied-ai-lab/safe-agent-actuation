#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${UV_CACHE_DIR:-/tmp/safe-agent-actuation-uv-${USER:-user}}"
python_bin="${PYTHON_BIN:-python3.11}"

test -f "${root}/third_party/openpi/pyproject.toml" || {
  echo "OpenPI submodule is missing. Run make submodules." >&2
  exit 2
}
test -f "${root}/third_party/iotauth/entity/python/pyproject.toml" || {
  echo "SST submodule is missing. Run make submodules." >&2
  exit 2
}
command -v uv >/dev/null || {
  echo "uv is required. Use the Apptainer image or install uv first." >&2
  exit 2
}

mkdir -p "${cache}"
if [[ ! -x "${root}/.venv/bin/python3" ]]; then
  UV_CACHE_DIR="${cache}" uv venv --python "${python_bin}" "${root}/.venv"
fi

UV_CACHE_DIR="${cache}" uv pip install --python "${root}/.venv/bin/python3" \
  -e "${root}[dev]" \
  -e "${root}/third_party/iotauth/entity/python"

echo "Course environment ready at ${root}/.venv"
echo "OpenPI's full GPU environment is installed only by scripts/setup_openpi.sh."


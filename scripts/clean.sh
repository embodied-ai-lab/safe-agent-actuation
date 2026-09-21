#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "WARNING: deleting generated results, runtime SST state, local environments, and caches"
rm -rf \
  "${root}/runtime/sst" \
  "${root}/results/baseline" \
  "${root}/results/inspect-actions" \
  "${root}/results/inspect-actions-configuration-only" \
  "${root}/results/low-risk" \
  "${root}/results/high-risk-allow" \
  "${root}/results/high-risk-allow-offline" \
  "${root}/results/high-risk-deny" \
  "${root}/results/high-risk-deny-offline" \
  "${root}/results/bypass" \
  "${root}/results/bypass-offline" \
  "${root}/results/threshold-sweep" \
  "${root}/results/threshold-sweep-offline" \
  "${root}/results/grad-key-validity" \
  "${root}/results/grad-key-validity-offline" \
  "${root}/.venv" \
  "${root}/.venv-openpi" \
  "${root}/.pytest_cache" \
  "${root}/.ruff_cache"
find "${root}" -path "${root}/third_party" -prune -o -type d -name __pycache__ -exec rm -rf {} +
echo "Generated project state removed. Pinned submodules and staged scratch weights were preserved."

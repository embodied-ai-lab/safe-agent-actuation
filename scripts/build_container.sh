#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(hostname -s)" in
  *login*) echo "Refusing to build an image on a login node" >&2; exit 2 ;;
esac
command -v apptainer >/dev/null || {
  echo "Apptainer is unavailable. On Sol run: module load apptainer/1.4.5" >&2
  exit 2
}
# Apptainer builds SIF images with mksquashfs. On Sol it comes from a module and
# is not on the base PATH, so load it when the command is missing.
command -v mksquashfs >/dev/null || {
  if command -v module >/dev/null; then
    module load squashfs-4.6.1-gcc-11.2.0 2>/dev/null \
      || module load squashfs 2>/dev/null || true
  fi
}
output="${SIF:-/scratch/${USER}/safe-agent-actuation.sif}"
mkdir -p "$(dirname "${output}")"

# A persistent cache on scratch lets a retry reuse already-fetched OCI layers,
# which matters because compute-node fetches from the container registry time
# out intermittently.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/.apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/.apptainer-tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

attempts="${BUILD_ATTEMPTS:-3}"
built=0
for attempt in $(seq 1 "${attempts}"); do
  echo "apptainer build attempt ${attempt} of ${attempts}"
  if apptainer build --force "${output}" "${root}/containers/Apptainer.def"; then
    built=1
    break
  fi
  echo "build attempt ${attempt} failed; retrying after a short pause" >&2
  sleep 20
done
test "${built}" = 1 || { echo "apptainer build failed after ${attempts} attempts" >&2; exit 1; }
sha256sum "${output}" | tee "${output}.sha256"


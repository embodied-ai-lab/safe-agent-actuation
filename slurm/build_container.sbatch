#!/bin/bash
# CPU compute job. Submit from the repository root.
#
# Building the image is a fallback. Prefer the course-provided image and set
# SIF to its path. When you do need to build, supply your account and QoS:
#   sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> slurm/build_container.sbatch
#SBATCH --job-name=safe-actuation-build
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/container-%j.out
#SBATCH --error=results/container-%j.err

set -euo pipefail

case "$(hostname -s)" in
  *login*) echo "Refusing to build on a login node" >&2; exit 2 ;;
esac

module load apptainer/1.4.5
# Apptainer builds SIF images with mksquashfs, which is not on the base PATH.
module load squashfs-4.6.1-gcc-11.2.0 2>/dev/null || module load squashfs 2>/dev/null || true
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${project_root}"
export SIF="${SIF:-/scratch/${USER}/safe-agent-actuation.sif}"
mkdir -p "$(dirname "${SIF}")" results
# scripts/build_container.sh caches OCI layers on scratch and retries the build,
# which is resilient to intermittent registry-fetch timeouts on compute nodes.
./scripts/build_container.sh


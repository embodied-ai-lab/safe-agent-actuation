#!/bin/bash
# Compute job for the four common CSE 494/598 parts.
#
# Supply your account and QoS on the command line, for example:
#   sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> slurm/run_experiments.sbatch
#
# CSE 598 groups add the extension to the same run:
#   sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> \
#       --export=ALL,RUN_GRAD_EXTENSION=1 slurm/run_experiments.sbatch
#
# The extension results go in the same ZIP. There is no second upload.
#SBATCH --job-name=safe-agent-actuation
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

set -euo pipefail

case "$(hostname -s)" in
  *login*) echo "Refusing to run on a login node" >&2; exit 2 ;;
esac

module load apptainer/1.4.5
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${project_root}"
image_path="${SIF:-/scratch/${USER}/safe-agent-actuation.sif}"

# Run only the four common parts unless the CSE 598 extension is requested.
RUN_GRAD_EXTENSION="${RUN_GRAD_EXTENSION:-0}"

test -f "${image_path}" || {
  echo "Missing ${image_path}." >&2
  echo "Set SIF to the course-provided image, or build one with sbatch slurm/build_container.sbatch." >&2
  exit 2
}

mkdir -p results /scratch/"${USER}"/safe-agent-actuation/uv
export UV_CACHE_DIR="/scratch/${USER}/safe-agent-actuation/uv"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export APPTAINERENV_UV_CACHE_DIR="${UV_CACHE_DIR}"
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_SAFE_ACTUATION_ROOT="${project_root}"

lab() {
  apptainer exec \
    --bind "${project_root}:${project_root}" \
    --bind "/scratch/${USER}:/scratch/${USER}" \
    --pwd "${project_root}" \
    "${image_path}" "$@"
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  lab make auth-stop >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

lab make setup
lab make doctor
lab make test-offline

# Part 1
lab make inspect-actions

# Part 2
lab make low-risk

# Part 3 needs a running Auth process.
lab make build-auth
lab make generate-sst
lab make auth-start
lab make test-sst
lab make high-risk-allow
lab make high-risk-deny
lab make bypass
lab make auth-stop

# Part 4
lab make threshold-sweep

# Part 5 runs only for CSE 598 groups.
if [[ "${RUN_GRAD_EXTENSION}" == "1" ]]; then
  lab make grad-key-validity
fi

lab python3 scripts/validate_results.py results


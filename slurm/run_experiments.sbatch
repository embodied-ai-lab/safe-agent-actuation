#!/bin/bash
# Compute job for the four common CSE 494/598 parts.
#
# Build the image and complete the student TODOs before submitting.
# The Fall 2026 course account and QoS are recorded below:
#   sbatch slurm/run_experiments.sbatch
#
# CSE 598 groups add the extension to the same run:
#   sbatch --export=ALL,RUN_GRAD_EXTENSION=1 slurm/run_experiments.sbatch
#
# The extension results go in the same ZIP. There is no second upload.
#SBATCH --job-name=safe-agent-actuation
#SBATCH --account=class_cse494598fall2026
#SBATCH --partition=public
#SBATCH --qos=class
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
runtime="${project_root}/runtime/sst"
auth_owner_file="${runtime}/.auth-owner-${SLURM_JOB_ID}.pid"
openpi_root="${project_root}/third_party/openpi"
iotauth_root="${project_root}/third_party/iotauth"

# Run only the four common parts unless the CSE 598 extension is requested.
RUN_GRAD_EXTENSION="${RUN_GRAD_EXTENSION:-0}"

test -f "${image_path}" || {
  echo "Missing ${image_path}." >&2
  echo "Build it with sbatch slurm/build_container.sbatch, then wait for that job to succeed." >&2
  echo "If you built it at another path, set SIF to that image." >&2
  exit 2
}
for dependency_root in "${openpi_root}" "${iotauth_root}"; do
  test -d "${dependency_root}" || {
    echo "Missing dependency tree: ${dependency_root}" >&2
    exit 2
  }
done

mkdir -p results /scratch/"${USER}"/safe-agent-actuation/uv
export UV_CACHE_DIR="/scratch/${USER}/safe-agent-actuation/uv"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export APPTAINERENV_UV_CACHE_DIR="${UV_CACHE_DIR}"
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_SAFE_ACTUATION_ROOT="${project_root}"
export APPTAINERENV_SST_AUTH_OWNER_FILE="${auth_owner_file}"
export APPTAINERENV_RUN_GRAD_EXTENSION="${RUN_GRAD_EXTENSION}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export APPTAINERENV_PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1

lab() {
  apptainer exec \
    --bind "${project_root}:${project_root}" \
    --bind "${openpi_root}:${openpi_root}:ro" \
    --bind "${iotauth_root}:${iotauth_root}:ro" \
    --bind "/scratch/${USER}:/scratch/${USER}" \
    --pwd "${project_root}" \
    "${image_path}" "$@"
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -f "${auth_owner_file}" && ! -f "${runtime}/auth.pid" ]]; then
    rm -f "${auth_owner_file}"
  elif [[ -f "${auth_owner_file}" && -f "${runtime}/auth.pid" ]]; then
    owned_pid="$(tr -d '\r\n' <"${auth_owner_file}")"
    current_pid="$(tr -d '\r\n' <"${runtime}/auth.pid")"
    if [[ -n "${owned_pid}" && "${current_pid}" == "${owned_pid}" ]]; then
      if lab make auth-stop >/dev/null 2>&1; then
        rm -f "${auth_owner_file}"
      fi
    else
      echo "Refusing to stop non-owned Auth PID ${current_pid}" >&2
    fi
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

lab make setup
lab make doctor
lab make test-baseline

# Part 1
lab make inspect-actions

# Completed-exercise checks intentionally fail in an untouched starter. They
# run only after Part 1 so unrelated TODOs cannot block response inspection.
lab make test-acceptance
lab make test-offline

# Part 2
lab make low-risk

# Parts 3 and 4 share one Apptainer lifetime. Auth is a background Java
# process, so leaving the container between auth-start and the clients can
# tear down its mounted runtime even while a stale PID remains visible.
lab bash -c '
  set -euo pipefail
  runtime="$PWD/runtime/sst"
  owner_file="${SST_AUTH_OWNER_FILE}"

  stop_owned_auth() {
    if [[ ! -f "${owner_file}" ]]; then
      return 0
    fi
    if [[ ! -f "${runtime}/auth.pid" ]]; then
      rm -f "${owner_file}"
      return 0
    fi
    owned_pid="$(tr -d "\r\n" <"${owner_file}")"
    current_pid="$(tr -d "\r\n" <"${runtime}/auth.pid")"
    if [[ -n "${owned_pid}" && "${current_pid}" == "${owned_pid}" ]]; then
      make auth-stop
      rm -f "${owner_file}"
    else
      echo "Refusing to stop non-owned Auth PID ${current_pid}" >&2
      return 1
    fi
  }

  cleanup_sst() {
    status=$?
    trap - EXIT INT TERM
    stop_owned_auth || true
    exit "${status}"
  }
  trap cleanup_sst EXIT INT TERM

  make build-auth
  make generate-sst
  make auth-start
  tr -d "\r\n" <"${runtime}/auth.pid" >"${owner_file}"
  make test-sst
  make high-risk-allow
  make high-risk-deny
  make bypass
  make threshold-sweep

  # The common Auth instance must remain alive through the real multi-threshold
  # evaluation. Stop it only afterward; the extension then owns each Auth instance it
  # regenerates, all within this same Apptainer process lifetime.
  stop_owned_auth
  if [[ "${RUN_GRAD_EXTENSION:-0}" == "1" ]]; then
    make grad-key-validity
  fi
  trap - EXIT INT TERM
'

# Validate exactly the deliverables selected for this group.
if [[ "${RUN_GRAD_EXTENSION}" == "1" ]]; then
  lab python3 scripts/validate_results.py --require-grad results
else
  lab python3 scripts/validate_results.py results
fi

#!/usr/bin/env bash
# CSE 598 extension: real SST session-key validity comparison.
#
# For each configured relative validity, this regenerates the Auth runtime with
# that validity, restarts Auth, and runs one live controlled sequence
# (authorize, reuse, security window after a context change, expiration, and
# reissue). It then aggregates the trials into CSV, JSON, and a plot. The
# comparison uses the real pinned Auth deployment, not a local timer.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${PY:-${root}/.venv/bin/python3}"
test -x "${py}" || py="python3.11"
results="${root}/results/grad-key-validity"
generate="${root}/sst/scripts/generate_runtime.sh"
auth="${root}/sst/scripts/auth.sh"

case "$(hostname -s)" in
  *login*) echo "Refusing to run the SST key-validity extension on a login node" >&2; exit 2 ;;
esac
test -f "${root}/runtime/sst/auth-build/source/auth/auth-server/target/auth-server-jar-with-dependencies.jar" \
  || { echo "Auth is not built. Run make build-auth first." >&2; exit 2; }

# Space separated relative validity periods in SST syntax. Shorter periods keep
# the expiration waits brief; the default spans a useful range.
periods="${SST_KEY_VALIDITY_PERIODS:-5*sec 20*sec 40*sec}"

mkdir -p "${results}"
rm -f \
  "${results}/live_trials.jsonl" \
  "${results}/trials.csv" \
  "${results}/summary.json" \
  "${results}/key_validity.png"

cleanup() {
  "${auth}" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for validity in ${periods}; do
  echo "=== relative validity ${validity} ==="
  "${auth}" stop >/dev/null 2>&1 || true
  SST_RELATIVE_VALIDITY="${validity}" "${generate}"
  "${auth}" start
  "${py}" -m safe_agent_actuation.cli key-validity --live-trial --label "${validity}"
  "${auth}" stop >/dev/null 2>&1 || true
done

"${py}" -m safe_agent_actuation.cli key-validity --summarize
echo "Wrote real-SST key-validity results under ${results}"

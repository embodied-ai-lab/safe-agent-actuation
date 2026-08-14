#!/usr/bin/env bash
set -euo pipefail

action="${1:-status}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="${root}/runtime/sst"
jar="${runtime}/auth-build/source/auth/auth-server/target/auth-server-jar-with-dependencies.jar"
properties="${runtime}/auth/auth101.properties"
password_file="${runtime}/auth_password"
pid_file="${runtime}/auth.pid"
fifo="${runtime}/auth.stdin"
log="${runtime}/logs/auth.log"

read_pid() {
  test -f "${pid_file}" && tr -d '\r\n' <"${pid_file}"
}

if [[ "${action}" == "stop" ]]; then
  pid="$(read_pid || true)"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}" "${fifo}"
    echo "Auth is not running"
    exit 0
  fi
  kill -TERM "${pid}"
  for _ in $(seq 1 50); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL "${pid}"
  fi
  rm -f "${pid_file}" "${fifo}"
  echo "Auth stopped"
  exit 0
fi

if [[ "${action}" == "status" ]]; then
  pid="$(read_pid || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Auth is running with PID ${pid}"
    exit 0
  fi
  echo "Auth is not running"
  exit 1
fi

if [[ "${action}" != "start" ]]; then
  echo "usage: sst/scripts/auth.sh start|stop|status" >&2
  exit 2
fi
case "$(hostname -s)" in
  *login*) echo "Refusing to start Auth on a login node" >&2; exit 2 ;;
esac
for path in "${jar}" "${properties}" "${password_file}" "${runtime}/database/auth101/auth.db"; do
  test -f "${path}" || { echo "Missing ${path}. Build Auth and generate SST state first." >&2; exit 2; }
done
pid="$(read_pid || true)"
if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
  echo "Auth is already running with PID ${pid}"
  exit 0
fi

mkdir -p "$(dirname "${log}")"
: >"${log}"
rm -f "${fifo}"
mkfifo -m 600 "${fifo}"
exec 9<>"${fifo}"
password="$(tr -d '\r\n' <"${password_file}")"
(
  cd "${runtime}/auth"
  exec java -jar "${jar}" -p "${properties}" --password "${password}"
) <"${fifo}" >>"${log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"

for _ in $(seq 1 180); do
  if grep -q "Enter command" "${log}" 2>/dev/null; then
    echo "Auth ready with PID ${pid}. Log: ${log}"
    exit 0
  fi
  kill -0 "${pid}" 2>/dev/null || break
  sleep 0.5
done
tail -n 40 "${log}" >&2 || true
"$0" stop >/dev/null 2>&1 || true
echo "Auth did not become ready" >&2
exit 1

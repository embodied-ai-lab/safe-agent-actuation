#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
iotauth="${root}/third_party/iotauth"
runtime="${root}/runtime/sst"
work="${runtime}/generation-work"
source_copy="${work}/source"
password_file="${runtime}/auth_password"
validity="${SST_RELATIVE_VALIDITY:-20*sec}"
python="${PY:-${root}/.venv/bin/python3}"

case "$(hostname -s)" in
  *login*) echo "Refusing to generate SST state on a login node" >&2; exit 2 ;;
esac
for command in git tar node npm mvn java openssl; do
  command -v "${command}" >/dev/null || { echo "Missing ${command}" >&2; exit 2; }
done
test -f "${iotauth}/examples/generateAll.sh" || { echo "Run make submodules" >&2; exit 2; }
test -x "${python}" || python="python3.11"

mkdir -p "${runtime}/logs"
chmod 700 "${runtime}"
if [[ ! -f "${password_file}" ]]; then
  umask 077
  "${python}" -c 'import secrets; print(secrets.token_urlsafe(24))' >"${password_file}"
fi
password="$(tr -d '\r\n' <"${password_file}")"

rm -rf "${work}"
mkdir -p "${source_copy}"
cleanup() {
  rm -rf "${work}"
}
trap cleanup EXIT

git -C "${iotauth}" archive HEAD | tar -x -C "${source_copy}"
"${python}" "${root}/sst/scripts/render_graph.py" \
  --source "${root}/sst/configs/safe_agent_actuation.graph" \
  --output "${source_copy}/examples/configs/safe_agent_actuation.graph" \
  --relative-validity "${validity}"

generation_log="${runtime}/logs/generate_iotauth.log"
(
  cd "${source_copy}/examples"
  ./generateAll.sh -g configs/safe_agent_actuation.graph -p "${password}"
) >"${generation_log}" 2>&1 || {
  tail -n 60 "${generation_log}" >&2
  exit 1
}

rm -rf "${runtime}/auth" "${runtime}/configs" "${runtime}/credentials" "${runtime}/database"
mkdir -p \
  "${runtime}/auth" \
  "${runtime}/credentials/auth" \
  "${runtime}/credentials/entities/keys" \
  "${runtime}/database"
install -m 0600 "${source_copy}/auth/credentials/ca/CACert.pem" \
  "${runtime}/credentials/auth/CACert.pem"
cp -a "${source_copy}/entity/auth_certs" "${runtime}/credentials/entities/auth_certs"
cp -a "${source_copy}/entity/credentials/keys/net1" "${runtime}/credentials/entities/keys/net1"
cp -a "${source_copy}/auth/databases/auth101" "${runtime}/database/auth101"

"${python}" - "${source_copy}/auth/properties/exampleAuth101.properties" \
  "${runtime}/auth/auth101.properties" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
text = text.replace("../databases/auth101", "../database/auth101")
text = text.replace("../credentials/ca/CACert.pem", "../credentials/auth/CACert.pem")
Path(sys.argv[2]).write_text(text, encoding="utf-8")
PY

"${python}" "${root}/sst/scripts/write_entity_configs.py" --runtime-dir "${runtime}"
"${python}" - "${runtime}/configs" <<'PY'
from pathlib import Path
import sys
from iotauth import IoTAuthContext

for path in sorted(Path(sys.argv[1]).glob("*.config")):
    IoTAuthContext.from_config(path)
    print(f"OK {path.name}")
PY
chmod -R go-rwx "${runtime}/credentials" "${runtime}/database" "${password_file}"
echo "Generated SST runtime state under ${runtime} with RelativeValidity=${validity}"


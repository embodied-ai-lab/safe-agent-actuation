#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
auth_dir="${root}/third_party/iotauth/auth"
build_root="${root}/runtime/sst/auth-build"
build_source="${build_root}/source"
jar="${build_source}/auth/auth-server/target/auth-server-jar-with-dependencies.jar"
log="${root}/runtime/sst/logs/build_auth.log"

case "$(hostname -s)" in
  *login*) echo "Refusing to build Auth on a login node" >&2; exit 2 ;;
esac
command -v java >/dev/null || { echo "Java 11 or newer is required" >&2; exit 2; }
command -v mvn >/dev/null || { echo "Maven is required" >&2; exit 2; }
test -f "${auth_dir}/pom.xml" || { echo "Run make submodules" >&2; exit 2; }
mkdir -p "$(dirname "${log}")"
rm -rf "${build_root}"
mkdir -p "${build_source}"
# Build from an isolated filesystem copy so Maven never writes into the pinned
# dependency.  Metadata and generated/cached files are excluded explicitly;
# no Git command is needed to prepare the copy.
tar \
  --exclude='./.git' \
  --exclude='*/.git' \
  --exclude='*/target' \
  --exclude='*/node_modules' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C "${root}/third_party/iotauth" -cf - . \
  | tar -C "${build_source}" -xf -
mvn -B -q package -DskipTests --file "${build_source}/auth/pom.xml" >"${log}" 2>&1 || {
  tail -n 40 "${log}" >&2
  exit 1
}
test -f "${jar}" || { echo "Auth build did not create ${jar}" >&2; exit 1; }
echo "Built ${jar}"

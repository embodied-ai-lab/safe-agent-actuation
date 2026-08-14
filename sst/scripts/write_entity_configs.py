#!/usr/bin/env python3
"""Write current IoTAuth Python API configs outside the pinned submodule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def relative(source_dir: Path, target: Path) -> str:
    return os.path.relpath(target, source_dir).replace(os.sep, "/")


def render(rows: list[tuple[str, str]]) -> str:
    return "# Generated. Contains paths but no key material.\n" + "\n".join(
        f"{key}={value}" for key, value in rows
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    configs = runtime / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    auth_cert = runtime / "credentials" / "entities" / "auth_certs" / "Auth101EntityCert.pem"
    key_dir = runtime / "credentials" / "entities" / "keys" / "net1"

    common = [
        ("entityInfo.number_key", "1"),
        ("authInfo.id", "101"),
        ("sessionKey.encryptionMode", "AES_128_CBC"),
        ("HmacMode", "on"),
        ("authInfo.pubkey.path", relative(configs, auth_cert)),
        ("auth.ip.address", "127.0.0.1"),
        ("auth.port.number", "22900"),
        ("entity.server.ip.address", "127.0.0.1"),
        ("entity.server.port.number", "23100"),
        ("network.protocol", "TCP"),
    ]
    monitor = [
        ("entityInfo.name", "net1.monitor"),
        ("entityInfo.purpose", json.dumps({"group": "Actuators"}, separators=(",", ":"))),
        ("entityInfo.privkey.path", relative(configs, key_dir / "Net1.MonitorKey.pem")),
        *common,
        ("targetServerInfo.name_1", "net1.actuator"),
        ("targetServerInfo.host_1", "127.0.0.1"),
        ("targetServerInfo.port_1", "23100"),
    ]
    actuator = [
        ("entityInfo.name", "net1.actuator"),
        ("entityInfo.purpose", json.dumps({"group": "Actuators"}, separators=(",", ":"))),
        ("entityInfo.privkey.path", relative(configs, key_dir / "Net1.ActuatorKey.pem")),
        *common,
    ]
    (configs / "monitor.config").write_text(render(monitor), encoding="utf-8")
    (configs / "actuator.config").write_text(render(actuator), encoding="utf-8")
    (configs / "entities.json").write_text(
        json.dumps(
            {
                "registered": ["net1.monitor", "net1.actuator"],
                "unregistered": ["net1.bypass"],
                "auth": {"host": "127.0.0.1", "port": 22900},
                "actuator": {"host": "127.0.0.1", "port": 23100},
                "source": "sst/configs/safe_agent_actuation.graph",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote entity configs under {configs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


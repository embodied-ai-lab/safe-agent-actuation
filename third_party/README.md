# Third-party dependencies

The project uses two clean Git submodules pinned to exact commits. Course code, monitoring, SST adaptation, and actuator enforcement live outside both submodules.

| Path | Official source | Exact commit | License |
|---|---|---|---|
| `third_party/openpi` | `https://github.com/Physical-Intelligence/openpi.git` | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | Apache-2.0 source, with Gemma Terms for applicable model assets and services |
| `third_party/iotauth` | `https://github.com/iotauth/iotauth.git` | `1a40ebece054880e4fe400c1c5880b05ac91ad79` | BSD-2-Clause |

Initialize both from the repository root:

```bash
make submodules
make doctor
```

The equivalent initialization command is:

```bash
git submodule update --init --recursive
```

`make doctor` prints the actual and expected commits. A missing checkout or mismatched commit is a setup failure.

Do not edit, patch, or commit files inside either submodule. Do not configure branch tracking. Do not update a pin for this assignment. If a submodule becomes dirty or moves away from its gitlink, stop and ask the instructor before changing its state.

## OpenPI and π0-FAST

The selected dependency is the official Physical Intelligence OpenPI repository. The course uses its π0-FAST implementation, FAST tokenizer and decoder, WebSocket policy interface, and ALOHA simulator support through a course-owned adapter.

The configured checkpoint is:

```text
gs://openpi-assets/checkpoints/pi0_fast_base
```

It is a fine-tuning base checkpoint, not a published ALOHA simulator expert. The course's zero-shot ALOHA pairing remains a live-validation release blocker. Do not replace it with OpenPI's default ALOHA simulator checkpoint because that checkpoint uses π0 rather than π0-FAST.

`make model-setup` is the only command that resolves model weights. It stores them in the scratch cache named by `OPENPI_DATA_HOME`. Model weights are never committed or placed in the submission ZIP.

OpenPI source is distributed under Apache-2.0. The pinned repository also contains `LICENSE_GEMMA.txt`. Applicable Gemma model assets and services remain subject to those terms.

## Secure Swarm Toolkit

The IoTAuth submodule supplies the current SST Python API and Auth implementation. The Python package is installed directly from:

```text
third_party/iotauth/entity/python
```

The pin includes context-based session-key authorization. The course uses public types such as `IoTAuthContext`, `SecureClient`, `SecureServer`, and `SecureChannel`. No feature-branch checkout or duplicate dependency copy is required.

All generated credentials, keys, Auth databases, properties, PIDs, and logs are written under ignored `runtime/sst/`. Nothing generated is stored inside the submodule.

## Other resolved dependencies

- `gym-aloha==0.1.1` is resolved by the pinned OpenPI environment. The locked wheel SHA-256 is `2698037246dbb106828f0bc229b61007b0a21d5967c72cc373f7bc1083203584`.
- The FAST tokenizer repository is `https://huggingface.co/physical-intelligence/fast` and is resolved by pinned OpenPI.
- MuJoCo and the remaining Python packages are resolved by the pinned environment and lock files.

See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the license files inside each dependency before redistribution. The course repository grants no rights beyond the upstream terms.

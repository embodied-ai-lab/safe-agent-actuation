# Third-party notices

This repository references third-party source through pinned Git submodules. No third-party model weights are committed here.

| Component | Source | Pin | License |
| --- | --- | --- | --- |
| OpenPI | https://github.com/Physical-Intelligence/openpi | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | Apache-2.0, with Gemma terms in `LICENSE_GEMMA.txt` |
| Secure Swarm Toolkit | https://github.com/iotauth/iotauth | `1a40ebece054880e4fe400c1c5880b05ac91ad79` | BSD-2-Clause |
| gym-aloha | OpenPI dependency from PyPI | `0.1.1` | Apache-2.0 |
| pi0-FAST base checkpoint | `gs://openpi-assets/checkpoints/pi0_fast_base` | resolved only by `make model-setup` | Subject to the OpenPI and Gemma notices |
| pi0-FAST-DROID checkpoint | `gs://openpi-assets/checkpoints/pi0_fast_droid` | used only to create the committed Part 1 response; weights are not redistributed | Subject to the OpenPI and Gemma notices |
| FAST tokenizer | https://huggingface.co/physical-intelligence/fast | resolved by pinned OpenPI | Apache-2.0 repository metadata |

Review the license files inside each submodule before redistribution. The course code does not grant permission beyond the terms supplied by each copyright holder.

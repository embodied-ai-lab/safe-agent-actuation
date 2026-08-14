# Model compatibility note

The official OpenPI ALOHA simulator default is `pi0_aloha_sim`, which is pi0 and is not used by this project. The course wrapper constructs `pi0_fast_aloha_sim_course` around `pi0_fast_base` so students can observe the FAST path without changing OpenPI. Physical Intelligence does not publish a pi0-FAST ALOHA simulator expert checkpoint at the pinned commit. This zero-shot pairing is a live-validation release blocker and must never be silently replaced by pi0 or pi0.5.

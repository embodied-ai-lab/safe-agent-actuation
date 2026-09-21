# Model compatibility note

The official OpenPI ALOHA simulator default is `pi0_aloha_sim`, which is pi0
and is not used by the graded project. Physical Intelligence does not publish
a pi0-FAST ALOHA simulator expert at the pinned commit. The course keeps a
`pi0_fast_aloha_sim_course`/`pi0_fast_base` compatibility probe for optional
instructor investigation only; it must never be presented as a validated
ALOHA policy or silently replaced by pi0 or pi0.5.

Part 1 instead reads a committed response captured from the documented
`pi0_fast_droid` inference configuration and
`gs://openpi-assets/checkpoints/pi0_fast_droid`. Its 8-dimensional DROID action
array is for response inspection only. It is never reshaped, executed, or
checked against the 14-dimensional ALOHA limits used by the synthetic graded
fixtures.

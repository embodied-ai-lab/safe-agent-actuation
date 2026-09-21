# Captured pi0-FAST response record

Part 1 reads the pi0-FAST response record through:

```bash
make inspect-actions
```

That graded command validates both committed capture files, verifies their
checksum, and writes `results/inspect-actions/response_summary.json`. It fails
clearly if the response is absent, corrupted, mislabeled, or lacks positive
FAST decode-boundary evidence; configuration dimensions never silently pass as
a real response.

The expected `action_source` is `captured_live_pi0_fast`: a real response from
the published pi0-FAST-DROID expert, recorded once on a Sol GPU and committed
to this directory. The pinned `Policy.infer` response exposes decoded actions,
state, and timing, but not generated token IDs. The record therefore reports
tokens as unavailable with a reason; it does not infer or invent them.

For debugging the configured contract without inspecting model output, run the
separate diagnostic:

```bash
python -m safe_agent_actuation.cli inspect-actions --configuration-only
```

Its `action_source` is `model_configuration_only` and
`response_available` is false. It is not a substitute for the Part 1 command.

## What a captured record contains

When present, this directory holds:

```text
pi0_fast_action_record.json   provenance and response metadata
pi0_fast_action_chunk.npz     the decoded action chunk
```

The JSON records the source type, OpenPI URL and exact dependency revision,
model and checkpoint, exact Hugging Face FAST-tokenizer revision/code checksum,
embodiment, deterministic offline input origin/seed/hash,
actual returned shape, configured and returned horizons, token availability,
decode validation, measured timing, capture environment, and the NPZ SHA-256.
The capture-only tokenizer requires the generated `Action: ` marker and records
the decoded `[10, 8]` array shape at `ExtractFASTActions`, before
unnormalization. This matters because OpenPI's missing-marker zero fallback can
become nonzero after unnormalization; a nonzero final array alone is not decode
evidence. No upstream dependency source is modified.
The offline input comes from OpenPI's seeded
`droid_policy.make_droid_example` helper; it is
not a recorded robot episode. The action chunk was inspected but never
executed, so `actions_executed` is false and `executed_horizon` is zero.

A synthetic fixture is never labeled as a captured record. The loader rejects
any file whose `source_type` is not `captured_live_pi0_fast`. The incompatible
DROID response is also never reshaped into an ALOHA command or checked against
ALOHA limits. Motion analysis uses only the separate, deterministic,
14-dimensional ALOHA fixtures in `data/fixtures/`.

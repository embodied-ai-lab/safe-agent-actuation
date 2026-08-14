# Captured pi0-FAST response record

Part 1 reads the pi0-FAST response record through:

```bash
make inspect-actions
```

That command writes `results/inspect-actions/response_summary.json` and always
runs. It uses whichever source is available and labels it in `action_source`:

| `action_source` | Meaning |
|---|---|
| `captured_live_pi0_fast` | A real pi0-FAST response recorded once and committed to this directory |
| `model_configuration_only` | No captured or live response is present, so the command reports the configured response contract and states that tokens are unavailable |

Both sources answer the Part 1 questions. Neither is ever used to compute a
physical motion score. That calculation runs on the 14-dimensional ALOHA
fixtures in `data/fixtures/`, which are deterministic and clearly labeled
synthetic in their JSON companions.

## What a captured record contains

When present, this directory holds:

```text
pi0_fast_action_record.json   provenance and response metadata
pi0_fast_action_chunk.npz     the decoded action chunk
```

The JSON records the source type, the OpenPI URL and commit, the model family,
the exact configuration and checkpoint, the embodiment, the task prompt, token
availability and count, decode status, the decoded model-space shape, the final
action shape, the executed horizon, timing, the capture date and environment,
the SHA-256 of the chunk, and the known cross-embodiment limitation.

A synthetic fixture is never labeled as a captured record. The loader rejects
any file whose `source_type` is not `captured_live_pi0_fast`.

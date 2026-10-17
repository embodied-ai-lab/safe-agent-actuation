# Safe Agent Actuation: Analyzing and Authorizing VLA Actions

CSE 494/598 · 4 points (common) + 1 point (CSE 598 extension) ·
estimated 4–6 hours after setup

Record every answer in `submission/answers.md` (copy
[`submission/answers_template.md`](submission/answers_template.md)). Keep
answers short—numbers, one-liners, and compact tables are better than essays.
For each experimental claim, cite the result file and field that support it.

Vision language action (VLA) models turn a camera image and a text instruction
into robot motion. This assignment builds a safety layer in front of one such
model. You inspect token availability and verified FAST decode-boundary evidence
in a genuine pi0-FAST response, distinguish token representations from decoded
physical actions, measure how much the robot is about to move, classify the
motion as low risk or high risk, and require context-based authorization before
any high-risk action reaches the recording actuator harness. The
authorization and delivery path uses the Secure Swarm Toolkit (SST) from the
IoTAuth project. The graded actuator is a deterministic EXECUTE/HOLD recording
harness; it does not simulate robot dynamics or claim task success. You never
control a real robot.

This work is graded out of **4 common points for CSE 494 and CSE 598**, plus
**1 additional point for CSE 598 only**. The additional point is Part 5. Submit
one ZIP for the whole assignment. There is no second upload for the CSE 598
point.

ASU Sol is the primary platform. Build the container from
`containers/Apptainer.def` using the CPU-allocation workflow in
[README.md](README.md). All graded parts use CPU compute and the committed
capture; no GPU, live model server, or model download is required. Do not run
compute work on a login node.

## Learning objectives

By the end of this assignment you can:

1. Explain why physical risk must be assessed after FAST decoding, not from raw
   token identifiers.
2. Inspect a pi0-FAST response record and report token availability, token count,
   and the decoded and final action shapes.
3. Compute a normalized joint motion score from a decoded action chunk and the
   observed robot state.
4. Classify an action against a configurable threshold and request authorization
   only when it is needed.
5. Protect a high-risk action, verify it at the actuator, and reject anything
   denied, malformed, expired, or unauthenticated.
6. Reason about the tradeoff between the threshold, the authorization rate, and
   latency, and about the security window created by session-key reuse.

## What you must not edit

Do not edit either submodule. OpenPI lives in `third_party/openpi` and IoTAuth
lives in `third_party/iotauth`. All of your work is in the course package under
`src/safe_agent_actuation/`, in `configs/`, and in your report. The graded code
regions are marked in the source with `TODO(student)` comments.

## Two data sources, and why

This project uses two clearly separated sources.

The **pi0-FAST response record** is what Part 1 reads. It contains a genuine
response captured by the instructor from the documented `pi0_fast_droid`
expert and OpenPI's documented example observation. It records whether the
standard API exposed action tokens, the actual returned action shape, decode
status, timing, input origin, and checksum. `make inspect-actions` validates
the committed JSON/NPZ pair, writes
`results/inspect-actions/response_summary.json`, and labels its
`action_source`. A missing, partial, or corrupted capture is an error; an
explicit configuration-only diagnostic is not a substitute for Part 1.

The **ALOHA action fixtures** under `data/fixtures/` drive Parts 2 through 5.
They are deterministic 14-dimensional ALOHA command chunks with known physical
limits, and each one ships a JSON companion recording that it is synthetic. The
physical motion score is computed on these fixtures because the score needs
real physical ranges for every commanded dimension.

Keep the two apart in your report. A fixture is never model output, and the
response record is never the thing you compute a motion score from.

## Part 0: Setup (required, ungraded)

Follow [README.md](README.md#run-on-asu-sol): create a private group repository
from the public template, clone it into scratch, initialize the submodules,
build `/scratch/$USER/safe-agent-actuation.sif` in a CPU allocation, and define
the `lab` wrapper. Use `lab` for project commands on Sol unless you are already
inside the container. On local Linux, omit the prefix.

```bash
lab make setup
lab make doctor
lab make test-baseline
lab make lint
```

Confirm that doctor reports the expected dependency pins, the baseline tests
pass, and Ruff is clean. Record the compute job ID and container checksum in
your answers.

Run `lab make test-acceptance` once to inspect the failures for the five
`TODO(student)` regions. These are intentional in the untouched student
template and do not block Part 1. After completing Parts 2 and 3,
`lab make test-acceptance` and `lab make test-offline` must both pass. Submit
the full batch workflow only after those implementations are complete.

## Part 1: Tokens and decoded actions (1 point)

Run `lab make inspect-actions` and read
`results/inspect-actions/response_summary.json`.

Report:

1. The `action_source` of the record you read, and what that source means.
   Also report the model configuration, checkpoint, OpenPI revision, and input
   origin recorded with it.
2. Whether action tokens are available, and the token count when they are.
   When they are not available, state that plainly and give the reason the
   record provides.
3. The actual action array shape returned by the policy and any separate
   decoded-model shape the API genuinely exposes. Do not infer an internal
   shape from the final array.
4. The configured horizon, the captured response horizon, and the fact that no
   captured action was executed.
5. Separately, how the 14 ALOHA fixture dimensions used in Parts 2 through 5
   map to arm joints and grippers. Use `configs/motion/aloha_limits.json`; do
   not describe the 8-dimensional DROID capture as ALOHA.
6. Why physical risk is assessed after FAST decoding rather than from raw token
   identifiers. Explain what a token identifier is and is not.
7. Any inference and decode latency the record reports, or a note that the
   standard response does not separately expose decode timing.
8. The recorded action-array checksum and why it matters.

You are not graded on a specific token sequence, and you are not penalized for
token identifiers being unavailable from the pinned standard response. You are
graded on reading the genuine record correctly and explaining the difference
between tokens and decoded actions. Configuration-only output does not satisfy
this part.

## Part 2: Motion-risk analysis (1 point)

Complete the three motion-analysis `TODO(student)` regions in
[`src/safe_agent_actuation/motion.py`](src/safe_agent_actuation/motion.py):
`_intra_chunk_motion`, `_inter_state_motion`, and `_score_and_classify`.

You must:

1. Compute the normalized intra-chunk motion for each dimension.
2. Compute the normalized motion from the current observed state to the first
   action.
3. Combine them with the maximum across dimensions and identify the dimension
   that produced the maximum.
4. Apply the threshold with the `>=` comparison so a score equal to the
   threshold is high risk.
5. Block invalid input with an explicit reason rather than letting it reach the
   actuator.
6. Analyze the low-risk, high-risk, and boundary fixtures and report their
   scores and classifications.

Run `lab make low-risk` and confirm the trusted low-risk path executes without
SST. Read `results/low-risk/action_record.json` for the motion analysis and
actuator decision.

## Part 3: Authorization and actuator enforcement (1 point)

Complete the two authorization and enforcement `TODO(student)` regions in
[`src/safe_agent_actuation/scenarios.py`](src/safe_agent_actuation/scenarios.py).
Then check the completed implementation:

```bash
lab make test-acceptance
lab make test-offline
lab make lint
```

Run the [README's real SST block](README.md#4-run-the-real-sst-experiments)
in your CPU allocation. It builds Auth, generates runtime credentials, starts
Auth, runs the live tests and Parts 3-4, and stops Auth on exit. Keep Auth and
its clients in that single container session. Commands below use plain `make`
because they run inside that block.

Demonstrate:

1. A high-risk action with the allowed context is authorized, protected,
   transferred, verified at the actuator, and executed: `make high-risk-allow`.
2. A high-risk action with the denied context is refused, and the recording
   harness records HOLD with no execution: `make high-risk-deny`.
3. A direct unauthenticated attempt on the actuator port is rejected:
   `make bypass`.
4. The actuator holds its pose on any denied, malformed, expired, or
   unauthenticated input.
5. A correct explanation of the security boundary: what authentication and
   authorization establish here and what they do not.

You must not edit OpenPI or IoTAuth to complete this part.

## Part 4: Multi-threshold authorization and latency evaluation (1 point)

This is a controlled multi-threshold evaluation: vary the risk threshold while
keeping the committed fixture/context pairs, order, repetition count, and
per-case cache scope for every threshold. Do not run new model inference per
threshold. Run `make threshold-sweep` before leaving the same container session
used for Part 3, while Auth is still running; the accepted
command uses real SST for high-risk classifications and the trusted local path
for low-risk classifications. `make threshold-sweep-offline` is only a
development diagnostic.

Report:

1. Each threshold and the resulting high-risk rate, using all successfully
   classified measured trials as the denominator.
2. The authorization-path activity rate at each threshold, using the same
   classified denominator.
3. Actual key-request attempts, successful key requests, and key reuse.
4. Allowed, denied, and infrastructure-error counts; transfer attempts,
   successful transfers, and failed transfers; and the failed-trial count.
5. The mean and observed maximum Monitor-to-actuator latency. The timed
   interval starts when the Monitor receives the action and ends only after the
   actuator harness records EXECUTE or HOLD. It includes applicable analysis,
   authorization, handshake, protected delivery, payload checks, and actuator
   enforcement. Every measured row is retained. Confirmed EXECUTE/HOLD
   intervals, including policy denials that end in HOLD, contribute to the
   latency aggregates. A failure before a confirmed decision instead records
   `elapsed_until_failure_ms`; it cannot supply a complete Monitor-to-actuator
   interval and is not used in the latency mean or observed maximum.
6. One table or the generated plot.
7. A concise interpretation that keeps motion classification, policy
   authorization, and physical safety distinct.

Do not force latency to be monotonic with threshold. The observed maximum is a
sample statistic, not a formal worst-case bound.

## Part 5: CSE 598 extension (1 point)

Evaluate how session-key validity interacts with a context change. After the
common SST block finishes and stops Auth, run `lab make grad-key-validity` in
the CPU allocation. The extension requires the existing Auth build and manages
its own runtime generation and Auth processes for each validity period. Do not
run it concurrently with the common experiments.

Include:

1. A hypothesis about how relative validity affects the security window and the
   number of key requests.
2. Results for the exact ordered relative-validity periods 5, 20, and 40
   seconds.
3. A controlled baseline: authorize in the allowed context, reuse the valid key,
   change the context to a denied state, then observe behavior while the key is
   still valid and after it expires.
4. Absolute validity, relative validity, and the recorded first-use time.
5. Attempted and successful key requests, key reuse, transfer attempts, and
   successful transfers.
6. Authorization, handshake, and protected-transfer latency.
7. An interpretation of the security window created by key reuse. A context
   change does not immediately revoke a still-valid issued key in this
   experiment.
8. One limitation of your evaluation.

The extension is part of the same ZIP as the common assignment. No second upload
is required.

## Deliverables

Put your written answers in `submission/answers.md` and fill in every field of
the answers template. Build the ZIP inside a CPU allocation using the `lab`
wrapper from the README. Choose the command for your course section:

```bash
# CSE 494
lab make submission GROUPID=<groupid>
# CSE 598
lab make submission GROUPID=<groupid> REQUIRE_GRAD=1
unzip -l submission/group<groupid>_safe-agent-actuation.zip
```

Upload only that ZIP to Canvas. The CSE 598 form fails if any required Part 5
artifact is absent.
The ZIP includes your report, your edited course code, your configuration, your
fixtures, and the applicable required JSON, JSONL, CSV, NPZ, and PNG results.
JSONL is required only for the CSE 598 Part 5 experiment.

| Part | Required result directory |
|---|---|
| 1 | `results/inspect-actions/` |
| 2 | `results/low-risk/` |
| 3 | `results/high-risk-allow/`, `results/high-risk-deny/`, `results/bypass/` |
| 4 | `results/threshold-sweep/` |
| 5 (CSE 598) | `results/grad-key-validity/` |

The packager includes current files whether or not they are committed. Do not
include model weights, the container image, environments, caches, or SST
credentials. Run `lab make clean` only after inspecting and saving the ZIP;
cleaning removes the results needed to rebuild it.

## Rubric

| Part | Points | Focus |
| --- | --- | --- |
| 0 | 0 | Setup and baseline, required |
| 1 | 1 | Tokens versus decoded actions |
| 2 | 1 | Motion-risk analysis |
| 3 | 1 | Authorization and actuator enforcement |
| 4 | 1 | Threshold and latency tradeoff |
| 5 | 1 | CSE 598 session-key validity extension |

Common total is 4 points for both CSE 494 and CSE 598. Part 5 adds 1 point for
CSE 598 only.

Partial credit is available on each part for correct work that stops short of
the full result. Code that does not run receives limited credit, so keep a
runnable baseline as you go.

## Group work

You may work in a group. One member creates a private repository on GitHub from
the public template, adds partners, and the group clones that private copy on
Sol. Put the group ID and every member
name in `submission/answers.md`. One member uploads the single ZIP. All members
receive the same grade.

## Final submission warning

Canvas grades the ZIP you upload, not a newer copy on GitHub or Sol. Inspect the
ZIP before you submit. Rebuild and re-upload if you change anything after
building it.

## Generative AI

You may use generative AI as an assistant for clarifying concepts, debugging,
organizing ideas, or improving writing. You may not rely on it to complete the
project without understanding the work. Every group member is responsible for
reviewing, testing, and understanding the submitted code, experiments, results,
and written answers, and must be able to explain the implementation and design
choices. Work that relies on generative AI without demonstrated understanding
may receive reduced credit for the affected parts.

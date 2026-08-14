# Safe Agent Actuation: Analyzing and Authorizing VLA Actions

Vision language action (VLA) models turn a camera image and a text instruction
into robot motion. This assignment builds a safety layer in front of one such
model. You observe how a pi0-FAST policy produces action tokens, separate those
raw tokens from the decoded physical action, measure how much the robot is about
to move, classify the motion as low risk or high risk, and require context based
authorization before any high-risk action reaches a simulated actuator. The
authorization and delivery path uses the Secure Swarm Toolkit (SST) from the
IoTAuth project. Everything runs in simulation. You never control a real robot.

This work is graded out of **4 common points for CSE 494 and CSE 598**, plus
**1 additional point for CSE 598 only**. The additional point is Part 5. Submit
one ZIP for the whole assignment. There is no second upload for the CSE 598
point.

Default platform is ASU Sol. Use the provided container and Slurm scripts. Do
not run compute work on a login node.

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

The **pi0-FAST response record** is what Part 1 reads. It describes one
pi0-FAST response: whether action tokens are exposed, how many there are, the
decoded model-space shape, the final action shape, and timing. `make
inspect-actions` writes it to `results/inspect-actions/response_summary.json`
and labels its `action_source`.

The **ALOHA action fixtures** under `data/fixtures/` drive Parts 2 through 5.
They are deterministic 14-dimensional ALOHA command chunks with known physical
limits, and each one ships a JSON companion recording that it is synthetic. The
physical motion score is computed on these fixtures because the score needs
real physical ranges for every commanded dimension.

Keep the two apart in your report. A fixture is never model output, and the
response record is never the thing you compute a motion score from.

## Setup

See `README.md` for the full Sol workflow. In short:

1. Clone your group template on Sol and initialize the submodules with
   `make submodules`.
2. Set `SIF` to the course image, or build one with
   `slurm/build_container.sbatch`.
3. Inside the container run `make setup` then `make doctor`.
4. Run `make test-offline` and `make lint` to confirm a clean baseline.

## Part 0: Setup (required, ungraded)

1. Set up the Sol environment and initialize dependencies.
2. Confirm `make doctor` reports the expected submodule pins.
3. Run `make test-offline`. The provided component tests pass. The graded
   acceptance tests fail until you complete the `TODO(student)` regions, which
   is the expected starting state.
4. Run `make lint` and confirm Ruff is clean.

## Part 1: Tokens and decoded actions (1 point)

Run `make inspect-actions` and read
`results/inspect-actions/response_summary.json`.

Report:

1. The `action_source` of the record you read, and what that source means.
2. Whether action tokens are available, and the token count when they are.
   When they are not available, state that plainly and give the reason the
   record provides.
3. The decoded model action shape and the final action shape.
4. The executed horizon and how it relates to the predicted horizon.
5. How the 14 ALOHA output dimensions map to arm joints and grippers. Use
   `configs/motion/aloha_limits.json`.
6. Why physical risk is assessed after FAST decoding rather than from raw token
   identifiers. Explain what a token identifier is and is not.
7. Any inference and decode latency the record reports, or a note that the
   record does not include timing.

You are not graded on a specific token sequence, and you are not penalized for
token identifiers being unavailable. You are graded on reading the record
correctly and explaining the difference between tokens and decoded actions.

## Part 2: Motion-risk analysis (1 point)

Complete the motion-analysis `TODO(student)` regions in
`src/safe_agent_actuation/`.

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

Run `make low-risk` and confirm the trusted low-risk path executes without SST.

## Part 3: Authorization and actuator enforcement (1 point)

Complete the authorization and enforcement `TODO(student)` regions, then run the
real SST path on a compute node.

Demonstrate:

1. A high-risk action with the allowed context is authorized, protected,
   transferred, verified at the actuator, and executed: `make high-risk-allow`.
2. A high-risk action with the denied context is refused, and no executable
   action reaches the simulator: `make high-risk-deny`.
3. A direct unauthenticated attempt on the actuator port is rejected:
   `make bypass`.
4. The actuator holds its pose on any denied, malformed, expired, or
   unauthenticated input.
5. A correct explanation of the security boundary: what authentication and
   authorization establish here and what they do not.

You must not edit OpenPI or IoTAuth to complete this part.

## Part 4: Threshold and latency tradeoff (1 point)

Use the fixed committed fixture set for every threshold. Do not run new model
inference per threshold. Run `make threshold-sweep`.

Report:

1. Each threshold and the resulting high-risk rate.
2. The authorization active rate at each threshold.
3. The allowed and denied counts.
4. The average and worst monitor-to-actuator latency.
5. One table or the generated plot.
6. A concise interpretation that keeps motion classification, policy
   authorization, and physical safety distinct.

## Part 5: CSE 598 extension (1 point)

Evaluate how session-key validity interacts with a context change. Run
`make grad-key-validity` on a compute node with SST available.

Include:

1. A hypothesis about how relative validity affects the security window and the
   number of key requests.
2. Results for three validity periods.
3. A controlled baseline: authorize in the allowed context, reuse the valid key,
   change the context to a denied state, then observe behavior while the key is
   still valid and after it expires.
4. The key-request and key-reuse counts.
5. Authorization and transfer latency.
6. An interpretation of the security window created by key reuse.
7. One limitation of your evaluation.

The extension is part of the same ZIP as the common assignment. No second upload
is required.

## Deliverables

Put your written answers in `submission/answers.md`. Copy
`submission/answers_template.md` and fill in every field. Build the ZIP with
`make submission GROUPID=<your-group-id>` and upload only that ZIP to Canvas.
The ZIP includes your report, your edited course code, your configuration, your
fixtures, and the required JSON, JSONL, CSV, NPZ, and PNG results.

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

You may work in a group. One member creates the group template repository on
Sol, adds partners, and the group clones it. Put the group ID and every member
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

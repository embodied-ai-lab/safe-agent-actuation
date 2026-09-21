# Safe Agent Actuation: Answers

Copy this file to `submission/answers.md` and fill in every field. Do not rename
the copy. CSE 494 builds the ZIP with `make submission
GROUPID=<your-group-id>`; CSE 598 adds `REQUIRE_GRAD=1` so missing Part 5
artifacts fail packaging.

## Group and run metadata

| Field | Value |
| --- | --- |
| Group ID | |
| Member names | |
| Course section (CSE 494 or CSE 598) | |
| Platform | |
| Sol job ID (container build) | |
| Sol job ID (GPU or compute run) | |
| OpenPI commit | |
| IoTAuth commit | |
| Model configuration | |
| Checkpoint URI | |
| Capture input origin | |
| Capture action SHA-256 | |
| Container SIF SHA-256 | |
| Result directories used | |

## Part 1: Tokens and decoded actions

Read `results/inspect-actions/response_summary.json`.

1. The `action_source`, model/checkpoint, OpenPI revision, and input origin:

2. Were action tokens available? Token count if yes, or the reason given if no:

3. Actual captured action shape and any separately exposed decoded-model
   shape (do not infer an internal shape the API did not expose):

4. Configured and captured horizons, and whether any captured action was
   executed:

5. Separately, the mapping of the 14-dimensional synthetic ALOHA fixtures to
   arm joints and grippers (the DROID capture is not ALOHA):

6. Why physical risk is assessed after decoding rather than from raw tokens.
   What is a token identifier, and what is it not?

7. Inference timing and separately exposed decode timing, or the recorded API
   limitation:

8. Captured action checksum and what its successful validation establishes:

## Part 2: Motion-risk analysis

1. How you computed the normalized intra-chunk motion:

2. How you computed the current-state-to-first-action motion:

3. How you combined them and identified the maximum dimension:

4. How the threshold comparison treats a score equal to the threshold:

5. How invalid input is blocked:

6. Scores and classifications for the low-risk, high-risk, and boundary
   fixtures:

| Fixture | Motion score | Max dimension | Classification |
| --- | --- | --- | --- |
| low_risk | | | |
| high_risk | | | |
| boundary | | | |

## Part 3: Authorization and actuator enforcement

1. High-risk allowed run result (authorization, transfer, actuator decision):

2. High-risk denied run result (authorization, actuator decision):

3. Bypass run result (what the unauthenticated client observed):

4. Evidence the actuator held its pose on denied or unauthenticated input:

5. Security-boundary explanation. What do authentication and authorization
   establish here, and what do they not establish?

## Part 4: Multi-threshold authorization and latency evaluation

Fill from `results/threshold-sweep/summary.json`. Rates use the summary's
documented denominators. Do not omit denied or failed measured trials.

| Threshold | Classified trials | High-risk rate | Auth-path rate | Request attempts | Successful requests | Reuses |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02 | | | | | | |
| 0.05 | | | | | | |
| 0.10 | | | | | | |
| 0.20 | | | | | | |

| Threshold | Allowed | Denied | Auth errors | Transfer success/failure | Failed trials | Mean latency (ms) | Observed max (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | | | | | | | |
| 0.05 | | | | | | | |
| 0.10 | | | | | | | |
| 0.20 | | | | | | | |

Latency definition used by the experiment and why “observed maximum” is not a
formal worst-case bound:

Interpretation. Keep motion classification, policy authorization, and physical
safety distinct:

## Part 5: CSE 598 extension

Complete only if you are in CSE 598. Fill from `results/grad-key-validity/`.

1. Hypothesis:

2. Results for the exact 5, 20, and 40 second relative-validity periods:

| Relative / absolute validity | First use | Request attempts / successes | Key reuses | Transfers attempted / successful | Remaining validity at context change | Denied after expiry |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |
| | | | | | | |
| | | | | | | |

3. Controlled baseline description:

4. Authorization and transfer latency:

5. Security-window interpretation:

6. One limitation of your evaluation:

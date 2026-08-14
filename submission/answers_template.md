# Safe Agent Actuation: Answers

Copy this file to `submission/answers.md` and fill in every field. Do not rename
the copy. Build your ZIP with `make submission GROUPID=<your-group-id>`.

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
| Container SIF SHA-256 | |
| Result directories used | |

## Part 1: Tokens and decoded actions

Read `results/inspect-actions/response_summary.json`.

1. The `action_source` of the record, and what that source means:

2. Were action tokens available? Token count if yes, or the reason given if no:

3. Decoded model action shape and final action shape:

4. Executed horizon and its relation to the predicted horizon:

5. Mapping of the 14 ALOHA output dimensions to arm joints and grippers:

6. Why physical risk is assessed after decoding rather than from raw tokens.
   What is a token identifier, and what is it not?

7. Inference and decode latency, or a note that the record has no timing:

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

## Part 4: Threshold and latency tradeoff

Fill from `results/threshold-sweep/`.

| Threshold | High-risk rate | Authorization active rate | Allowed | Denied | Avg monitor-to-actuator (ms) | Worst (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02 | | | | | | |
| 0.05 | | | | | | |
| 0.10 | | | | | | |
| 0.20 | | | | | | |

Interpretation. Keep motion classification, policy authorization, and physical
safety distinct:

## Part 5: CSE 598 extension

Complete only if you are in CSE 598. Fill from `results/grad-key-validity/`.

1. Hypothesis:

2. Results for three validity periods:

| Relative validity | Key requests | Key reuses | Security-window sends | Denied after expiry |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |
| | | | | |

3. Controlled baseline description:

4. Authorization and transfer latency:

5. Security-window interpretation:

6. One limitation of your evaluation:

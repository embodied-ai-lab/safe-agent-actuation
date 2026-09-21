# Security model

This project demonstrates context-based authorization and actuator enforcement
for controlled VLA-action experiments. The graded path uses a deterministic
EXECUTE/HOLD recording harness. An optional headless MuJoCo ALOHA integration
is kept separate from grading. Nothing in this repository is approved for a
real robot.

## Potentially untrusted components

Treat these inputs and components as potentially malicious, compromised, malformed, or simply wrong:

- the natural-language command source
- the π0-FAST model
- the model-generated token sequence
- the decoded action chunk before monitoring
- any process that tries to send a command directly to the Actuator

The VLA is inside the problem, not inside the trusted computing base. A valid model response still requires structural checks and monitoring before controlled actuation.

## Trusted components for this project

The demonstrated guarantee assumes these components behave correctly:

- Auth
- the Joint Motion Analyzer
- the Threshold Gate
- the deterministic Context Builder and context source
- the Monitor's SST logic
- the low-risk monitor-to-actuator capability
- actuator-side verification and decryption
- the controlled current-state fixture and recording harness used for evaluation

Compromise of any trusted component is outside the demonstrated guarantee.

## Enforcement paths

Part 1 validates a genuine pi0-FAST-DROID response and its FAST decode-boundary
evidence, but does not execute or reinterpret that eight-dimensional DROID
output as ALOHA. Parts 2 through 5 start from explicitly synthetic, already
decoded-space 14-dimensional ALOHA fixtures. The Monitor validates those
fixtures before computing motion risk from the executable prefix, current
state, and configured physical ranges.

A valid low-risk action has `motion_score < threshold`. It does not request an SST session key, but it still travels in a trusted low-risk envelope that binds the checked action and decision. A caller cannot gain execution merely by labeling its own action low-risk.

A valid high-risk action has `motion_score >= threshold`. It requires a
context-authorized SST session key and an authenticated protected channel to
the Actuator. The Actuator verifies and decrypts the protected payload before
it can record or forward an EXECUTE decision.

An invalid action is neither low-risk nor high-risk. It is blocked.

## Fail-closed enforcement behavior

The Monitor withholds transfer after Auth denial or a Monitor-side failure.
The Actuator rejects protected input that reaches it with:

- missing or unknown input
- malformed protected payloads
- wrong action shapes
- empty action chunks
- NaN or infinity
- out-of-range commands
- invalid physical ranges
- authentication or integrity failures
- unknown, not-yet-valid, or expired keys
- a direct unauthenticated bypass

Either path records HOLD in the graded harness with the validated current
commanded pose and records no execution. The optional MuJoCo adapter separately
checks that a requested hold pose matches its current observation. Neither
behavior establishes that holding a pose is universally safe on physical
hardware.

Decode failure also produces no executable action. A normalized all-zero fallback is not a safe hold because later unnormalization can turn it into a nonzero physical command.

## What SST establishes

Within the tested configuration, SST authenticates registered entities, authorizes a request context, and protects command confidentiality and integrity on the high-risk path. The Actuator accepts a protected command only after the SST channel and application payload checks succeed.

SST does not establish that:

- the VLA selected a physically correct action
- the motion heuristic detects every hazard
- a trusted context source reports physical truth
- a registered Monitor or Actuator is uncompromised
- the robot, environment, or sensors are free from physical attack
- service will remain available during denial of service

Do not describe this design as zero trust. Do not claim that authentication is a proof of physical safety.

## Motion-score limits

The motion score normalizes each joint or gripper command by its configured physical range. It combines within-chunk movement with the jump from current state to the first proposed command. This makes the policy transparent and testable, but it is only a risk heuristic.

Low motion can be dangerous near a person or obstacle. High motion can be appropriate in an approved context. The score does not model velocity, acceleration, torque, force, collision geometry, payload, task intent, or environmental dynamics.

## Context policy

The deterministic context contains exactly:

- `location`
- `nearby_people`
- `permitted_time`
- `task_name`
- `robot_id`

The course maps these fields to the exact names used by the pinned SST context verifier. Missing, unknown, incorrectly typed, or incorrectly formatted fields are denied.

The default allowed case is the ASU Robotics Lab, no nearby people, a time from 08:00 through 18:00, task `transfer_cube`, and robot `aloha-sim-01`. The default denied case changes `nearby_people` from 0 to 1.

The deterministic provider makes the experiment repeatable. It does not sense real people or attest a physical location.

## Session-key validity

Session-key reuse reduces authorization latency but creates a security window
after context changes. The CSE 598 experiment measures that tradeoff with three
relative-validity periods, actual elapsed time, and distinct counts for request
attempts, successful requests, reuse, and protected transfers.

The pinned IoTAuth API begins relative validity at first use. Its Python cache
is indexed by key ID rather than the complete authorization purpose. Course
code tracks the requested purpose, context, first use, reuse, and expiration
explicitly. The basic denied scenario starts without a valid allowed-context
key so it measures a fresh denial. The graduate experiment intentionally
preserves the post-context-change window of a still-valid issued key; it does
not claim immediate revocation.

## Data and secret handling

Generated SST credentials, Auth databases, properties, passwords, PID files, and logs belong only under ignored `runtime/sst/`. Model weights and package caches belong in `/scratch/$USER`. The repository and submission ZIP must not contain either category.

Action records may contain token IDs only when an upstream response actually
exposes them. They may also contain decoded-action summaries, contexts, key
identifiers, decisions, and timing. They must never contain session-key bytes,
private keys, passwords, credential files, or other secret material.

Use the submission builder rather than assembling the ZIP by hand. It uses a fixed allowlist and excludes both submodules, runtime state, model caches, SIF images, virtual environments, credentials, databases, and prior archives.

## Out of scope

The required project does not claim protection against:

- compromised trusted components
- operating-system or hypervisor compromise
- physical sensor spoofing or tampering
- adversarial examples
- side channels
- denial of service
- multi-host network attacks beyond the tested SST path
- comprehensive replay across separate sessions beyond the tested channel and key-validity behavior
- unsafe behavior on real hardware

Report an unexpected actuation, leaked secret, authentication bypass, or cleanup failure to the instructor. Stop the experiment and preserve only secret-free logs needed to diagnose the problem.

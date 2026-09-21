# Safe Agent Actuation: Analyzing and Authorizing VLA Actions

> This template repository is designed as a project lab for the combined course of
> [CSE 494](https://catalog.apps.asu.edu/catalog/classes/classlist?keywords=85268&searchType=all&term=2267)
> and [CSE 598](https://catalog.apps.asu.edu/catalog/classes/classlist?keywords=87933&searchType=all&term=2267),
> "Topic: Intelligent and Safe Cyber-Physical Systems" (ISCPS in short),
> at Arizona State University (ASU) in Fall 2026.
> If you have any questions about this project, please get in touch with the
> instructor, [Hokeun Kim](https://hokeun.github.io/), via
> [hokeun@asu.edu](mailto:hokeun@asu.edu).

A vision language action (VLA) model turns a camera image and a text
instruction into robot motion. This project puts a safety layer in front of one.
You inspect a genuine recorded pi0-FAST response, measure how far controlled
ALOHA action fixtures would move, classify that motion as low risk or high
risk, require context-based authorization from the Secure Swarm Toolkit (SST)
before a high-risk action may pass, and enforce the decision at an actuator
harness. Nothing here controls a real robot.

```text
instructor pi0-FAST DROID inference           synthetic ALOHA fixtures
                 |                                      |
       captured response (Part 1)                       v
                 |                                  Monitor
       inspect only; no execution          motion analysis + threshold
                                                |                |
                                      trusted low-risk      real-SST high-risk
                                                |                |
                                                +-------v--------+
                                                Actuator harness
                                          validate, EXECUTE, or HOLD
```

pi0-FAST internally emits discrete codes, and the FAST decoder turns them into
a continuous trajectory. The pinned policy's standard response does not expose
those token IDs, so the captured record says they are unavailable rather than
inventing them. A token identifier is not a joint command and is not safe or
unsafe on its own. Parts 2 through 5 therefore score explicitly synthetic,
decoded-space ALOHA fixtures against a controlled current state.

| Property | Value |
|---|---|
| Model family | pi0-FAST |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| IoTAuth commit | `1a40ebece054880e4fe400c1c5880b05ac91ad79` |
| Graded ALOHA fixture dimension | 14 |
| Graded ALOHA fixture horizon | 10 |
| Analyzed/executed fixture prefix | all 10 fixture steps |
| Default threshold | 0.10, compared with `>=` |

The Part 1 capture has the embodiment, output shape, and horizon reported by
its own metadata. It is never reshaped into ALOHA commands. The graded
EXECUTE/HOLD path uses `RecordingSimulator`, a deterministic actuator spy that
establishes forwarding and enforcement behavior—not physical trajectory safety
or task success. The optional live ALOHA/MuJoCo integration remains separate
from the graded commands.

## Start here

Read [ASSIGNMENT.md](ASSIGNMENT.md) for the graded parts, required results, and
the rubric. Read [SECURITY.md](SECURITY.md) before you implement the Monitor or
the Actuator. This file covers setup and the command surface.

## Create your private repository

1. Open this GitHub template repository.
2. Select **Use this template**, then **Create a new repository**.
3. Create it under your group's account with a clear name, for example
   `group05-safe-agent-actuation`.
4. Set the visibility to **Private**. Do not publish course work publicly.
5. Add only your project partners as collaborators.
6. On Sol, clone your new private repository into scratch, not this template.

```bash
cd /scratch/$USER
git clone --recurse-submodules git@github.com:<OWNER>/<PRIVATE_REPOSITORY>.git
cd <PRIVATE_REPOSITORY>
make submodules
```

Put your group ID and every member name in `submission/answers.md`.

## Run on ASU Sol

[Sol](https://docs.rc.asu.edu/about) is the default platform. Login nodes are
for cloning, editing, and submitting jobs. Run the environment setup, Auth, and
every experiment inside a compute allocation. The setup, Auth, key-validity,
and Slurm launchers refuse hostnames containing `login`; the lightweight Python
commands rely on you to follow the same compute-allocation rule.

As in the other Fall 2026 ISCPS project labs, the batch script carries the
course account and QoS. Submit it directly:

```bash
sbatch slurm/run_experiments.sbatch
```

CSE 598 groups add the extension to the same run:

```bash
sbatch --export=ALL,RUN_GRAD_EXTENSION=1 slurm/run_experiments.sbatch
```

The job produces results, not a submission archive. After the job succeeds and
you complete `submission/answers.md`, run the explicit `make submission`
command shown below. CSE 598 work goes in that same ZIP; there is no second
upload.

### Container image

Point `SIF` at the course image announced in Canvas:

```bash
export SIF=<COURSE_PROVIDED_SIF>
```

If no image is provided to you, build one once in a compute allocation:

```bash
sbatch slurm/build_container.sbatch
```

The build writes the image and its SHA-256 to your scratch space. Model
weights, credentials, and results are never stored in the image.

For an interactive allocation:

```bash
interactive -A class_cse494598fall2026 -p public -q class \
    -t 120 -c 8 --mem=32G
module load apptainer/1.4.5
cd /scratch/$USER/<PRIVATE_REPOSITORY>
export SIF=<COURSE_PROVIDED_SIF>
lab() {
    apptainer exec \
        --bind "$PWD:$PWD" --bind "/scratch/$USER:/scratch/$USER" \
        --pwd "$PWD" "$SIF" "$@"
}
lab make setup
lab make doctor
lab make test-baseline
lab make lint
```

## Command guide

Run `make help` for the live list.

| Command | Purpose |
|---|---|
| `make submodules` | Initialize the exact OpenPI and IoTAuth pins |
| `make setup` | Create the Python environment on a compute node |
| `make doctor` | Check host, Python, submodule pins, and tools |
| `make test-baseline` | Run supplied infrastructure checks; passes in the untouched starter |
| `make test-acceptance` | Run the completed-exercise checks; intentionally fails before the TODOs are done |
| `make test-offline` | Run the complete model-free suite after implementation |
| `make lint` | Run Ruff on the course-maintained code |
| `make inspect-actions` | Part 1: report the available pi0-FAST response |
| `make low-risk` | Part 2: trusted low-risk path, no SST |
| `make build-auth` | Build Auth from the pinned IoTAuth source |
| `make generate-sst` | Generate SST runtime state under `runtime/sst` |
| `make auth-start` | Start the generated Auth process |
| `make test-sst` | Test the pinned SST API and the course integration |
| `make high-risk-allow` | Part 3: authorized high-risk path through real SST |
| `make high-risk-deny` | Part 3: denied context holds before actuation |
| `make bypass` | Part 3: unauthenticated actuator attempt is rejected |
| `make threshold-sweep` | Part 4: real-SST fixed-case multi-threshold evaluation and measured latency |
| `make threshold-sweep-offline` | Explicit offline diagnostic; not an accepted Part 4 result |
| `make grad-key-validity` | Part 5 (CSE 598): real SST session-key validity |
| `make auth-stop` | Stop only the Auth process this project recorded |
| `make submission GROUPID=<id>` | Build the one ZIP to upload |
| `make clean` | Delete generated results, runtime state, and caches |

The `-offline` variants (`high-risk-allow-offline`, `high-risk-deny-offline`,
`bypass-offline`, `threshold-sweep-offline`, and
`grad-key-validity-offline`) use a deterministic in-process authorizer so you
can develop without a running Auth. They are development aids and are never
accepted as real SST results. Their defaults are kept separate under
`results/*-offline/`, so running a diagnostic cannot replace graded live
evidence. The configuration-only Part 1 diagnostic likewise writes to
`results/inspect-actions-configuration-only/`.

## Results

Each scenario writes a concise record:

```text
results/<scenario>/action_record.json
```

Large action chunks go to compressed NPZ files instead of being repeated in
JSON. The controlled multi-threshold evaluation writes
every measured trial to matching CSV and JSON,
recomputes its summary from those rows, and writes one PNG under
`results/threshold-sweep/`. Rates use all classified trials as their
denominator; authorization activity counts trials that entered the high-risk
path; request attempts, successful requests, reuse, policy denials,
infrastructure errors, and transfers remain distinct. Every measured row is
retained. Confirmed EXECUTE/HOLD intervals, including denied paths that reach
HOLD, contribute to latency aggregation; a pre-decision failure instead keeps
its separate elapsed-to-failure value because no complete Monitor-to-actuator
interval exists. The largest complete sample is an observed maximum, not a
worst-case bound. The CSE 598 experiment writes to
`results/grad-key-validity/`. Part 1 writes
`results/inspect-actions/response_summary.json`.

Every record states its action source. Never relabel a fixture as model output.
Never write session keys, private keys, passwords, or credentials to results.

Check your records with either form:

```bash
python3 scripts/validate_results.py
python3 scripts/validate_results.py results
```

## Cleanup

After any SST run:

```bash
make auth-stop
make doctor
```

The project stops only the process IDs it recorded. Do not use `pkill`.
`make clean` removes generated results, SST runtime state, environments, and
caches. It keeps the pinned submodules.

## Submission

```bash
test -e submission/answers.md || \
    cp submission/answers_template.md submission/answers.md
# fill in submission/answers.md; the command above never overwrites it
# CSE 494
make submission GROUPID=<groupid>
# CSE 598: fail unless all Part 5 artifacts are present
make submission GROUPID=<groupid> REQUIRE_GRAD=1
unzip -l submission/group<groupid>_safe-agent-actuation.zip
```

One group member uploads the single ZIP to Canvas. CSE 598 groups set
`REQUIRE_GRAD=1` so packaging fails if any required extension artifact is
missing; the extension remains in that same archive. Canvas grades the uploaded
ZIP, not a newer copy on GitHub or Sol, so inspect it before submitting.

From your local machine, copy the inspected archive from Sol before uploading:

```bash
scp <ASURITE>@sol.asu.edu:/scratch/<ASURITE>/<PRIVATE_REPOSITORY>/submission/group<groupid>_safe-agent-actuation.zip .
```

## License

BSD 2-Clause. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

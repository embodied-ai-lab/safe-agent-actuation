# Safe Agent Actuation: Analyzing and Authorizing VLA Actions

> This template repository is a project for the combined course of
> [CSE 494](https://catalog.apps.asu.edu/catalog/classes/classlist?keywords=85268&searchType=all&term=2267)
> and [CSE 598](https://catalog.apps.asu.edu/catalog/classes/classlist?keywords=87933&searchType=all&term=2267),
> "Topic: Intelligent and Safe Cyber-Physical Systems" (ISCPS in short),
> at Arizona State University (ASU).
> If you have any questions about this project, please get in touch with the
> instructor, [Hokeun Kim](https://hokeun.github.io/), via
> [hokeun@asu.edu](mailto:hokeun@asu.edu).

A vision language action (VLA) model turns a camera image and a text
instruction into robot motion. This project puts a safety layer in front of one.
You measure how far a decoded action would move a simulated ALOHA bimanual
robot, classify that motion as low risk or high risk, require context based
authorization from the Secure Swarm Toolkit (SST) before a high-risk action may
move, and enforce the decision at the actuator. Everything runs in simulation.
Nothing here controls a real robot.

```text
language command and ALOHA observation
                  |
                  v
         pi0-FAST Policy Engine
                  |
          generated action tokens
                  v
             FAST Decoder
                  |
        decoded physical action chunk
                  v
              Monitor
        +---------------------+
        | Joint Motion Analyzer
        | Threshold Gate
        | Context Builder
        | SST authorization
        +---------------------+
            |             |
     low-risk path   high-risk SST path
            |             |
            +------v------+
             Actuator
      verify, decrypt, execute or hold
                  |
                  v
       headless MuJoCo ALOHA simulator
```

pi0-FAST emits discrete codes. The FAST decoder turns them back into a
continuous trajectory. A single token identifier is not a joint command and is
not safe or unsafe on its own, so the Monitor scores the decoded and
transformed action chunk against the current observed joint state.

| Property | Value |
|---|---|
| Model family | pi0-FAST |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| IoTAuth commit | `1a40ebece054880e4fe400c1c5880b05ac91ad79` |
| Simulator command dimension | 14 |
| Predicted horizon | 32 |
| Executed prefix | first 10 steps |
| Default threshold | 0.10, compared with `>=` |

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
6. Clone your new private repository, not this template.

```bash
git clone --recurse-submodules git@github.com:<OWNER>/<PRIVATE_REPOSITORY>.git
cd <PRIVATE_REPOSITORY>
make submodules
```

Put your group ID and every member name in `submission/answers.md`.

## Run on ASU Sol

[Sol](https://docs.rc.asu.edu/about) is the default platform. Login nodes are
for cloning, editing, and submitting jobs. Run the environment setup, Auth, and
every experiment inside a compute allocation. The scripts refuse to run when the
hostname contains `login`.

Supply your course account and QoS on the command line:

```bash
sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> slurm/run_experiments.sbatch
```

CSE 598 groups add the extension to the same run:

```bash
sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> \
    --export=ALL,RUN_GRAD_EXTENSION=1 slurm/run_experiments.sbatch
```

Both produce one ZIP. There is no second upload.

### Container image

Point `SIF` at the course image announced in Canvas:

```bash
export SIF=<COURSE_PROVIDED_SIF>
```

If no image is provided to you, build one once in a compute allocation:

```bash
sbatch -A <COURSE_ACCOUNT> -q <COURSE_QOS> slurm/build_container.sbatch
```

The build writes the image and its SHA-256 to your scratch space. Model
weights, credentials, and results are never stored in the image.

For an interactive allocation:

```bash
export SIF=<COURSE_PROVIDED_SIF>
lab() {
    apptainer exec \
        --bind "$PWD:$PWD" --bind "/scratch/$USER:/scratch/$USER" \
        --pwd "$PWD" "$SIF" "$@"
}
lab make setup
lab make doctor
lab make test-offline
lab make lint
```

## Command guide

Run `make help` for the live list.

| Command | Purpose |
|---|---|
| `make submodules` | Initialize the exact OpenPI and IoTAuth pins |
| `make setup` | Create the Python environment on a compute node |
| `make doctor` | Check host, Python, submodule pins, and tools |
| `make test-offline` | Run the public model-free tests |
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
| `make threshold-sweep` | Part 4: compare thresholds and latency |
| `make grad-key-validity` | Part 5 (CSE 598): real SST session-key validity |
| `make auth-stop` | Stop only the Auth process this project recorded |
| `make submission GROUPID=<id>` | Build the one ZIP to upload |
| `make clean` | Delete generated results, runtime state, and caches |

The `-offline` variants (`high-risk-allow-offline`, `high-risk-deny-offline`,
`bypass-offline`, `grad-key-validity-offline`) use a deterministic in-process
authorizer so you can develop without a running Auth. They are development aids
and are never accepted as real SST results.

## Results

Each scenario writes a concise record:

```text
results/<scenario>/action_record.json
```

Large action chunks go to compressed NPZ files instead of being repeated in
JSON. The threshold sweep writes CSV, JSON, and PNG under
`results/threshold-sweep/`. The CSE 598 experiment writes to
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
cp submission/answers_template.md submission/answers.md
# fill in submission/answers.md
make submission GROUPID=<groupid>
unzip -l submission/group<groupid>_safe-agent-actuation.zip
```

One group member uploads the single ZIP to Canvas. CSE 598 groups include the
extension in that same archive. Canvas grades the uploaded ZIP, not a newer copy
on GitHub or Sol, so inspect it before submitting.

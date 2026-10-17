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
6. Clone your new private repository on Sol and do your work there.

```bash
cd /scratch/$USER
git clone git@github.com:<OWNER>/<PRIVATE_REPOSITORY>.git
cd <PRIVATE_REPOSITORY>
make submodules
```

Replace `<OWNER>` with the GitHub account that owns your group's repository
and `<PRIVATE_REPOSITORY>` with its name. `make submodules` initializes the
pinned OpenPI and IoTAuth dependencies; do not edit those dependencies.

Create your answers file once and put your group ID and every member name in it:

```bash
test -e submission/answers.md || \
    cp submission/answers_template.md submission/answers.md
```

## Run on ASU Sol

[Sol](https://docs.rc.asu.edu/about) is the primary platform. Login nodes are
for cloning, editing, and Slurm submission. Build the container, set up Python,
run tests, run Auth and experiments, and package results only inside a compute
allocation. All graded parts, including the CSE 598 extension, use CPU compute.
The pi0-FAST capture is committed, so no GPU, model download, or model server is
required.

The repository, image, and `.venv` persist between allocations while their
files remain on scratch. Recreate the allocation, loaded modules, `SIF`, and
`lab` function each time. Auth must also be started again. Scratch is temporary
and not backed up; keep your source work in the group's private repository.

### 1. Build the container in a CPU allocation

```bash
interactive -A class_cse494598fall2026 -p public -q class \
    -t 120 -c 8 --mem=32G
module load apptainer/1.4.5 squashfs-4.6.1-gcc-11.2.0
cd /scratch/$USER/<PRIVATE_REPOSITORY>
export SIF=/scratch/$USER/safe-agent-actuation.sif
make build
```

`make build` uses `containers/Apptainer.def` and writes the image plus
`/scratch/$USER/safe-agent-actuation.sif.sha256`. Record the allocation's
`SLURM_JOB_ID` and the image checksum in your answers. Build once; subsequent
allocations can reuse the image. Model weights, credentials, and results are
not stored in it.

Alternatively, submit the build from the repository root on a login node:

```bash
sbatch slurm/build_container.sbatch
squeue -u "$USER"
```

Wait for the build to finish successfully before running experiments. Its logs
are `results/container-<jobid>.out` and `.err`. The default output path is the
same as in the interactive build.

### 2. Set up and check the environment

Continue in the CPU allocation above. If it has ended, request another with
the same `interactive` command, load `apptainer/1.4.5`, and return to your
repository. Define the wrapper in each allocation:

```bash
export SIF=/scratch/$USER/safe-agent-actuation.sif
lab() {
    apptainer exec \
        --bind "$PWD:$PWD" --bind "/scratch/$USER:/scratch/$USER" \
        --pwd "$PWD" "$SIF" "$@"
}
lab make setup
lab make doctor
lab make test-baseline
lab make lint
lab make inspect-actions
```

Use `lab` for project commands in the allocation shell. It runs them in the
container; `make setup` creates the locked Python environment in `.venv`.
The starter's baseline tests, lint, and Part 1 inspection should pass before
you implement any exercises.

### 3. Complete the student TODOs

Follow [ASSIGNMENT.md](ASSIGNMENT.md). The student template leaves five graded
regions unfinished:

| Source under `src/safe_agent_actuation/` | Work | Required before |
|---|---|---|
| `motion.py` | Three motion-score and classification regions | Part 2 low-risk run |
| `scenarios.py` | Protected delivery and actuator enforcement | Parts 3-5 real SST runs |

Run `lab make test-acceptance` once to see the expected `TODO(student)` failures.
After implementing the five regions, run:

```bash
lab make test-acceptance
lab make test-offline
lab make lint
lab make low-risk
```

`make setup` exposes the course source directly, so edits take effect on the
next command without rebuilding the image. The full batch workflow also runs
acceptance tests and will stop on unfinished student code.

### 4. Run the real SST experiments

Auth and its clients must share one container session on the same compute
node. Run this entire block after completing the TODOs. The commands inside
the block use plain `make` because `lab bash` has already entered the container:

```bash
lab bash <<'SST'
set -euo pipefail
make build-auth
make generate-sst
trap 'make auth-stop' EXIT
make auth-start
make test-sst
make high-risk-allow
make high-risk-deny
make bypass
make threshold-sweep
SST
```

The exit trap stops Auth even if a command fails. Do not start Auth with one
`lab make auth-start` invocation and run its clients in separate invocations;
the background process can lose its container mounts when the first exits.
Run one SST experiment workflow at a time per checkout.

After the common experiments finish, CSE 598 groups also run:

```bash
lab make grad-key-validity
```

This command generates fresh runtime state and starts and stops its own Auth
process for each of the required 5, 20, and 40 second validity periods. It
requires the Auth build from the common workflow.

### Batch alternative after implementation

Once the image exists and the TODOs pass their checks, exit the interactive
allocation and submit from the repository root. Choose the command for your
course section:

```bash
exit
cd /scratch/$USER/<PRIVATE_REPOSITORY>
export SIF=/scratch/$USER/safe-agent-actuation.sif
# CSE 494: common experiments
sbatch slurm/run_experiments.sbatch
# CSE 598: common experiments and the extension in one job
sbatch --export=ALL,RUN_GRAD_EXTENSION=1 slurm/run_experiments.sbatch
squeue -u "$USER"
```

Submit only one of the two jobs. Both use the course account and QoS recorded
in the script, set up the environment, check the implementation, run the
selected experiments, validate results, and stop Auth. Logs are
`results/slurm-<jobid>.out` and `.err`; results use the paths below. Re-running
experiments replaces those scenario results, so preserve any earlier evidence
you need. The job does not create a submission ZIP.

## Command guide

Run `make help` for the command list. The table lists Make targets; prefix them
with `lab` on Sol unless you are already inside the container. Build the image
with host-side `make build` and initialize dependencies with `make submodules`.

| Command | Purpose |
|---|---|
| `make submodules` | Initialize the exact OpenPI and IoTAuth pins |
| `make build` | Build the container on a CPU compute node |
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

Check your records inside a CPU allocation:

```bash
lab python3 scripts/validate_results.py results
# CSE 598: also require the Part 5 artifacts
lab python3 scripts/validate_results.py --require-grad results
```

## Submission

Finish `submission/answers.md`, then package inside a CPU allocation with the
`lab` wrapper from step 2. If you ran the batch job, request a new allocation,
load Apptainer, return to your repository, and define `SIF` and `lab` again.
Choose the packaging command for your course section:

```bash
# CSE 494
lab make submission GROUPID=<groupid>
# CSE 598: fail unless all Part 5 artifacts are present
lab make submission GROUPID=<groupid> REQUIRE_GRAD=1
unzip -l submission/group<groupid>_safe-agent-actuation.zip
```

One group member uploads the single ZIP to Canvas. CSE 598 groups set
`REQUIRE_GRAD=1` so packaging fails if any required extension artifact is
missing; the extension remains in that same archive. The ZIP includes current
project files whether or not they are committed. It excludes the image, model
weights, `.venv`, caches, and SST credentials. Canvas grades the uploaded ZIP,
not a newer copy on GitHub or Sol, so inspect it before submitting.

From your local machine, copy the inspected archive from Sol before uploading:

```bash
scp <ASURITE>@sol.asu.edu:/scratch/<ASURITE>/<PRIVATE_REPOSITORY>/submission/group<groupid>_safe-agent-actuation.zip .
```

## Cleanup after submission

The SST blocks and batch script stop their Auth processes automatically. To
check the environment and remove generated state after inspecting and saving
your ZIP:

```bash
lab make auth-stop
lab make doctor
lab make clean
```

`make clean` deletes experiment results, SST runtime state, local environments,
and caches. It keeps the source, pinned submodules, and submission ZIP. Do not
clean before packaging: the deleted results are needed to rebuild the ZIP.
The project stops only the Auth process ID it recorded; do not use `pkill`.

## Optional: run on your own Linux machine

Use Python 3.11 or 3.12, uv, Git, Make, Java 17, Maven, Node.js 22 with npm, OpenSSL,
and ZIP tools. Initialize the submodules and run `make setup`, `make doctor`,
`make test-baseline`, and `make lint`. Follow the same assignment and experiment
sequence without the `lab` prefix; run the SST block with `bash` in place of
`lab bash`. A local run needs no Apptainer or GPU. Report your platform because
latency measurements depend on the machine.

## License

BSD 2-Clause. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

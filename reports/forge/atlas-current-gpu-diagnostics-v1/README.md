# Fixed Atlas current-suite GPU diagnostics

This launcher is a separately preregistered diagnostic batch. It does not enqueue
an ordinary request, bypass an ordinary prerequisite, alter the provisional
calibration profile or qualify a shipping default. The existing 44,100-second
ordinary draft remains unchanged.

One Atlas configuration uses shared `lr=0.0053125`, `prior_lr_mult=1.5`, protocol
seed 0, and each frozen task's explicit C6 host exceptions. All tasks use their
prospective GPU `policy_selected_cloud_v1` declarations: public policy lifecycle,
actual selected-state serving, declared particle rows and original numerical
metrics. Historical Atlas19/C6 results motivate this identity but fill no cell.
The emitted structural readiness card claims host/API preflight only; it is not
a capacity witness or learned convergence result.

The complete ledger retains 26 required questions at tiers 5/19/2. Eighteen
supported slots form 17 executable jobs with a **34,800-second inclusive cap**.
Eight unsupported conditional/component hosts remain pretraining BLOCKED and
reserve nothing. All 26 definitions have 45,300 seconds of grouped allowance;
the unsupported definitions account for 10,500 seconds of that amount.

The two ring slots share one uninterrupted 3,600-second job, their original
maximum 7,500 updates and earlier concluded-FAIL stop. Native tasks retain full
7,000-update/34-observation protocols, 20k clean selected-policy checks and
independent 100k accuracy holdouts. Transfer tasks retain their original budgets,
24 observations and sustained gates. This is distinct from earlier noisy Atlas19
or the API 24-check/no-100k cohort. CPU controls establish structure only.

Run preparation only after root commits the final implementation/task source.
Preparation copies a content-addressed source tree and writes an immutable
sidecar outside the fresh output, so registration can still admit an empty
canonical output. It initializes no queue, reserves no device and performs no
training. It may query hardware identity without initializing CUDA.

```sh
PG_DIAGNOSTIC_OUTPUT=/ml2/hypergan/forge-atlas-current-gpu-diagnostics-v1
PG_SHARED_QUEUE=/ml2/hypergan/ParticleGAN-single-recipe/runs/forge
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venvs/particlegan-develop-integration/bin/python \
  reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py \
  --spec reports/forge/atlas-current-gpu-diagnostics-v1/protocol.json \
  --output "$PG_DIAGNOSTIC_OUTPUT" --prepare-only
```

Root launches the single admitted physical GPU1 lane. The wrapper establishes
PCI_BUS_ID ordering, CUBLAS `:4096:8`, CPU/BLAS threads 1, no bytecode writes and
CUDA visibility 1 before importing the actual execution code. Admission requires
the declared A6000 identity, at least 12,288 MiB free and temperature at most
82 C. The numerical child sets memory fraction 0.2 before fixture allocation.

```sh
.venvs/particlegan-develop-integration/bin/python \
  reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py \
  --output "$PG_DIAGNOSTIC_OUTPUT" --queue-root "$PG_SHARED_QUEUE" --gpus 1
```

Each job uses `PolicyCoordinator` and its shared exclusive lease/durable
supervisor. The admitted inherited descriptor is checked against its durable
token/source and bridged to `FORGE_LEASE_FD`. Fresh independent processes invoke
the unchanged `runtime.execute` and `evaluate.evaluate`; no trainer loop is
copied. Execution, independent grading and draw-free goal rendering share the
original inclusive job allowance, with no extra export grace.

A valid completed numerical FAIL permits the next diagnostic job. Source,
contract, policy-health, runtime, checkpoint or infrastructure invalidity halts
the batch. Remaining slots retain NOT_RUN; interrupted work retains INCOMPLETE.
There are no automatic failed retries, seed changes or shorter resources/budgets.
The external `mode_hold` gate does not prevent the independent ring diagnostic,
but ordinary ring qualification still requires it. Actual checkpoint/data
dependencies are never substituted or ignored.

`study.json` and its adjacent readout preserve all 26 slots and per-attempt
source/Recipe/runtime, independent grades, artifact hashes, raw logs and costs.
Completed attempts charge measured paid time; interruptions retain actual paid
time plus the remaining conservative reservation. A complete next allowance must
fit the finite cap. Resume recertifies retained completed evidence and cannot
rerun an interrupted/invalid attempt. Logs are easy to tail:

```sh
tail -f "$PG_DIAGNOSTIC_OUTPUT/attempts/two_pole_policy_selected_cloud_v1/run.log"
```

Goal GIFs use actual retained observations, without new samples, forwards,
optimizer updates or interpolated frames. Static declared means/spiral
centerlines can illustrate reference geometry and are labelled separately.
Native frame badges retain recorded 20k decisions; the plotted 4096 points do
not supply additional qualification. Every GIF/readout states diagnostic scope.
Raw tensors, JSONL events and checkpoints remain outside Git. Default/ordinary
tier/calibration/speed credit is always false.

No science or GPU run has been performed by the implementation author. Root owns
final source freeze, admission and execution. If later adapters make the eight
blocked hosts executable, this frozen 18-slot contract must not silently expand;
declare and review a new bounded diagnostic identity.

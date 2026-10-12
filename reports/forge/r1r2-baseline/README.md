# R1/R2 baseline and failure readout

The selected R1/R2 recipe is the reference baseline for subsequent R1/R2
comparisons: `recipe-prior-refactor-r1r2-v1`, measured at source
`a0493ca3c213d559523eadc76e244ec9f9c1f81f`. It passes **20/22 Tier 1 view
requirements**, representing **five PASS and two FAIL across seven distinct
tasks**. The failures are **Ring16 density acquisition** and **clock independence**.
Keep both failures as part of the baseline. Its ordinary stability result is
**5/6**, its attained qualification tier is **0**, and its screen remains
provisional. A reference baseline need not pass every gate.

[Current technique leaderboard](../technique-inventory.md) ·
[R1/R2 family and per-view results](../families/r1r2.md) ·
[Exact baseline metrics, recipe and protocol](results.json) ·
[Frozen source evidence](../technique-evidence/29979e41c9c05e3304156cf4a43ed9c9dda6953bffd0fb47096fe47d6b8af13a.json)

## What 20/22 counts

Three passing behavioral tasks (`two_pole`, `unused_token_hold`, `ae_gan_hold`)
are required by all six displayed views: 18 passing requirements. Gaussian and
word smoke add one pass each. Ring16 contributes the failed requirement in
`discriminator_stability` (5/6); the optional stability clock measurement supplies
the failed requirement in `clockfree_continuous` (3/4). Each of the other four
views has 3/3 Tier 1 requirements. These totals describe repeated view
requirements, rather than 22 independent executions.

All seven distinct measurements completed through the public ParticleGAN API.
Their original paid cost is **179.668 seconds**, including the clock audit.
This report adds **zero updates, sampling draws or paid training seconds**.

## Reference recipe and comparison contract

Use the [current baseline declaration](../../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json)
as one whole global recipe across tasks. Its lineage is the previously selected
`r1r2--302b6baa…` Modern GAN toy adaptation. The ownership migration explicitly
moves prior updates and regularization into the recipe; the older configuration's
archived grades remain a separate source and training contract.

| Setting | Recorded baseline |
| --- | --- |
| Adversarial objective | Paired relativistic logistic loss |
| Critic regularizer | R1 and R2 squared input-gradient penalties, applied every update; coefficient 1 to 0.1 over the first 20% of the task-bound schedule horizon |
| Optimizer | Native PyTorch Adam; beta1 0; beta2 0.9 to 0.99 over the first 20%; epsilon 1e-8; AMSGrad disabled |
| Rates | Base LR 0.0085; critic and learned-prior multipliers 1; cosine decay after 20% of the horizon to a 0.1 floor; no network horizon cap |
| Prior policy | Learned locations where applicable; VICReg selected with weight `prior_reg=0`; `prior_l2=0` |
| Additional mechanisms | A2, critic anchor, direct-particle gain and critic spike guard disabled |
| Serving | Live weights; EMA disabled; clean public-prior sampling; additive input/output training noise zero |
| Protocol | Seed 0; public deterministic initializer; isolated named RNG streams; task-owned architecture, data, initial prior/capacity, budget and cadence |
| Recorded runtime | CUDA, RTX A6000, Python 3.12.13, PyTorch 2.14.0; deterministic algorithms; one CPU thread; TF32 and autograd multithreading disabled |

The task owns the actual initial prior. Ring16 and Gaussian smoke use learned MoG
priors with sigma **0.1**. Word smoke explicitly uses five learned particle-cloud
rows with sigma **0**, preserving its finite-vocabulary exception. The direct
particle and embedding controls keep their declared parameter-only conditions;
the AE and clock tasks retain their sigma **0.025** MoG declarations. The generic
protocol's sigma 0.025 does not replace these task bindings.

The report's machine-readable bindings retain the candidate revision
`594012d6f37dcf9bb8d8c2e9c84085cfcd896d5d44843496901fb5d0019c617a`,
source digest, recipe/protocol/RNG hashes, runtime cohort, task fingerprints,
receipt hashes and original raw-result identities. Subsequent comparisons must
declare their trainer delta, including prior policy or regularization changes,
and hold each task's initial tensors, target minibatches, sampling law, budget and
cadence fixed. The completed ownership campaign verifies actual batch alignment
for Gaussian smoke/stability; other hosts lack an actual batch trace alignment
proof. Word continuations begin from different acquisition lengths. Thus these
historical cross-family results do not establish a fully matched superiority
comparison. [Original campaign protocol and limits](../recipe-prior-refactor/README.md).

## The two failures

### Ring16: modes found, local density inaccurate

The task fits 16 equally weighted 2D Gaussian clusters on a radius-3 ring, each
with sigma 0.1. It uses 256 latent rows, latent dimension 4, batch size 128,
width 64, two layers and Fourier setting 2. The frozen task allows **1,600
updates**, evaluates **96 observations** with **4,096 samples** each, and
requires at least **five consecutive passing terminal observations**.

| Endpoint metric | Recorded value | Gate | Result |
| --- | ---: | --- | --- |
| Modes | 16 | >= 16 | PASS |
| Mass total variation | 0.051514 | <= 0.15 | PASS |
| High-quality fraction | 0.844238 | >= 0.85 | FAIL |
| Component covariance error | 2.066388 | <= 0.85 | FAIL |
| Minimum component eigenvalue ratio | 0.223238 | >= 0.15 | PASS |
| Sample count | 4096 | >= 4096 | PASS |

The full gate passes at **none of the 96 observations**, leaving a terminal
passing suffix of **0/5**. Occupancy and balance succeed; the generated local
distributions have incorrect covariance and too much mass outside the target's
3-sigma region. Covariance error is about **2.43 times** its allowed maximum.
The precision shortfall is **0.576 percentage points**, so fixing that small
shortfall alone would still leave a substantial covariance failure.

Saved diagnostic endpoint statistics include maximum component spill **0.230126**
and core covariance error **1.331667**; these are context, rather than substitute
gates. The baseline's minimum-spread bound passes. This evidence does not identify
which prior rows or generator mappings cause the covariance error, or establish
that the R1/R2 penalty itself causes it.

The task's **original schedule horizon is 400**, while its execution cap is
1,600. The early beta2/penalty transition ends at 80; the LR reaches its floor
by 400. This is part of the recorded baseline contract. It is a plausible
contributor to slow density fitting, but this receipt alone does not isolate a
schedule cause. Changing the schedule horizon or execution budget creates a
separate comparison and cannot replace this FAIL.

[Ring receipt](../technique-receipts/a54bc96e82f34c9e98e7fa1f1d268a33.json) ·
[Actual-training Ring16 GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-ring16_acquisition.gif)

### Clock audit: scheduled updates depend on time labels

`clockfree_audit_measurement_v1` compares the same saved learning state and
training RNG state after four warmup updates. Four-update probes independently
change the external step label (+8192), schedule horizon (128 to 4096), evaluation
cadence, or checkpoint restart. Exact common-prefix state equality is the
numerical requirement, with a source audit as an additional requirement.

| Perturbation | Recorded state comparison |
| --- | --- |
| Step label | FAIL: differs from the reference |
| Schedule horizon | FAIL: differs from the reference |
| Evaluation cadence | Exact match |
| Checkpoint restart | Exact match |

The independent grader records **“step_label changed the update or common-prefix
state”** as its first rejection. The byte-exact original result also records
horizon divergence; the grader's first rejection does not mean later probes
were unmeasured. The source audit explicitly identifies LR annealing's completed
step/horizon dependence. Beta2 and penalty schedules are additional declared
time-dependent recipe settings. Their individual contributions were not isolated.

This is an expected limitation of the scheduled baseline's clock-free claim;
the recipe explicitly declares `schedule=scheduled`. Cadence and restart parity
are useful controls, but do not supply clock-free qualification. The clock
measurement is optional in the ordinary stability view and required in the
displayed clock-free view. **Ring16 alone blocks ordinary Tier 2 eligibility.**

[Clock receipt](../technique-receipts/12bc543e8b564690953453bfaf8720cc.json) ·
[Compact original clock comparisons and artifact hashes](clock-audit.json) ·
[Clock audit implementation](../../../experiments/forge/clockfree.py)

## What the baseline already solves

| Distinct task | Recorded numerical evidence | Result and training illustration |
| --- | --- | --- |
| `two_pole` | mean_abs 0.381033 >= 0.3; grad_med 0.692541 <= 1; eight terminal passing observations | [PASS / GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-two_pole.gif) |
| `unused_token_hold` | concept_move 0.99999976 >= 0.85; unused_hold 0.99999997 >= 0.85; 19 terminal passing observations | [PASS / GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-unused_token_hold.gif) |
| `ae_gan_hold` | recon_mse 0.002850 <= 0.05; hold 0.001638 <= 0.35; 15 terminal passing observations | [PASS / GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-ae_gan_hold.gif) |
| `gaussian1d_smoke` | First full independently confirmed acquisition at update 292; six passing scheduled observations | [PASS / GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-gaussian1d_smoke.gif) |
| `five_word_joint_smoke` | First full independently confirmed joint generation/inverse acquisition at update 2501; one passing scheduled observation | [PASS / GIF](../recipe-prior-refactor/publication/media/r1r2-ordinary-five_word_joint_smoke.gif) |

Gaussian and word smoke score confirmed acquisition, rather than terminal
retention. Word smoke's endpoint fails: only four modes, mass TV **0.217188**,
exact reconstruction **0**, and reconstruction NLL **7.169692**. Its recorded
acquisition PASS is correct under its original gate; it does not establish a
stable word solution. The separate word-hold diagnostic fails too.

## Separate deeper measurements and recommendation

Ordinary Tier 2 and Tier 3 cells remain unmeasured after the required Ring16
failure; their displayed zeros are unknowns. The completed, explicitly registered
Tier 2 diagnostic cohort measures **9 PASS / 12 FAIL**, providing useful context
without ordinary qualification credit. The 12 diagnostic failures are:

- Gaussian stability, five-word hold, mode hold.
- Trajectory identity, residual-student identity.
- Unequal-mass, unequal-width and overlapping vector distributions.
- Two-stripe images; grid100, rotated100 and staggered100 native distributions.

[Exact diagnostic statuses and failed endpoint checks](results.json) retain their
original scope; an endpoint check does not replace a sustained gate. All R1/R2
measurements in the ownership campaign, including two separately charged
diagnostic prefix producers, cost **852.587 seconds across 30 attempts**. This
overlaps the ordinary cost above and must not be added to it.

Retain this exact whole recipe as the baseline, with the current leaderboard as
the single ranking. Prioritize saved per-component covariance, spill and
prior-row/generator mapping analysis for Ring16; inspect the word acquisition-to-
failure trajectory for retention. Any later repair needs an explicit global
trainer delta under the fixed task suite. Clock independence requires its own
mechanism change and qualification. Neither per-task recipe selection, a different
seed, a looser gate nor a diagnostic pass establishes an ordinary replacement.
No new experiment, calibration acceptance or public-default adoption follows
from this reporting PR.

## Reproduce the report checks

Run from the repository root. The audit verifies committed compact receipts,
original source task definitions and report projections; it launches no models.
If the original source object is missing, fetch its exact commit first.

```sh
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
$PY reports/forge/r1r2-baseline/audit.py
# Optional independent check against the archived byte-exact clock result:
$PY reports/forge/r1r2-baseline/audit.py --clock-result \
  /mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/queue/recipe-prior-refactor-selected-leaders-v1/12bc543e8b564690953453bfaf8720cc/result.json
```

Use `--write` only to refresh this report's compact projection. Qualification,
selection and original receipts remain owned by their existing publications.
Bulk checkpoints, stdout and per-update traces remain in the original artifact
archive. Reporting validation logs are local and easy to tail:
`tail -F runs/forge/r1r2-baseline-report/validation.log`.

[Validation receipt](validation.json): the baseline audit, original clock-result
hash verification, report-link/cost checks and `forge compile --check` pass.
The existing family documentation/report suites record 109 passes and two
failures; both failures reproduce on the unchanged PR base
`6cdc209a3956aee44a2f42fac4a92bcaddab46a4` (generated-page freshness and legacy
historical-family alias expectations). This report preserves those existing
publications and does not repair their unrelated presentation drift.

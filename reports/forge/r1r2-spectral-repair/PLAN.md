# Fixed-task R1/R2 spectral update comparison

The user's continuing objective is a single R1/R2 recipe that passes the original
Tier 1 tasks and clock audit. This new finite hypothesis follows the completed
[direction/floor comparison](../r1r2-direction-floor-repair/README.md).
Read [compiled memory](../EXPERIMENT_MEMORY.md), the original
[baseline](../r1r2-baseline/README.md), [constant-policy](../r1r2-tier1-search/README.md),
[five-track](../r1r2-five-track-repair/README.md) and
[optimizer-role](../r1r2-role-repair/README.md) readouts. Their failed recipes and
closed registrations remain unchanged. No task, protocol or gate changes.

## Why this different optimizer question

Direction preservation and critic epsilon failed density and word acquisition.
The [saved spectral note](../r1r2-direction-floor-repair/optimizer/spectral-note.md)
finds that Ring16 hidden G/D gradients concentrate 59%/89% of squared norm in
their leading singular mode. AdaNSGDA retains those singular-value ratios;
the existing DualNorm polar update flattens retained singular values and gives
each matrix a dimensional operator-norm step budget. The parent's saved hidden
critic update operator norm is about .266 versus .0085 for this polar law.
This is a distinct structural update constraint, not another epsilon or rate grid.

It does not constrain the spectrum of existing weights or the trained generator
Jacobian. The saved parent Jacobians have moderate conditioning, so no claim of
extreme Jacobian ill-conditioning motivates this test. Constant normalized steps
may still fail density convergence. Prior BCAP full and critic-only DualNorm
failures remain negative evidence under their original source, formulation,
rates, momentum and initialization; no prior result grants a new R1/R2 pass.

## Whole global recipes and ownership

All new arms inherit the exact constant parent
`r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9`,
revision `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`.
Keep paired R1/R2 coefficient .1 every update, constant LR .0085, all role
multipliers 1, declared betas (0,.99), epsilon 1e-8, null D/prior epsilon and
beta overrides, learned prior locations, zero extra prior regularization/noise/EMA,
and no AMSGrad, transport, guard, geometry or continuous policy.
`optimizer_momentum=0`, `optimizer_smoothing=0`, native SVD backend and no
convolution variant remain fixed. `optimizer_adam_lr` stays null.

| Role | Whole recipe | New runs |
| --- | --- | ---: |
| `baseline` | Exact scheduled reference | 0 |
| `adam` | Exact constant Adam parent | 0 |
| `ada` | Exact failed direction-preserving AdaNSGDA recipe | 0 |
| `spectral` | `optimizer_family=dualnorm` | 7 |
| `critic_spectral` | `optimizer_family=dualnorm_D_only` | 7 |

The optimizer family is the only changed recipe field. Full DualNorm changes
G/E and D to the polar law and the sampled learned prior to row normalization
on the actually drawn generator-side rows. This is an explicit prior update-law
change while `prior_update=learned`, prior regularization, initial distribution,
capacity and sampling remain fixed. Beta2 is inactive in polar/row-normalized
groups; all native Adam groups retain (0,.99). Critic-only DualNorm changes D
alone and keeps G/E, sampled prior and direct pole native Adam at .0085.
The contrast therefore bundles G/E and prior changes; it cannot isolate G alone.

Direct pole coordinates remain the explicitly declared generator-owned fixture.
Their full polar step has `sqrt(12)` dimensional scaling: mean absolute travel
is bounded above by .0085 per update, or .68 over 80 updates. This permits the
.3 gate but proves neither transport nor sustained qualification. Token has no
sampled prior. The word host consumes these same public factories and actual
prior row indices. All original acquisition and hold controls must pass.

## Protocol, budget and stopping

Preserve protocol seed 0, public deterministic initialization, architecture,
data law and ordered actual target batches, prior law/capacity, sampling,
update caps, evaluation cadence and original gates. Checkpoint every consumed
constructor/data/training/evaluation stream. A repair requires all six ordinary
gates plus its own clock gate under one global recipe; no per-task mixture.
Prediction is the original sustained Ring16 full gate, including covariance
<=.85 and minimum eigen ratio >=.15 for five terminal checks. Any original task
or clock failure rejects the complete recipe. No new draws or updates for media.

Common scientific source stays
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`,
runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Preparation, [registration](registration.json) and independent review must verify
all effective public task bindings and three exact cached projections/results.
The predecessor must be complete before preparing or admitting this campaign.
No cached control may rerun; retain its original single-attempt identity.

Exactly 35 final cells: 14 new runs and 21 exact cached cells. Full reservation
is 12,600 seconds (5 × 2,520), while the aggregate shared paid cap stays 8,000.
It includes all 175 prior charges totaling 4,587.580182052945 seconds and every
new/concurrent full-task reservation. Temporary concurrent reservation denial
may serialize a job; no scientific timeout or task allowance is reduced.
Use GPUs 0/1, one worker per GPU, one CPU thread, unchanged deterministic/TF32
settings. Combined five-archive storage cap is 8 GiB and free-space floor 5 GiB.

One round, no retries, seeds, Tier 2, longer budgets or automatic refinement.
Close and audit the comparison before another hypothesis. No default promotion
without the declared screening calibration. Keep original failures and archive
identities visible; store bulk logs/checkpoints outside Git. The technique
inventory remains the only current generated leaderboard.

## Reproduction

Commit every frozen source/declaration/registration before admission. Use a
separate checkout at the execution commit named in the final report for admission;
use retained original artifacts for read-only collection rather than rerunning.
The new archive's `queue` symlink must resolve to the original shared queue.
Redirect coordinator stdout to `prepare.log`, `run.log`, and `collect.log` there.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/workflow.py prepare --manifest reports/forge/r1r2-spectral-repair/manifest.json --registration reports/forge/r1r2-spectral-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/workflow.py run --manifest reports/forge/r1r2-spectral-repair/manifest.json --registration reports/forge/r1r2-spectral-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012 --source-commit COMMITTED_SHA --submit --drain --gpus 0,1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/workflow.py collect --manifest reports/forge/r1r2-spectral-repair/manifest.json --registration reports/forge/r1r2-spectral-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012 --output reports/forge/r1r2-spectral-repair/publication --media
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012/queue/events.jsonl
```

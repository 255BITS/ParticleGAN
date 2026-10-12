# R1/R2 gradient direction and critic denominator comparison

This is a new finite comparison under the user's continuing request to make
R1/R2 pass Tier 1 without changing its failing experiments. It follows the
completed [role-momentum comparison](../r1r2-role-repair/README.md); that
registration is closed and unchanged. No task, protocol, gate, initialization,
architecture, data distribution, prior capacity, sampling or update allowance
changes. Every candidate must pass all six original ordinary tasks and its own
clock audit. One global recipe applies to the whole suite.

## Evidence and hypotheses

Read the [compiled memory](../EXPERIMENT_MEMORY.md), the original
[baseline](../r1r2-baseline/README.md), the
[constant-policy search](../r1r2-tier1-search/README.md),
[five-track readout](../r1r2-five-track-repair/README.md) and completed role
readout before this declaration. None has supplied a complete repair.

The [saved-gradient analysis](../r1r2-role-repair/optimizer/normalization.md)
finds that native Adam rotates Ring16 updates relative to the raw gradients.
The existing `ada_nsgda` family preserves each nonzero parameter tensor's
gradient direction and grafts its Adam step norm onto that direction. This
isolates within-tensor preconditioning without a uniform rate change. It does
not preserve the direction of an entire multi-tensor role, nor does it prove
that the observed rotations caused density failure. Prior failed BCAP AdaNSGDA
recipes remain separate negative evidence under their original source and
formulation, rather than repeated or substituted R1/R2 experiments.

The [saved denominator analysis](../r1r2-role-repair/prior/denominator-readme.md)
supports one critic-only epsilon change. At fixed parent moments, `d_eps=1e-4`
retains about 69% of Ring16 critic displacement L2 and 94% of pole critic
displacement L2. It preserves the direct pole generator epsilon and same-moment
step; changed critic feedback can still alter its training trajectory.
Gaussian and word critics also change, so no passing control is assumed.
In AdaNSGDA, this epsilon enters both `sqrt(v_hat)+eps` and `||g||+eps`.
The joint arm therefore also attenuates the critic norm graft on tiny-gradient
tensors. The two-factor comparison exposes this interaction explicitly.

## Frozen whole recipes

All arms inherit the exact measured constant parent
`r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9`,
revision `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`:
paired `a_r1r2`, coefficient 0.1 every update; constant LR 0.0085; beta1=0,
beta2=0.99; global epsilon 1e-8; all role rate multipliers 1; learned prior
locations with zero extra prior regularization, noise and EMA. D/prior beta
overrides remain null. No first momentum, AMSGrad, transport, geometry,
continuous policy or prior intervention is added.

| Role | Optimizer family | Critic epsilon override | New runs |
| --- | --- | --- | ---: |
| `baseline` | Exact archived scheduled reference | Original settings | 0 |
| `adam` | Exact constant Adam parent | null, effective 1e-8 | 0 |
| `adam_eps` | Adam | 1e-4 | 7 |
| `ada` | AdaNSGDA | null, effective 1e-8 | 7 |
| `ada_eps` | AdaNSGDA | 1e-4 | 7 |

`Recipe.optimizer_family` changes G, E, D and the sampled learned prior through
their public factories. Direct pole coordinates use the generator law in their
explicit native fixture; the token fixture has no sampled prior. `d_eps`
changes every supported critic group including the word host. `prior_eps`
remains null; its unsupported word-host override is not introduced.

Scientific source remains
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`,
runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
The [prospective preflight](optimizer-preflight.json), generated
[registration](registration.json) and admission independently bind the effective
recipe and every unchanged task. Cached reference projections must exactly
match measured candidate/source/runtime/task/job identities and each original
single-attempt result. No cached control may rerun.

## Predictions, budget and stopping

Prediction: the original sustained Ring16 covariance, eigen spread, HQ,
coverage, mass and terminal suffix pass under at least one of these three whole
recipes, with all acquisition/hold tasks and its clock gate also passing.
Falsifier: any original gate fails. An improved endpoint, reduced critic norm or
passing acquisition alone is insufficient. Retain all failures and identify
which factors changed; select no per-task mixture.

Exactly 35 final cells: 21 new runs and 14 exact cached cells. Full unchanged
reservation is 12,600 seconds (5 × 2,520). The original shared queue preserves
154 prior charges totaling 4,000.9795464819763 seconds. Aggregate paid cap stays
8,000 seconds, including every prior charge, new full task reservation and
concurrent worker reservation. No retries, seeds, Tier 2, task modifications,
longer training or automatic refinement. Close and audit this campaign before
another hypothesis. Provisional calibration confers no default promotion.

Use protocol seed 0, public deterministic initialization, separate named RNG
streams with complete checkpoints, identical actual initial tensors and consumed
target batches, GPUs 0/1 with one worker per GPU and one CPU thread. Collection
validates original stored gates and exports actual-training GIFs from saved
inputs with zero added updates or draws. Keep bulk logs/checkpoints outside Git.
Combined four-archive storage cap remains 8 GiB with a 5 GiB free-space floor.

## Reproduction and logs

The frozen declarations and registration must be committed before admission.
Use a checkout at the execution commit named in the final publication and the
exact project environment. Do not rerun an admitted campaign to reproduce a
report; read-only collection uses its retained original queue.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-direction-floor-repair/workflow.py prepare --manifest reports/forge/r1r2-direction-floor-repair/manifest.json --registration reports/forge/r1r2-direction-floor-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-direction-floor-repair/workflow.py run --manifest reports/forge/r1r2-direction-floor-repair/manifest.json --registration reports/forge/r1r2-direction-floor-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012 --source-commit COMMITTED_SHA --submit --drain --gpus 0,1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-direction-floor-repair/workflow.py collect --manifest reports/forge/r1r2-direction-floor-repair/manifest.json --registration reports/forge/r1r2-direction-floor-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012 --output reports/forge/r1r2-direction-floor-repair/publication --media
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012/queue/events.jsonl
```

Redirect coordinator stdout to `prepare.log`, `run.log`, and `collect.log`
in this campaign's archive. Its `queue` symlink resolves to the original archive
queue, preserving all paid ownership and artifacts. Compact final reports,
provenance, saved gate metrics, reproduction sources and GIFs enter Git.
The current technique inventory remains the only generated leaderboard.

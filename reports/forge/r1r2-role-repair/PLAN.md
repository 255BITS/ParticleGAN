# R1/R2 optimizer-role repair registration

The active user goal is to make Tier 1 R1/R2 work without changing the failing
experiments. This new finite campaign follows the closed five-track comparison;
its old stopping decisions and failed evidence remain unchanged.

The saved all-player beta1=.5 recipe improves Ring16 covariance but fails local
minimum spread, Gaussian acquisition and words. The AMSGrad recipe passes five
ordinary tasks and clock but fails density. Three subagents inspect saved
optimizer states and supported role bindings before admission. Existing public
Recipe fields permit a complete role-momentum factorial without engine changes.

Use native Adam beta1 in {0,.5} for D, G/E and sampled learned-prior locations.
All eight combinations use beta2=.99, constant LR=.0085 and paired R1/R2
gamma=.1 every update. Reuse the exact all-zero and all-.5 cards and seven
results per card. Six previously unmeasured role subsets are new recipes.
The seventh new recipe adds AMSGrad to the exact all-.5 parent. Reuse the
scheduled reference's seven original results separately. This means ten global
recipes, seventy result cells, forty-nine new runs and twenty-one exact reuses.

G and E share the G factor. Direct-coordinate pole uses the native G optimizer
betas even though its rate-bookkeeping label says prior. The sampled-prior factor
does not apply to that fixture or unsampled-token hold. Preserve those explicit
separate cohorts; no task-specific numerical rule is introduced.

Keep task files, architecture, target/data law, seed0/public initialization,
ordered consumed data, initial prior distribution/capacity, sampling, update
limits, evaluation cadence and every original gate fixed. Learned locations and
zero prior regularization remain explicit. Isolate/checkpoint constructor, data,
training-noise and evaluation streams. A solution requires all six ordinary
gates and its own clock PASS under one global recipe; a passing density endpoint
or mixing task winners is insufficient. Complete runnable Tier1 peers regardless
of independent failures. No Tier2, seed repeats, shortened training, new sampler,
task changes, scientific retries or defaults promotion belong to this campaign.

Full reservation is 25,200 seconds, 2,520 per whole recipe. Retain all original
charges in the original queue. Aggregate paid ceiling is 8,000 seconds, including
2,731.7028826579044 already paid; new headroom is 5,268.297117342096 seconds.
Every launch must fit its full original allowance. One finite round ends after
these seven new recipes; another goal-directed hypothesis needs a new bounded
registration based on its evidence. All three archives together retain the
8GiB cap and 5GiB free-space floor. Use one worker per A6000, one CPU thread,
the same scientific runtime and source as the measured controls.

Primary literature supplies context, not a gate guarantee: [R3GAN](https://arxiv.org/html/2501.05441v1)
uses beta1=0 and warns about momentum in adversarial games; the more recent
[Adam-DA analysis](https://arxiv.org/abs/2605.19392) finds first/second momentum
effects can differ from minimization. The role-factorial hypothesis is an
inference from our saved failures and supported API, not a theorem about this
finite nonconvex test suite.

Freeze all declarations and preflight receipts, commit them before admission,
and export final metrics, provenance and actual-training GIFs. Raw logs,
checkpoints and per-update traces remain in the original archive. Use the single
current technique inventory; failure leaves its representative selected.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/workflow.py prepare --manifest reports/forge/r1r2-role-repair/manifest.json --registration reports/forge/r1r2-role-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/workflow.py run --manifest reports/forge/r1r2-role-repair/manifest.json --registration reports/forge/r1r2-role-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012 --source-commit COMMITTED_SHA --submit --drain
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012/queue/events.jsonl
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/workflow.py collect --manifest reports/forge/r1r2-role-repair/manifest.json --registration reports/forge/r1r2-role-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012 --output reports/forge/r1r2-role-repair/publication --media
```

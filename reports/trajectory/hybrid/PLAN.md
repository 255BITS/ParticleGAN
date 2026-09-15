# Hybrid trajectory discriminator

Continue on `experiment/trajectory-diversity` from5012487. Test whether retaining
full-path MLP features alongside temporal features improves diversity without
sacrificing fidelity. Run MLP/continuous and hybrid/continuous on GPUs0 and1,
one worker each; 1k viability then both at10k, endpoint-only evaluation. No
seed-only runs. The repeated MLP is a matched-source control, not a seed sweep.

Hybrid D: reuse the temporal D's three kernel5 stages and four ordered pooled
bins per stage, now with widths32/64/64. A parallel two-hidden-layer MLP of
width192 reads the flattened candidate, noisy trajectory, and continuous
context. Concatenate its192 features with640 temporal features; fuse through
96 hidden units into the same eight joint-UCD heads. No class/timestep input
to either backbone branch. A single D score/loss and original-coordinate bcap
apply to the combined network; no auxiliary head or extra regularizer.

D parameters: hybrid202,280 vsMLP204,040 (-0.9%). Width is redistributed, so
this is an architecture comparison, not adding a free branch to an unchanged
MLP. G102,402 and learned latent prior640,000 parameters remain unchanged.

Both use continuous geometry within the existing coordinate bounds, the same
fixed test scenes, learned latent particles, Gaussian noise, four-step DDGAN,
joint UCD, Rp logistic, exact lazy4 bcap, VICReg, EMA and constant LR. Batch128,
10k updates =1.28M examples/phase, 2.56M real draws. Same evaluation protocol
as prior rounds (512/context); no checkpoint selection or metric-based stopping.
No modifications to loss, diffusion, prior, generator, or evaluation code.

Success: variance ratio toward1 with valid coverage, without worse conditional
SW1 or validity. Also report route calibration and collisions. Compare both
current endpoints; earlier temporal/continuous is historical context only.
Single-run evidence, with the known baseline reproducibility limitation from
the geometry round. Do not select a recipe from 1k ranks.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/trajectory/hybrid/scout_1k --log results/trajectory/hybrid/live.log -- \
  --configs 'configs/trajectory/hybrid/scout_1k/*.yaml' \
  --gpus 0,1 --workers_per_gpu 1 --trainer experiments/train_trajectory.py
tail -F results/trajectory/hybrid/live.log
```

Replace scout_1k with confirm_10k for the full comparison. Inspect metrics only
after the whole batch finishes. Finish the round, document results and a motion
completion handoff, then commit. Do not start the real motion experiment yet.

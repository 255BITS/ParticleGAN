# Trajectory diversity: discriminator × geometry coverage

Branch: `experiment/trajectory-diversity`. Four configurations at 1k for
execution viability, then all four at 10k. One worker per GPU, GPUs 0 and 1.
Inspect metrics only after each entire grid completes. No seed-only runs.
The existing baseline is repeated as a control under the same source revision.

| Arm | D | Training geometry |
|---|---|---|
| mlp_discrete | Original MLP | Original four geometries |
| mlp_continuous | Original MLP | Uniform coordinate box |
| temporal_discrete | Temporal convolutional D | Original four geometries |
| temporal_continuous | Temporal convolutional D | Uniform coordinate box |

Continuous box: start height [-.2,.2], obstacle height [-.12,.12], radius
[.22,.32], independently uniform. Same coordinate extrema as the original
four points, but broader joint coverage (not the same convex hull). Evaluation
uses the same fixed contexts in every arm. First three test geometries fall
inside this box; the fourth (.3,.18,.35) is outside on all coordinates.
An exact continuously sampled training geometry almost surely differs from a
test geometry; interpolation tests prediction within the training distribution,
not out-of-distribution generalization. The `train` evaluation split always
means the original four reference geometries, including for continuous arms;
it is not an average over the continuous training distribution.

Temporal D has three kernel-5 convolution stages at sequence lengths 64/32/16,
widths 64/128/128, LeakyReLU .2. A position channel preserves physical location.
Each stage is pooled into four ordered bins; all are concatenated with the
same continuous context and fed through a 64-unit head. 208,456 parameters vs
204,040 for MLP (+2.2%). No normalization, analytic route features, geometry
projection, or extra losses. This changes architecture/inductive bias, not an
isolated test of capacity. Gradient penalty remains on original candidate
coordinates through the full D, with noisy future/context held fixed.

Unchanged: G, learned latent particles, Gaussian terminal/forward/step noise,
four-step DDGAN posterior, joint time/class UCD, Rp logistic, exact lazy-4 bcap,
VICReg, EMA, constant LR, seed, batch128, evaluation512/context. Each 10k run
has 1.28M examples per optimizer phase / 2.56M real draws total. Reuse all shared
loss/schedule/sampling code; no changes to CIFAR or default model selection.

Main outcomes: within-route variance toward 1 alongside validity and support
error, conditional SW1, route probability calibration, interpolation versus
extrapolation validity. Variance uses valid outputs only; report missing groups
and invalidity rather than treating increased variance alone as success.
No intermediate checkpoint selection. Do not eliminate on 1k quality rank.

Run either stage (replace scout_1k with confirm_10k):

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/trajectory/diversity/scout_1k \
  --log results/trajectory/diversity/live.log -- \
  --configs 'configs/trajectory/diversity/scout_1k/*.yaml' \
  --gpus 0,1 --workers_per_gpu 1 --trainer experiments/train_trajectory.py
tail -F results/trajectory/diversity/live.log
```

Nine unit tests pass, including temporal D joint-label exclusion, candidate
input gradients and double backward, parameter budget, continuous real path
validity/bounds and unchanged held-out contexts. All training happens on CUDA.

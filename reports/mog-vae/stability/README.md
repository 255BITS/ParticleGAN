# Particle VAE without an explicit KL penalty

Yes, with a restricted posterior whose KL is constant:

```
E(X) -> one particle k(X)
z = particle[k(X)] + fixed_sigma * eps, eps ~ N(0,I)
G(z) -> reconstruction
```

Prior selection is uniform over K particles. Posterior selection is one-hot;
inside its chosen particle, posterior noise exactly matches prior noise.
Therefore joint `KL(q(k,z|X) || p(k,z)) = log(K)`. It can be removed from the
training loss without changing the objective's optima. We retain the constant
when evaluating the negative ELBO. This is a constrained variational family,
related to the [VQ-VAE constant-KL argument](https://arxiv.org/html/1711.00937v2#S3.SS1),
with Gaussian noise retained inside the chosen component. It is not a novelty claim.

The family is stochastic in z but deterministic in particle identity. It does
not learn an input-dependent offset or variance, and does not guarantee useful
output variation or balanced aggregate particle use. A valid bound also does
not imply an exact optimizer: our hard query routing uses the existing biased
straight-through gradient approximation. Soft categorical controls instead use
an unbiased score-function gradient. See the [protocol](PROTOCOL.md).

For soft `q(k|X)`, KL is `log(K)-entropy(q(k|X))`, which generally changes during
training. Simply deleting it gives our stochastic AE-GAN control. More generally,
a posterior family with fixed categorical entropy and matching conditional noise
also has constant joint KL; hard one-hot inference is the simplest special case.
Adding an unrestricted learned offset/variance generally restores a nonconstant
local Gaussian KL, so the earlier deterministic bounded-offset AE-GAN is not
retroactively a VAE under this argument.

## Completed results

All five full runs completed successfully in **2.0 minutes queue wall time**;
3.12 summed GPU-training minutes and3.25 child-process minutes including evaluation.
Nothing remains queued. Final6k updates,100k prior decoder-mean samples:

| Model | Modes /100 | HQ % | Reconstruction MSE | Posterior pair RMS | Input-mode retention |
|---|---:|---:|---:|---:|---:|
| Deterministic AE-GAN | 97 | 89.71 | **.002341** | 0 | 100% |
| Soft stochastic AE-GAN, no KL | 96 | **92.92** | .007220 | .03216 | 98.03% |
| GAN | 96 | 75.49 | — | — | — |
| Hard constant-KL VAE-GAN | 88 | 85.21 | .003264 | .01066 | 100% |
| Soft categorical VAE-GAN | 86 | 82.50 | .003884 | .02946 | 100% |

See [verified leaderboard](LEADERBOARD.md) for full metrics, same-count curves,
likelihood predictive samples, and comparison to previous learning rates.

**The answer is yes:** the hard family has a valid variational bound without
a variable KL penalty. Across65,536 posterior draws, all remain in the input's
grid mode. Shuffling the latent code raises MSE .003264 ->16.50. However,
variation is modest: pairRMS .01066 versus .02946 for the soft VAE. Exactly one
particle is selected/input, with194 effective particles used over held-out inputs.
The KL is correctly reported as5.9915nats, not zero. The actual likelihood
samples including tau=.03 noise have99 covered modes and75.76%HQ; those are
distinct from the88 modes/85.21%HQ for decoder means above. Negative ELBO4.443
versus softVAE3.916: lower reconstruction loss alone does not imply a better
bound, especially with different restricted posterior families.

**Lower learning rates help some objectives, not all.** AE-GAN now improves
smoothly at2k/4k/6k (82/96/97modes;62.80/87.49/89.71%HQ) and its final MSE
roughly halves versus the original round. The soft no-KL control has the best
final HQ but worse reconstruction and a large4k dip (60modes/49.38%HQ). Hard
VAE loses coverage late (96 ->88), and softVAE also falls95 ->86 despite HQ
rising. Thus stability is not solved across the board. GAN retains best SW1
(.1685), while all mode-width ratios remain too small (.40–.60 of real).
No single model wins every metric; one shared-seed trajectory is not robust
superiority evidence.

**Recommendation:** keep lower-LR deterministic AE-GAN as the reconstruction
baseline. If we want particle-choice uncertainty without a variable KL penalty,
a clean next toy ablation is `E(X)->a set of exactly m particles; sample uniformly
from that set; add fixed prior noise`. Its joint KL is also constant, `log(K/m)`.
Withm=4 it could retain uncertainty over particle identity, unlike the hardm=1
family. Choosing the set still needs an approximate discrete gradient; it is
not a guaranteed improvement. This is a proposed next experiment only, not
launched. Keep intermediate checkpoints and usage/width/retention metrics;
defer images until choosing the desired tradeoff.

Five lower-learning-rate configurations use the same initialization/data/prior
draws and budget: GAN, deterministic AE-GAN, stochastic categorical AE-GAN without
KL, soft categorical VAE-GAN, and hard constant-KL particle VAE-GAN. Every generation
evaluation uses100k samples at2k/4k/6k. All learning rates are halved. Tau=.03 is
fixed for every likelihood evaluation; prior sigma stays fixed. No seed sweep.

Validation:24tests plus13subtests pass, including exact hard-forward/noise,
constant joint density ratio, correct true-posterior diagnostics and the query
surrogate gradient. A dtype issue exposed by the double-precision known-answer
test was corrected before the five implementation pilots; all five pilots pass.
The previous scout's sources/certificates remain untouched.

```bash
# From /home/martyn/dev/ParticleGAN-mog-autoencoder
tail -f runs/mog_vae/stability.live.log

/home/martyn/dev/ParticleGAN/.venv/bin/python -u experiments/follow_grid.py \
  --root runs/mog_vae/stability --log runs/mog_vae/stability.live.log -- \
  --configs 'configs/mog_vae/stability/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
  --python /home/martyn/dev/ParticleGAN/.venv/bin/python \
  --trainer experiments/train_mog_vae_stability.py

/home/martyn/dev/ParticleGAN/.venv/bin/python experiments/analyze_mog_vae_stability.py
```

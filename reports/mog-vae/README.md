# Particle VAE toy experiments

**Follow-up completed:** [lower-LR round and constant-KL particle VAE](stability/README.md).
The results below describe the original12-arm scout.

All 12 full runs completed on both GPUs in **4.4 minutes wall time**, with
**7.57 GPU-minutes of training** (7.89 summed child-process minutes including
evaluation/I/O). No full-run failures; nothing remains queued. One shared seed,
6,000 updates per arm, final online weights. These are matched new controls,
not reuse of the earlier toy trajectories.

| Model / setting | Modes /100 | HQ % | Sampled reconstruction MSE |
|---|---:|---:|---:|
| Stochastic particle AE-GAN, no KL | **95** | **76.01** | **0.004246** |
| Categorical VAE-GAN, sharp, tau=.03 | 90 | 72.92 | 0.005626 |
| Deterministic particle AE-GAN | 84 | 73.50 | 0.004629 |
| GAN | 80 | 59.31 | — |

These generation numbers evaluate decoder means `G(z)` on 100k prior samples.
Full [leaderboard](LEADERBOARD.md) includes all 12 settings, likelihood sampling,
posterior diagnostics, learning curves, costs, and matched-count stability audit.

**What worked:** a genuine VAE with categorical uncertainty over particles,
fixed local noise and no learned per-input Gaussian ball. The best VAE by our
coverage-first ranking uses approximately 3.23 effective particles per input;
eight posterior draws on 8,192 held-out inputs give pair RMS .03864 while all
65,536 draws retain the input's grid mode. Shuffling the code raises MSE from
.005626 to 16.23, so the code carries the input. This demonstrates useful
stochastic encoding here, not complete within-mode distribution modeling.

**What did not win:** adding KL did not beat the stochastic no-KL control at
this budget. The no-KL control is a stochastic AE-GAN, not a VAE. Broader routing
and removing GAN hurt this scout. Learning local variance did not beat the
simpler categorical model: its final std stayed near the prior (.982/.997),
with very small local KL (.00124/.00010). This is evidence for keeping the
simpler family in the next toy round, not proof local inference is unnecessary.
The GAN still has the best global SW1 (.2755) and a closer mode-width ratio
(.771) than the leading stochastic control (.668): no universal winner.

**Stability is unresolved.** The frozen 4k checkpoint audit uses the same
100k sample count/RNG as final evaluation. AE-GAN falls from 99 modes / 92.67%
HQ at 4k to 84 / 73.50% at 6k; GAN from 94 / 76.78% to 80 / 59.31%.
The leading stochastic control improves 91 / 71.81% to 95 / 76.01%; categorical
VAE sharp/.03 improves 76 / 48.87% to 90 / 72.92%. Thus the final ranking does
not establish general superiority over AE-GAN. No earlier checkpoint is silently
substituted into the final leaderboard. See [audit](late_audit.json).

**Likelihood caveat:** a VAE's Gaussian decoder models `G(z)+tau*noise`.
For the leading VAE at tau=.03, those actual likelihood samples get 67.12% HQ
versus 72.92% for decoder means. Tau=.1/.3 gives only roughly 16–28% / 4% HQ,
so these are poor likelihood calibrations for modes whose true std is .03.
Categorical KL is about 4.8 nats, of which approximately log(100)=4.605 is
categorical mutual information: the posterior largely identifies the grid mode.
A useful code is not equivalent to matching aggregate posterior to prior; the
leading VAE's aggregate categorical TV is still .248.

**Recommendation:** stay on the toy problem for one matched stability round:
GAN, deterministic AE-GAN, sharp stochastic no-KL AE-GAN, and sharp categorical
VAE-GAN with tau=.03; halve all learning rates, retain intermediate checkpoints,
and compare fixed-count curves. Keep prior sigma fixed and defer learned local
variance/images. Fix tau=.03 for subsequent likelihood comparisons; changing
tau changes the likelihood model as well as its KL weight. These are proposed
next experiments only; none have been launched. No seed-only sweeps.

Twelve configurations compare deterministic particle AE-GAN with genuine
categorical particle VAE and VAE-GAN. The main posterior samples particle
identity and uses the fixed prior noise inside each particle; its only
variational penalty is categorical KL. Two additional arms learn a local
Gaussian posterior. See [protocol](PROTOCOL.md) and [leaderboard](LEADERBOARD.md).

Implementation: `experiments/train_mog_vae.py`; analysis:
`experiments/analyze_mog_vae.py`; configs: `configs/mog_vae/`.
Raw runs/checkpoints/source archives remain under ignored `runs/mog_vae/`.

```bash
# From /home/martyn/dev/ParticleGAN-mog-autoencoder
tail -f runs/mog_vae/scout.live.log

/home/martyn/dev/ParticleGAN/.venv/bin/python -u experiments/follow_grid.py \
  --root runs/mog_vae/scout --log runs/mog_vae/scout.live.log -- \
  --configs 'configs/mog_vae/scout/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
  --python /home/martyn/dev/ParticleGAN/.venv/bin/python \
  --trainer experiments/train_mog_vae.py

/home/martyn/dev/ParticleGAN/.venv/bin/python experiments/analyze_mog_vae.py
```

Validation: 22 tests plus 13 subtests passed (exact-enumeration categorical
gradient, KL against torch distributions, Gaussian likelihood scaling, queue
regressions). Five 200-step pilots passed with matched initialization and
data/prior RNG states. The initial pilot attempts exposed an evaluation-list
unpacking error and undefined sparse-sample core-width; these were fixed before
full training. Failed attempts are preserved in the pipeline's history folder.
The combined pilot log includes the original failures; `pilots.json` records
the five verified successful replacement attempts. No full-run settings were
selected from pilot quality metrics.

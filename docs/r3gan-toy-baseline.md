# Configurable Modern GAN training baseline

The [Forge idea](../configs/forge/ideas/r3gan-stacked-training-toy-v1.json)
transfers the [Modern GAN paper, Appendix D Table 19](https://arxiv.org/html/2501.05441v1#A4)
Stacked MNIST training profile to the existing toy hosts. The earlier
`k3p-r1r2-matched-v1` changes only the gradient penalty inside a K3P recipe;
it remains a separate technique.

| Recipe setting | Configured value |
| --- | --- |
| `optimizer_family`, `eps`, `amsgrad` | `adam`, `1e-8`, `false` |
| `lr`, `d_lr_mult`, `prior_lr_mult` | `0.0002`, `1`, `1` |
| `lr_floor`, `network_lr_floor`, `network_lr_horizon_cap` | `1`, `1`, `null` |
| `betas`, `beta2_end`, `beta2_anneal_end` | `[0, 0.9]`, `0.99`, `0.2` |
| `reg_arm`, `reg_coeff`, `reg_coeff_end`, `reg_coeff_anneal_end` | `a_r1r2`, `1`, `0.1`, `0.2` |
| `reg_every` | `1` |
| `input_noise_std`, `output_noise_std` | `0`, `0` |
| `reg_anchor_weight`, `d_guard_ratio`, `latent_damping_max_rate` | `0`, `0`, `0` |
| `direct_particle_gain` | `false` |

The beta2 and penalty schedules interpolate with a cosine, using completed
updates before the next update. They hold at their endpoint after the declared
fraction of `total_steps`. Here the paper's 2/10 million-image burn-in becomes
20% of each host's frozen update horizon. At 0%, 10%, and 20% of that horizon,
beta2 is 0.9, 0.945, and 0.99; gamma is 1, 0.55, and 0.1.
The penalty is gamma/2 times the sum of the mean squared real and fake input
gradient norms. The [pinned reference implementation](https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/R3GAN/Trainer.py)
and [training loop](https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/training/training_loop.py)
provide the loss and update-order comparison.

All listed values are ordinary public `Recipe` fields, honored by the shared
factories, `GANTrainer`, `UpdatePolicy`, and Forge's component adapters. Explicit
endpoint fields enable the new schedules; omitting them retains fixed beta2 and
gamma. Standalone callers can use
`particlegan.recipe_schedules.apply_training_schedules(completed_steps, recipe,
optimizers, penalty)` before their forwards, alongside their LR helper. The fixed
penalty also observes the critic optimizer's checkpointed completed-update count
before computing gamma. Plain Adam rejects configurations requesting K3P
interventions instead of silently discarding them.

The hosts retain their declared architectures, initialization, auxiliary losses,
learned priors or particle exceptions, named streams, budgets and clean/live
evaluation. This experiment does not reproduce the paper's image architecture,
fixed Gaussian latent prior, image count, or EMA image evaluation. The configured
`ema_decay=0` is unused by live scoring. A failed 80-update transport screen
describes that host and budget, and does not establish failure of the paper's
longer experiment.

Plan and run this substantive recipe revision through the ordinary gates:

```sh
.venv/bin/python -m experiments.forge plan r3gan-stacked-training-toy-v1 --through-tier 3
.venv/bin/python -m experiments.forge run r3gan-stacked-training-toy-v1 --through-tier 3 \
  --campaign configs/forge/campaigns/r3gan-stacked-training-toy-v1.json --gpus 0,1
.venv/bin/python -m experiments.forge logs --follow --campaign r3gan-stacked-training-toy-v1
```

The campaign declares a 44,100-second reservation ceiling. Required failures stop
remaining work. No seed repetitions or post-failure diagnostics follow. To test a
different substantive configuration, create a new idea and campaign id, document
the changed factors, inspect the plan, and preserve the resulting source and task
bindings. New cards enter the technique inventory automatically. Reporting and
frozen-source replay launch no training; see [the inventory workflow](../EXPERIMENTATION.md).

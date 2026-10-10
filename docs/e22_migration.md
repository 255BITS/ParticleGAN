# Migrating to the reusable E22 policy

Use the installed recipe and factories instead of copying the research JSON
or assembling the old regularizer primitives:

```python
from particlegan import GANTrainer, get_recipe, init

recipe = get_recipe("e22", num_particles=20_000, z_dim=2,
                    batch_size=2048, output_noise_std=0.029,
                    reg_anchor_min_decay=0.9)
init.deterministic_orthogonal_(D, seed=1)  # fresh networks only
trainer = GANTrainer(recipe, G, D, prior=prior)
for real in batches:
    trainer.step(real)
```

For an existing caller-owned loop, construct `E22Policy` with those same
modules and explicit optimizer roles, then add its hooks around the existing
backward/optimizer calls. Follow
[`examples/e22_external_loop.py`](../examples/e22_external_loop.py) and the
[lifecycle table](e22.md#caller-owned-updates). `GANTrainer` and `E22Policy`
share the implementation; the application does not need to reproduce LR,
row-evidence, noise, birth/death or serving controls.

| Earlier integration | Current integration |
| --- | --- |
| `from particlegan import GradientPenalty` or `recipe.make_gradient_penalty(...)` | `penalty = recipe.make_critic_penalty(opt_d)`; add `penalty(D, real, fake)` to the critic loss. Pass it as `penalty=` to the policy or call `policy.attach_penalty(penalty)`. |
| Top-level `CriticAnchor`, `CriticSpikeGuard`, `LatentRowDamping`, `DirectParticleResponse`, `K3PCritic`, or their `make_*` regularizer factories | Build current optimizers with `recipe.make_optimizers(...)` (or the role-named optimizer factories). Their ordinary `step()` performs the formulation's regularization work. |
| `LOCKED_SHARED`, `locked_adv_defaults`, `make_b_cap`, `make_gan_loss` | Build a current loss with `recipe.make_loss()` and current penalty with `recipe.make_critic_penalty(...)`. Archived benchmark replays live under `benchmarks.legacy`, outside the installed API. |
| `initialize_(module, key=k)` | `init.deterministic_orthogonal_(module, seed=k)` before creating a fresh run. |
| `reg_anchor_decay=0.999` | Remove this option. KA2 uses `reg_anchor_min_decay=0.9` by default: the lower bound of its moment-surprise-controlled EMA decay. It is not a numerical replacement for a fixed .999 decay. Choose another lower bound explicitly only when intended. |
| `scale_learning_rates(...)` for the E22 preset | `policy.begin_step(...)` owns stationarity rates. Keep the schedule helper for recipes without stationarity control. |

KA2 replaces the earlier K3P critic formulation. Trainer checkpoint schemas
1–3 cannot resume under the current formulation; use the release that wrote
them (0.8.0 for K3P), or start a fresh run. Compatible schema-4 trainer
checkpoints retain the native loader. A policy's complete `state_dict()` is a
separate recovery format; rebuild the same recipe, parameter roles and
optimizer layout before loading it, and save the application's data cursor
alongside it.

The full E22 row mechanisms require unconditional independent particles.
Conditional and densely soft-routed banks raise explicitly. Their applications
can use `UpdatePolicy` with both row evidence and birth/death disabled and
the appropriate `row_semantics`; they need their own restructuring design.

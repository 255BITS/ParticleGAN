# Locked shared

Importable demo adversarial stamp. Other code can depend on it. This page is
the contract. Lunar Lander imports it from
`configs/gym/lunar_lander_particle_finetune/locked_shared.yaml`
([gym note](gym-particle-finetune.md#locked-shared-arm)).
`particle.yaml` is still the YuE2 cap (`lazy_k=4`), not this stamp.

```python
from particlegan.locked_shared import LOCKED_SHARED, locked_adv_defaults, make_gan_loss, make_b_cap

stamp = locked_adv_defaults()   # frozen mapping
loss = make_gan_loss()          # GANLoss("logistic", "rp")
penalty = make_b_cap()          # GradientPenalty b_cap, coeff=1, kappa=1, norm=l2, lazy_k=1
```

`LOCKED_SHARED`, `locked_adv_defaults`, `make_gan_loss`, and `make_b_cap`
are also exported from `particlegan`. The builders accept only
`LOCKED_SHARED`. A drifted copy raises `ValueError`.

`n_particles`, `particle_l2`, and `z_dim` describe the demo cloud. Use them
when you build particles. The builders do not construct a prior and do not
add `particle_l2` to the GAN loss. A host prior keeps its own width. The
critic stays the caller's (`critic="host"`).

## Stamp

| Field | Pin | Also called |
| --- | --- | --- |
| `loss_type`, `gan_mode` | logistic, `rp` | RpGAN logistic |
| `reg_arm`, `reg_coeff`, `reg_kappa`, `reg_norm` | `b_cap`, 1, 1, `l2` | slider `grad_arm` / `b_cap` / `kappa` / `grad_norm` |
| `lazy_k` | 1 | slider `grad_lazy`, conceptmod `reg_lazy` |
| `reg_method`, `target_anneal` | `autograd`, `none` | `GradientPenalty` defaults |
| `fm_weight` | 0 | feature matching off |
| `cover_weight`, `cover_posture` | 1.5, `demo` | demo cover, not Music |
| `n_particles`, `particle_l2`, `z_dim` | 12, 0.02, 2 | tiny demo cloud, when particles apply |
| `pairing` | `live` | not a stranger row |
| `critic` | `host` | conceptmod `critic_arch`; not a Music MLP |
| `reg_impl` | `grad_regularizer` | `GradientPenalty`, whose center is `kappa` |

`make_b_cap()` returns `GradientPenalty` itself, so `center(step)` follows
`kappa`. A thinned cap that stores kappa and hardcodes the center is not
this object.

## Not this stamp

| Name | What it is |
| --- | --- |
| Music cover 1.0 | `cover_weight=1.0`, `cover_posture="music"` |
| hub128 | 128-particle gmix cloud |
| FM-on | any `fm_weight` other than 0 (named probe: 0.1) |
| stranger | `pairing="stranger"` |
| thinned κ | stored kappa, penalty center hardcoded (1 in one probe, 100 in another) |
| `Recipe("gan")` | 20_000 particles, VICReg `prior_reg=1`, no cover and no FM field |
| YuE2 gym controller | same cap, `lazy_k=4` (`EDIT_CAP_EVERY` in `lib/gym_particle_finetune.py`; still `particle.yaml`) |

`drift(name)` returns the first five rows as `LockedShared` copies.
`NAMED_DRIFTS` is that table.

This is not a Music or Anima GPU transfer. It does not run a conceptmod toy.
The Lunar apply is the gym config above. It does not re-extract this stamp.
GPU landings for that arm have not been run.

## Measured conceptmod verification

The [behavioral leaderboard](../reports/locked_shared/README.md) runs extracted
two-pole, shared-trajectory and ring-diversity experiments through these
builders. Every variant trains and is scored from measurements; no config
equality or refusal is counted as a behavioral result.

The recorded run matches all ten original conceptmod runs exactly, including
negative controls. That establishes builder/extraction parity. It does not
reproduce the original all-pass claim: two-pole passes, trajectory fails,
and the ring result is inconclusive. See the table for metrics, provenance
and [reproduction commands](../benchmarks/locked_shared/SOURCE.md).

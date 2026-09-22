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

The latest [convergence leaderboard](../reports/behavioral_baseline/convergence/README.md)
adds a shared cosine schedule to the best cap: decay from 60% of the budget to a
5% floor. It passes 29/29 bounds and sustains all nine toys, including 8/8 ring
modes. Ten independent application checks pass. Sensitivity and the actual
100-Gaussian comparison support retaining the stock defaults, with
`get_recipe("gan_behavioral")` as an opt-in candidate. The
[public training helper](api.md#gantrainer) applies the recipe consistently.
The following paragraph describes the earlier search using original host schedules.


The expanded [live-weight baseline](../reports/behavioral_baseline/README.md)
trains each candidate on nine toys and requires all 29 numerical bounds. The
leading `b_cap` candidate uses cap target 1.25, coefficient 3, no particle L2 and
LR multiplier 0.85: all nine toys pass, with 8/8 live modes at 100% HQ. All five
late ring checkpoints keep eight modes; four also meet 90% HQ. R1+R2 at
coefficient 0.1 passes the original bounds with seven final modes. Ten shared geometry
and application checks also pass and are shown separately. See the
[protocol](../benchmarks/locked_shared/BASELINE.md) for config fields, fixed
budgets, scope, and commands for comparing another approach. EMA cannot rescue a
live failure. The [search ledger](../reports/behavioral_baseline/search/README.md)
also preserves failed and incomplete attempts.

The leaderboard now exposes actual ring mode counts, HQ, effective modes and
late-checkpoint stability. Its original regression threshold permits 7/8 modes;
that minimum PASS alone does not select a production default. See the
[coverage and default-selection analysis](../reports/behavioral_baseline/default_selection.md).

The [behavioral leaderboard](../reports/locked_shared/README.md) runs extracted
two-pole, shared-trajectory and ring-diversity experiments through these
builders. Every variant trains and is scored from measurements; no config
equality or refusal is counted as a behavioral result.

The recorded run matches all ten original conceptmod runs exactly, including
negative controls. That establishes builder/extraction parity. It does not
reproduce the original all-pass claim: two-pole passes, trajectory fails,
and the ring result is inconclusive. See the table for metrics, provenance
and [reproduction commands](../benchmarks/locked_shared/SOURCE.md).

The [follow-up comparison](../reports/locked_shared/comparison.md) finds that
removing the host particle L2 term clears all three measured targets while
keeping RpGAN and `b_cap`. Other existing formulations have tradeoffs. This
is recorded as an experimental candidate; it does not change the stamp.

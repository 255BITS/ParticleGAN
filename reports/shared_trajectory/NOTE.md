# Shared-trajectory pairing

CPU toy. Same-seed slow and fast arcs are one relativistic pair. A stranger
or nearest-stranger fast arc is a different identity.

This pass is not a Music or Anima GPU result. Cover here is demo **1.5**
(12-particle cloud). Music Arm B pole/cover **1.0** is a different host and
is not this run.

## What was reviewed

- `HyperGAN/particle-sliders` `analysis/slider2d/locked_baseline_defaults.py`
  (locked #94: RpGAN-side `b_cap` coeff=1, kappa=1, norm=l2, FM 0,
  `particle_l2` 0.02, n=12, cover 1.5, VICReg 0.05).
- `analysis/gan_bcap/paired_fm_findings.md` and `tests/test_gan_paired_fast.py`:
  row-paired targets keep identity; a marginal / mismatched pair does not.
  The fast check also rejects a duplicated seed as two observations.
- `docs` Music transfer card: pole 1.0, FM 0, parts 0. Not copied onto this
  particle cloud.
- This repo had no gym or Lunar toy to extend. The slow→fast rule is enforced
  on a 12-seed arc family instead.

## Recipe

RpGAN logistic, `GradRegularizer` `b_cap` coeff=1 kappa=1 norm=l2 (not a
thinned stub), `fm_weight=0`, demo cover 1.5, `particle_l2` 0.02, VICReg
0.05, n=12, lr 5e-3, betas (0, 0.99), 400 steps, seed 0. The critic is this
toy's MLP. `TrajectoryDiscriminator` is unchanged.

The locked entry point only trains `pairing=shared`. Stranger and
nearest-stranger raise. Drift arms use the same modules; the only delta is
the fast-target index.

Identity MSE is against the fast arc of the **same** seed. Pass line is
**0.02**. A perfect copy of the nearest other fast arc still sits near 0.09,
so that line is below the nearest-stranger floor.

## Leaderboard

| arm | pairing | identity MSE | gate |
|---|---|---:|---|
| locked_shared | shared | 0.00167 | PASS |
| drift_nearest_stranger | nearest other slow arc | 0.297 | FAIL |
| drift_stranger | opposite seed (shift 6) | 0.567 | FAIL |

Lower is better. One seed. No second seed of the same arm.

## Why

The critic scores `(slow_i, fast)`. Shared pairing teaches
`fast` of seed `i`. A shuffled fast target teaches a different seed, and
demo cover only asks the generated cloud to cover the true set. Cover does
not restore identity. Nearest-stranger is the sharper failure: the partner
looks close in the slow arc and still misses the fast arc (0.297 vs 0.00167).

## Recommendation

Keep the locked refusal. Do not train slow→fast with stranger or
nearest-stranger targets. Do not treat this PASS as Music or Anima transfer.
A later motion toy should keep shared-identity pairs and change the data,
not the adversarial recipe.

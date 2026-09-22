# Locked-shared RpGAN + b_cap regression floor

CPU toy and pytest gate for the particle-sliders locked_shared / #94 shape,
scored with this repo's `GANLoss` and `GradientPenalty`. A PASS here is
formulation identity on one seed. It is not a Music or Anima GPU transfer.

```bash
python -u analysis/locked_shared_floor.py   # JSONL; tail -n 1 is the board
python -m pytest tests/test_locked_shared_floor.py -q
```

## Stamp

Demo cover posture, not the Music transfer row.

| knob | locked_shared floor | particle-sliders name |
|---|---|---|
| RpGAN logistic pair | `GANLoss("logistic", "rp")` | `rp_d_loss` / `rp_g_loss` |
| faithful b_cap | `GradientPenalty` coeff=1, κ=1, norm=l2, lazy=1, anneal=none | `b_cap` / `kappa` / `grad_norm` |
| FM | `fm_weight=0` (term omitted) | `fm_weight=0` |
| cloud | 12 particles, z_dim=2, `particle_l2=0.02` | demo `n_particles=12`, `particle_l2=0.02` |
| cover | **1.5 demo** mode-pin on two centers | demo `cover_weight=1.5` (Music Arm B is 1.0) |

Leftover / Field3D leak is not scored: this toy has no unused axis.
VICReg is not in the floor (`Recipe.prior_reg=1` is a different family).
The critic is a caller-owned MLP. The gate never reads its layers.
`Recipe("gan")` stays the 100-Gaussians default (20_000 particles, no FM, no cover).

`GradientPenalty` is an alias of `GradRegularizer`. The gate requires that
class object, not a subclass.

## Leaderboard

One seed (`0`), 8 Adam steps, CPU. Errors are absolute distance from the
locked formula. `kappa_probe` is ‖∇D‖=0.4 at κ=0.2 (faithful penalty 0.04).

| arm | verdict | adv err | g err | cap err | κ probe |
|---|---|---:|---:|---:|---:|
| `locked_shared` | **PASS** | 0 | 0 | 0 | 0 |
| `thinned_kappa` | FAIL | 0 | 0 | 0 | **0.040** |
| `fm_on` (0.1) | FAIL | 0 | 7.28e-4 | 0 | 0 |
| `vanilla_logistic` | FAIL | 0.693 | 0.0231 | 0 | 0 |
| `stranger_pair` | FAIL | 2.44e-4 | 1.19e-6 | 0 | 0 |
| `music_cover_1` | FAIL | 0 | 0.233 | 0 | 0 |
| `hub128` | FAIL | 0 | 0 | 0 | 0 |

## What the errors mean

The thinned arm stores κ and then hardcodes the cap center to 1. At the
locked κ=1 its step penalty matches `GradientPenalty` exactly (cap err 0).
The κ=0.2 probe is the regression: a slope of 0.4 is under a cap of 1 and
over a cap of 0.2, so the faithful penalty is 0.04 and the thinned penalty
is 0. A check that only reads the stored κ would pass this arm.

FM-on, vanilla logistic, reversed (stranger) pairs, Music cover 1.0, and a
128-particle cloud all fail the same gate. The 128-particle row is the Hub
gmix size; it is not a locked cloud. `Recipe("gan")` at 20_000 particles is
the study prior, also not this stamp.

## Recommendations

Keep `tests/test_locked_shared_floor.py` as the CI check for this family.
Do not drop the κ probe. Do not retune `Recipe("gan")` down to 12 particles
to make the floor look like the library default. Do not treat a toy PASS as
evidence that Music Arm B or an Anima run matches this recipe: those add a
trainer, a leftover gate, and a GPU eval the floor does not run.

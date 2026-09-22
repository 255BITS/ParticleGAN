# Unipolar direction toy (CPU)

One family. One-sided residual / direction match in the sense of
[particle-sliders `FORMULATION_LEADERBOARD_UNIPOLAR`](https://github.com/HyperGAN/particle-sliders/blob/master/docs/FORMULATION_LEADERBOARD_UNIPOLAR.md):
train scales `{0, +1}` only, score plus cover, plus leak, and neutral hold.
Scale `-1` is an unscored canary. This is not a Music or Anima result.

## Locked shape vs intentional drift

Matches locked_shared / Arm B on the adversarial triple:

| knob | value |
|---|---|
| objective | RpGAN logistic (relativistic pair), `particlegan.GANLoss` |
| gradient penalty | `GradientPenalty` `b_cap`, coeff=1, kappa=1, norm=l2, lazy=1, anneal=none |
| FM | `fm_weight=0` |
| LR | `5e-3`, Adam betas `(0, 0.99)`, delayed cosine hold 80, floor 0.05 |
| critic | scale-conditioned MLP, hidden 64 (locked `critic_hidden`) |

Reported drift (not silent):

| knob | this toy | locked demo / Music |
|---|---|---|
| cover | **train weight 0**. Cover is the eval gate `>= 0.85`. | demo locked cover **1.5** (Music sometimes 1.0). A nonzero cover loss is supervised MSE; the unipolar GAN winner is GAN-only. |
| particles | none (residual student, same posture as `rpgan_bcap_plus_neu`) | Music `n=12`, `particle_l2=0.02`. Not Hub 128 gmix. |
| steps | 400 (unipolar leaderboard budget) | locked 1200 |

## Gates

`delta(s) = s*odd + |s|*even + origin` with a **free** origin (hold is learned).

* **cover** at `+1`: `max(0, cos(delta, +e0)) * max(0, 1 - || ||delta||/||+e0|| - 1 ||)`. Need `>= 0.85`.
* **leak**: fraction of `delta(+1)` energy on the off-caption axis `e1`. Need `<= 0.05`.
* **neu_hold** at `0`: `1 - min(1, ||delta(0)||)`. Need `>= 0.85`.

## Leaderboard (seed 0, 400 Adam steps, CPU)

| arm | cover | leak | neu_hold | cos(+1, +e0) | hit |
|---|---:|---:|---:|---:|---|
| `locked_rpgan` | 0.9639 | 0.0005 | 0.9557 | +0.9995 | **PASS** |
| `mse_only` | 1.0000 | 0.0000 | 0.6667 | +1.0000 | **FAIL** |
| `polarity_flipped` | 0.0000 | 0.0000 | 0.9632 | -1.0000 | **FAIL** |

`mse_only` is plus-only coordinate MSE (no scale-0 term, no RpGAN, no `b_cap`). The three free vectors split the plus pole evenly, so `delta(0)` stays at `1/3` and neu_hold is `2/3`. Cover of `+1` still passes. That is the `faithful_plus` failure: plus match without a learned neutral.

`polarity_flipped` keeps the locked GAN and `b_cap` and trains the `+1` slot on the minus pole. Neutral hold still passes. Plus cover is 0.

Scale `-1` stayed unscored. On the PASS arm it landed toward the minus pole (canary, not a gate).

## Run

```bash
python examples/unipolar_dir_toy.py
python -m pytest tests/test_unipolar_dir_toy.py -q
```

Lines look like `unipolar_dir arm=... step=... cover=... leak=... neu_hold=... hit=PASS|FAIL`.

## Recommendation

Keep the PASS definition on this adversarial shape. Do not "fix" the MSE fail by deleting the origin, and do not put demo cover 1.5 back on this arm without a new gate: that turns it into the supervised loss the unipolar GAN winner excludes. A CPU pass here does not transfer to Music or Anima; the next real check is the particle-sliders unipolar board (`--polarity uni`, seed 0, 400 steps), not another seed of this toy.

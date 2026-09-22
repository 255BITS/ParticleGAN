# Formulation leaderboard honesty (CPU toy)

PASS is not a high score. The harness marks PASS only when the
`locked_shared` cell wins this toy and every declared bad arm fails the
same toy. An empty negative list raises `HonestyError`.

CPU only. 80 Adam steps, seed 0. This is not a Music or Anima GPU result,
and it does not move the package's live recipe defaults.

## Locked cell

Demo cover **1.5** (the Music-transfer posture is 1.0; this toy does not
use that). Leftover does not apply: there is no unused axis.

| knob | value |
|---|---|
| loss | RpGAN logistic (`GANLoss`, mode `rp`) |
| penalty | `GradRegularizer` `b_cap`, coeff 1, kappa 1, norm l2 |
| FM | off (`fm_weight=0`) |
| cloud | `n_particles=12`, `particle_l2=0.02` |
| cover | 1.5, logged on `cover_score`, not a generator attraction |
| critic | host 1-D MLP, shared, not swapped |

The cell wins when the cloud leaves the origin (`mean_abs >= 0.30`) and
the median critic slope on reals plus the live cloud stays at or below
kappa (`grad_med <= 1`). Both-pole balance is not a gate.

## Board

| order | arm | drift | won | mean_abs | grad_med | cover_score |
|---|---|---|---|---:|---:|---:|
| 1 | `locked_shared` | — | **yes** | 0.514 | 0.420 | 0.772 |
| 2 | `stranger_pairing` | pairing `live` → `stranger` | no | 0.000 | 0.073 | 0.000 |
| 3 | `thinned_b_cap` | κ hardcoded at 100 | no | 0.694 | 2.342 | **1.041** |

`thinned_b_cap` posts the higher cover score and still fails: the hinge
never binds, so `grad_med` sits above kappa. Ranking by cover alone would
crown that fail. Stranger pairing keeps a legal slope and never moves the
cloud, because the relativistic pair is not the live particles.

Unreported drift (FM on, a new GAN mode, a critic swap) is refused before
a score exists. A 128-particle cloud is refused; this toy does not adopt
a Hub gmix.

```bash
python -m particlegan.leaderboard_honesty
pytest tests/test_leaderboard_honesty.py -q
```

# Ring mode-hold under RpGAN + b_cap

CPU toy. One family. Seed 0, 1200 steps. A PASS here is **not** a Music or Anima transfer.

The particle-sliders 8-Gaussian smoke still covers with the cap off, because those particles start on the ring. This ring starts the 12-particle cloud at the origin (`init_std=0.5`) and keeps the 100-Gaussians Fourier critic (`SimpleMLPDiscriminator`). Without the cap that critic strands modes. With the locked cap it does not.

## Locked card

| knob | value | note |
|---|---|---|
| pairing | RpGAN logistic | `GANLoss(mode='rp')`, not vanilla |
| regularizer | `b_cap` coeff=1 kappa=1 norm=l2 | host `GradientPenalty`, kappa explicit |
| FM | 0 | FM-on is refused unless the drift is named |
| cover | **1.5 demo pin** | recorded, not a training loss. Music pole/cover 1.0 is not this toy |
| particles | n=12, `particle_l2=0.02` | tiny cloud. Not Hub 20k and not a 128-particle gmix |
| vicreg | 0.05 | host `ParticleRegularizer` |
| critic | `SimpleMLPDiscriminator` | host critic, not an MLP swap |
| budget | 1200 steps, Adam β1=0, β2=0.99, EMA 0.995, lr 2e-3 constant, D mult 1 | lr is this toy's budget, not the slider 5e-3 |

Geometry: 8 Gaussians on a radius-3 ring, σ=0.07. HQ = inside 3σ of a center. Those balls do not overlap.

Tail the run:

```bash
PYTHONPATH=. python -u lib/mode_hold.py
PYTHONPATH=. python -u -m pytest -s tests/test_mode_hold.py
```

## Leaderboard

Endpoint, EMA weights, 4096 samples, seed 0.

| arm | drift from locked_shared | modes | HQ | cover | effective modes | verdict |
|---|---|---:|---:|---:|---:|---|
| locked | none | 8/8 | 1.000 | 1.000 | 7.48 | **PASS** |
| b_cap off | `reg_arm` b_cap → `f_none`, coeff 0 | 1/8 | 0.090 | 0.125 | 1.00 | **FAIL** |

Mid-run (same seed, EMA): locked is still empty at step 400, 7/8 at step 800 (HQ 0.751, not yet the veto), and held at 8/8 from the step-1200 read. Cap-off is at 0 modes through step 800 and 1 mode at step 1200. The gate reads the endpoint only.

Refused without a drift report (not trained): stranger pairing (`gan_mode=vanilla`), FM-on (`fm_weight=0.1`), and any other moved knob. A kappa-hardcoded stub (`relu(||g||−1)^2`) is silent on a slope of 0.5; `GradientPenalty(..., kappa=0.2)` is not.

## Why the cap holds the ring

RpGAN's pair needs a critic slope to tell an empty mode from a covered one. `b_cap` charges only for `||grad D|| > 1`, so the Fourier critic can stay steep enough to point at missing modes and cannot run away. Turning the same penalty off lets the critic saturate on the first mode it finds. The generator then has no usable gradient toward the other seven, and the cloud stays there (1/8, effective modes 1).

The veto is modes ≥ 7 and HQ ≥ 0.90 to PASS, modes ≤ 2 to FAIL. A point mass at the origin scores 0 modes and FAILs the same function, so the collapse branch is not an empty assertion.

## Recommendation

Keep this card when the question is mode hold on a small mixture: RpGAN logistic, host `b_cap` at coeff=kappa=1, L2, FM off, 12 particles, `particle_l2=0.02`. Report any adv change before training it. Do not read a toy PASS as evidence the same weights would hold Music or Anima. Do not copy Music cover 1.0 or a 128-particle gmix onto this card without naming the drift.

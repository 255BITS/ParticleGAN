# Orbit radius hold (CPU toy)

Closed-loop feedforward on a circle. A constant-speed tangential command,
stepped with explicit Euler, keeps heading and linear speed on target while
the radius walks out:

```
r_{k+1}^2 = r_k^2 + (dt * omega * R)^2
```

One ablated step at the pinned protocol (`R=1`, `omega=1`, `dt=0.25`) already
inflates radius by `sqrt(1.0625) - 1 ≈ 0.0308`, which is above the radius cap
`0.02`. Heading on that step is exactly tangential and linear speed is exactly
`omega * R`. The gate therefore requires radius hold. Heading (`cos >= 0.98`)
and speed (`| |v|/(omega R) - 1 | <= 0.05`) are necessary and not sufficient.

The closed-loop residual head adds the inward radial velocity that puts each
Euler step back on the circle. Ablating that head (zero residual) is the
intentional fail. It is not a new adversarial recipe.

## Locked shape

Taken from particle-sliders `locked_shared` / #94 (`analysis/slider2d/locked_baseline_defaults.py`).

| knob | value | note |
| --- | --- | --- |
| pairing | RpGAN logistic, matched | vanilla and shuffled pairs are refused |
| grad penalty | `GradRegularizer` `b_cap`, coeff=1, kappa=1, norm=l2 | real penalty, not a thinned stub |
| FM | `fm_weight=0` | nonzero weight is refused |
| cover | **1.5** | demo / locked_shared cover. Music 1.0 is refused |
| particles | n=12, `particle_l2=0.02` | 128-particle hub cloud is refused |
| critic | frozen linear `D(x)=3 x_0` | this toy has no host critic; weights are not stepped |

`vicreg_weight=0.05` from the Music prior is not applied. These particles are
the orbit state, not a latent prior. That is a reported omission, not a second
adv recipe.

No Music or Anima GPU transfer is claimed from a pass here.

## How to tail

```
python -m particlegan.orbit_hold
pytest -s tests/test_orbit_hold.py
```

Lines are prefixed `ORBIT`.

## Leaderboard

Protocol: 12 particles on the unit circle, 32 steps, `dt=0.25`, `omega=1`.
`gates` are radius, direction, speed. Only `locked_shared` passes.

| arm | radius_rel | direction | speed_rel | d_loss | b_cap | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `locked_shared` | 4.7e-17 | 0.99203 | 8.0e-3 | 0.693147 | 4 | **PASS** |
| `ablate_residual` | **0.410** | 1.00000 | ~0 | 0.693714 | 4 | FAIL `radius_drift` |
| `stranger_vanilla` | ~0 | 0.99203 | 8.0e-3 | 2.295 | 4 | FAIL `stranger_pairing` |
| `stranger_shuffle` | ~0 | 0.99203 | 8.0e-3 | 0.834 | 4 | FAIL `stranger_pairing` |
| `fm_on` | ~0 | 0.99203 | 8.0e-3 | 0.693147 | 4 | FAIL `fm_on` |
| `hinge_drift` | ~0 | 0.99203 | 8.0e-3 | 1.000 | 4 | FAIL `loss_not_logistic` |
| `thin_bcap` | ~0 | 0.99203 | 8.0e-3 | 0.693147 | **9** | FAIL `thinned_bcap` |

`b_cap=4` is the faithful penalty on this critic: `||grad D||=3`, `relu(3-1)^2=4`,
`(coeff/2)*(4+4)=4`. The thinned control hardcodes the cap center to 0, so the
same critic scores `relu(3-0)^2` and the penalty jumps to 9. The orbit still
holds. The fail is the stub.

`fm_on` keeps the residual, so the feature gap is zero and `fm_term` prints as
0. The gate still refuses `fm_weight=1` because locked FM is off.

## Why radius is its own gate

On the ablation, heading is 1, linear speed error is ~0, and matched Rp
logistic only moves from 0.693147 to 0.693714. Final radius is `sqrt(3) ≈ 1.732`
(`particle_l2` term 0.0107). Angular cover stays uniform for both arms, so a
cover score does not see the spiral either. Mean absolute radius error over
the 32 steps is 0.410 against a cap of 0.02.

## Recommendations

- Score this toy on radius. Heading, linear speed, angular cover, and the frozen critic's Rp loss all look healthy on the spiral.
- Keep the radial residual head. Zeroing it is the drift.
- Keep the locked adv block: Rp logistic, real `b_cap` at kappa 1, FM off, demo cover 1.5, 12 particles. Music cover 1.0 and a 128-particle cloud are refusals, not silent aliases.
- Do not read a CPU pass as evidence about Music or Anima.

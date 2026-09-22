# Particle posture toy

One CPU family. Locked shared adv is RpGAN logistic + ParticleGAN
`GradientPenalty` `b_cap` (coeff=1, kappa=1, norm=l2, lazy=1) + FM off.
The critic is one MLP, held fixed. This is not a Music or YuE GPU result.

Cover: **1.5** is the demo lock for the n=12 cloud. **1.0** is Music Arm B
at `--parts 0` only. Those covers are not interchangeable
(`test_music_arm_b_gates.py` rejects cover 1.5 on the Music row).

ParticleGAN `Recipe()` (20_000 particles, `prior_reg=1`, variance target 1)
is a different prior. The tiny arm uses `ParticleRegularizer` weight 0.05
at std target 0.05, matching the slider cloud rather than that default.

Budget: dim=2, batch=32, steps=600, seed=0, lr=0.005.
PASS requires cloud second moment ≤ 0.05 and residual L2 ≤ 0.05
on the tiny cloud, and no cloud at all on parts 0.

| arm | n | particle_l2 | init_std | cover | cloud_ms | anchor_drop | residual_l2 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `locked_tiny` | 12 | 0.02 | 0.05 | 1.5 | 0.002461 | 1.663e-05 | 0.0037 | **PASS** |
| `music_parts0` | 0 | 0.0 | 0.0 | 1.0 | 0.000000 | 0.000e+00 | 0.0116 | **PASS** |
| `hub128_routed` | 128 | 0.0 | 1.0 | 1.5 | 0.998865 | 0.000e+00 | 0.0455 | **FAIL** |
| `zero_particle_l2` | 12 | 0.0 | 0.05 | 1.5 | 0.002105 | 0.000e+00 | 0.0023 | **FAIL** |

## Why

`locked_tiny` is the demo lock (n=12, particle_l2=0.02, cover=1.5). cloud_ms=0.002461, residual_l2=0.0037, anchor_drop=1.663e-05. Gate PASS.
`music_parts0` is Music `--parts 0` (cover=1.0, no cloud, VICReg off). residual_l2=0.0116. Gate PASS. Same adv lock, empty cloud — not a claim that a Music GPU run passed.
`hub128_routed` is the Hub bridge cloud (n=128, init N(0,1), routing=all_examples, particle_l2=0). cloud_ms=0.998865. Gate FAIL. routing 'all_examples' is not none (Hub routes every example)
`zero_particle_l2` keeps n=12 and the 0.05 draw, and drops only particle_l2. cloud_ms=0.002105 can stay small (the 2026-09-09 slider sweep also passed the sheet at l2=0), but anchor_drop=0.000e+00. Gate FAIL: the 0.02 anchor is part of the lock.

## Recommendation

- Keep the living toy cloud at n≤12, init_std 0.05, particle_l2=0.02, demo cover 1.5.
- Music transfer stays `--parts 0`, cover/pole 1.0, vicreg 0. Do not copy n=12 or cover 1.5 onto that row.
- Do not treat Hub 128 routed N(0,1) particles, or `Recipe()`'s 20_000-particle prior, as this lock.
- Do not read a toy PASS as a Music or YuE training result.

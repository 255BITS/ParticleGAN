# Cover / leftover / faithful-teacher gate

CPU toy in this repo (`lib/cover_leftover.py`, `tests/test_cover_leftover.py`).
One shared odd+even residual on a one-row leftover field (û, content, ê, lyric),
trained with this repo's RpGAN logistic, `GradRegularizer` `b_cap` (coeff=1,
kappa=1, norm=l2), a 12-particle cloud, and `particle_l2=0.02`. Feature matching
is off. The critic is a Fourier-2 training instrument; ParticleGAN does not ship
a critic, and the gate scores the residual.

Cover posture is the **demo lock `cover_weight=1.5`** (`LOCKED_COVER` in
particle-sliders). Music's pole analogue is 1.0. This note does not claim
Music or Anima GPU transfer.

Teacher is `faithful_guard_e`: leftover ê is subtracted from the odd part only
while the blend guard still prefers the caption to the midpoint. When ê
restates the axis, the guard keeps the raw poles.

`steps=800` is a budget override of the 1200-step demo lock, not a formulation
change. Logs: `results/cover_leftover/train.log` (line-buffered, `tail -f`).

## Leaderboard (seed 0)

| arm | pass | cover | teacher | u_kept | content | leak | pole_err | same_dir | particle_rms | why |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| locked | PASS | 1.5 | faithful_guard_e | 0.936 | 0.973 | 0.000 | 0.045 | 0.001 | 0.061 | locked demo cover + guard |
| cover_zero | FAIL | 0.0 | faithful_guard_e | 0.577 | 0.593 | 0.060 | 0.321 | 0.030 | 0.210 | undershoot,content |
| teacher_drift | FAIL | 1.5 | faithful | 0.935 | 0.970 | 0.468 | 0.043 | 0.000 | 0.069 | teacher_leak |

Thresholds, from the Field3D leftover cell: `u_kept ≥ 0.85`, `content_kept ≥ 0.75`,
`leak_ratio ≤ 0.20`, pole relative error `≤ 0.20`, `same_dir ≤ 0.25`.

## What the arms show

- **locked** lands on the guarded poles. Leak is ~0 because the teacher already
  dropped unused ê. Particles stay small (`rms 0.06`); the residual carries û
  and content.
- **cover_zero** keeps the guard, so leak stays under 0.20, but the pole pin is
  off. The residual stalls inside the real cloud (`u_kept 0.58`, pole error
  0.32) and the particle cloud is larger (`rms 0.21`). That is the cover=0
  undershoot: guard alone does not cover.
- **teacher_drift** keeps demo cover, so û and content are covered, but the
  teacher is raw `faithful`. Cover copies ê. `leak_ratio 0.47` matches the
  field's leftover amplitude 0.45 on a unit slider.

Stranger pairing, FM-on under `b_cap`, a thinned cap (kappa / coeff / norm),
and a 128-particle cloud are refused rather than trained.

## Recommendation

Keep both knobs. Demo cover 1.5 without `faithful_guard_e` pins the leak.
`faithful_guard_e` without cover undershoots the poles. Do not stretch this
gate toward the 1200-step demo length until cover=0 is checked again: at 1600
steps on the same seed the unpinned arm had crept to `u_kept 0.73` (still
under 0.85) while locked and the leaky teacher both sat on their poles.

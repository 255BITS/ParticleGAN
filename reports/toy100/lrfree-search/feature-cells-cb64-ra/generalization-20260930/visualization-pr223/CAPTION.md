Four observed snapshots from the original rotated100 two-turn experiment (seed 1234).
Dots show 4,096 saved noisy generated samples; open rings show current target centers.
The snapshots are the original float16 observations at updates 0, 500, 1,000 and 1,500.
Target angles are additional rotations from the initial rotated100 geometry.
HQ and mode captions use separate original 20,000-sample gate draws.
No interpolation, extra samples or fresh training was used.

Both runs start from HQ 96.09% at update 500. After the second turn, RA14 scores
85.57% HQ / 99 modes (FAIL); fresh RA15 scores 92.59% HQ / 98 modes (PASS).
The unchanged gate is HQ ≥86.481% and at least 95 modes after both turns.
RA15 fixes this rotated failure; it does not produce uniform HQ gains across
the grid, rotated, and staggered moving tasks. See moving-score-comparison.md.

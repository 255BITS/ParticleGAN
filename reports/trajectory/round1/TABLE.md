# Trajectory comparison

Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 0.0466 | 0.0466 | 99.4% | 21.4% | 6.3% | 0.35 | 100.2 |
| concat | 0.0483 | 0.0390 | 99.8% | 14.8% | 6.0% | 0.33 | 99.3 |
| gaussian | 0.0498 | 0.0615 | 99.7% | 21.3% | 7.1% | 0.34 | 90.8 |
| learned_noise | 0.0506 | 0.0625 | 97.9% | 37.4% | 2.8% | 0.26 | 99.7 |
| one_shot | 0.0520 | 0.0271 | 98.8% | 0.0% | 11.4% | 0.29 | 93.1 |
| fixed_noise | 0.0637 | 0.1198 | 96.4% | 49.7% | 3.1% | 0.45 | 99.6 |

Real-vs-real test floor: SW1 0.0070; route TV 0.0156; validity 100.0%; collision 0.0%.

All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.

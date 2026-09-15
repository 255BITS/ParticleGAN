# Trajectory comparison

Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal_discrete | 0.0506 | 0.0637 | 88.3% | 34.3% | 6.1% | 2.39 | 13.0 |
| mlp_discrete | 0.0531 | 0.0598 | 84.3% | 36.9% | 7.4% | 1.89 | 10.6 |
| temporal_continuous | 0.0552 | 0.0616 | 93.0% | 0.0% | 2.6% | 1.27 | 12.5 |
| mlp_continuous | 0.0602 | 0.0977 | 94.8% | 59.8% | 3.1% | 0.83 | 10.6 |

Real-vs-real test floor: SW1 0.0070; route TV 0.0156; validity 100.0%; collision 0.0%.

All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.

# Trajectory comparison

Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_continuous | 0.0390 | 0.0655 | 99.8% | 80.3% | 1.4% | 0.35 | 99.0 |
| temporal_continuous | 0.0418 | 0.0588 | 99.0% | 53.5% | 0.5% | 0.72 | 119.9 |
| mlp_discrete | 0.0468 | 0.0458 | 99.1% | 1.8% | 10.2% | 0.31 | 100.9 |
| temporal_discrete | 0.0963 | 0.0495 | 18.0% | 0.0% | 36.5% | 1.08 | 124.7 |

Real-vs-real test floor: SW1 0.0070; route TV 0.0156; validity 100.0%; collision 0.0%.

All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.

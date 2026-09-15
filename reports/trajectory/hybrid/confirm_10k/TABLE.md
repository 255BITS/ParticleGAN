# Trajectory comparison

Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_continuous | 0.0320 | 0.0483 | 99.8% | 91.9% | 2.0% | 0.36 | 126.4 |
| mlp_continuous | 0.0385 | 0.0667 | 99.6% | 76.7% | 0.2% | 0.34 | 98.6 |

Real-vs-real test floor: SW1 0.0070; route TV 0.0156; validity 100.0%; collision 0.0%.

All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.

# Trajectory comparison

Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.

| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_continuous | 0.0486 | 0.0652 | 94.7% | 54.5% | 3.1% | 0.97 | 10.3 |
| hybrid_continuous | 0.0547 | 0.0816 | 96.0% | 53.5% | 2.6% | 1.00 | 12.9 |

Real-vs-real test floor: SW1 0.0070; route TV 0.0156; validity 100.0%; collision 0.0%.

All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.

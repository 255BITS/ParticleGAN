# Support diagnostic leaderboard

Each entry is recall / all false positives / rare false positives.
Every entry uses its original held-out calibration, Q=.05 and unchanged family gate.
None of these support laws qualifies across the required family.

| Fixture | Original | Prediction only | Fisher | Student | Full covariance NIW | Actual anchor posterior |
|---|---:|---:|---:|---:|---:|---:|
| geometry/fold N1024 trained | 0.000/0/0 | 0.000/0/0 | 0.000/0/0 | 0.000/0/0 | 0.000/0/0 | 1.000/0/0 |
| geometry/fold N2048 trained | 1.000/3/1 | 1.000/2/0 | 1.000/1/1 | 1.000/1/0 | 1.000/4/0 | 1.000/2/1 |
| geometry/fold N4096 trained | 1.000/4/0 | 1.000/5/0 | 1.000/7/0 | 1.000/8/0 | 1.000/1/0 | 1.000/4/0 |
| geometry/fold N1024 frozen | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 |
| geometry/fold N2048 frozen | 1.000/4/0 | 1.000/5/1 | 1.000/1/1 | 1.000/6/1 | 1.000/3/0 | 1.000/2/0 |
| geometry/fold N4096 frozen | 1.000/11/0 | 1.000/13/0 | 1.000/1/0 | 1.000/14/0 | 1.000/6/1 | 1.000/11/0 |
| cost/nominal N1024  | 1.000/0/0 | 1.000/0/0 | — | — | — | 0.000/0/0 |
| cost/nominal N2048  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |
| cost/nominal N4096  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |
| cost/nominal N8192  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |
| cost/highdim N1024  | 0.000/0/0 | 0.000/0/0 | 1.000/0/0 | 1.000/0/0 | 0.000/0/0 | 0.000/0/0 |
| cost/highdim N2048  | 0.000/0/0 | 0.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 |
| cost/highdim N4096  | 0.000/0/0 | 0.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 |
| cost/highdim N8192  | 0.000/0/0 | 0.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 | 1.000/0/0 |
| cost/rare_hole N1024  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |
| cost/rare_hole N2048  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/2/2 |
| cost/rare_hole N4096  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |
| cost/rare_hole N8192  | 1.000/0/0 | 1.000/0/0 | — | — | — | 1.000/0/0 |

Cost gates report rare FP but do not impose the geometry zero-rare-FP condition.
The actual-anchor posterior flags two legitimate rare cost rows at N2048/rare_hole, despite passing its cost detector gate.
Support-only passes do not establish parent, mass, replay or learned-model quality.

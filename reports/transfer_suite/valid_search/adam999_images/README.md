# Solvability search

Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. These are inspected development tasks. Different training budgets and capacities are explicit. A per-task witness establishes solvability, not a shared default.

| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| adam999_transpose12 | img_stripes2 | FAIL | None | 600 | 10.43 | modes=1 |
| adam999_transpose12 | img_bars4 | FAIL | None | 600 | 8.69 | modes=3 |
| adam999_transpose12 | img_blobs4 | PASS | 475 | 600 | 7.98 | — |
| adam999_transpose12 | img_intensity2 | FAIL | None | 600 | 7.35 | modes=0, hq=0 |
| adam999_residual12 | img_stripes2 | PASS | 300 | 600 | 8.06 | — |
| adam999_residual12 | img_bars4 | FAIL | None | 600 | 9.27 | modes=3 |
| adam999_residual12 | img_blobs4 | FAIL | None | 600 | 8.00 | modes=3 |
| adam999_residual12 | img_intensity2 | PASS | 550 | 600 | 7.45 | — |
| adam999_residual24 | img_stripes2 | PASS | 200 | 600 | 9.62 | — |
| adam999_residual24 | img_bars4 | PASS | 475 | 600 | 9.70 | — |
| adam999_residual24 | img_blobs4 | FAIL | None | 600 | 9.76 | modes=2 |
| adam999_residual24 | img_intensity2 | PASS | 525 | 600 | 9.36 | — |

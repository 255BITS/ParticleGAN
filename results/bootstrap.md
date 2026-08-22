# Bootstrap comparison vs. the best baseline coefficient

Statistic: median final `w1_exact` of each main-stage (arm, coeff) minus the median final `w1_exact` of the baseline `a_r1r2` @ coeff `1.0` (the a_r1r2 coefficient with the lowest mean final exact W1).

CI95 from 10,000 bootstrap resamples of the seeds *within each group* (numpy default_rng, seed 20250821). Negative differences favour the arm (lower W1 is better).

**Verdict rule.** An arm *wins* only if its CI95 excludes 0 in its favour AND its collapse rate is <= the baseline's (0.00). It *matches* if the CI includes 0 and its collapse rate is no worse. Otherwise it loses, or is flagged as buying W1 at the cost of stability.

| arm | coeff | n | median W1 | Δ median W1 | CI95 | collapse rate | verdict |
|---|---|---|---|---|---|---|---|
| a_r1r2 | 0.005 | 5 | 0.8947 | 0.7209 | [0.4336, 3.1500] | 0.00 | loses |
| a_r1r2 | 0.02 | 5 | 0.5638 | 0.3899 | [0.2767, 0.6716] | 0.00 | loses |
| a_r1r2 | 0.1 | 5 | 0.3647 | 0.1909 | [0.0787, 0.2105] | 0.00 | loses |
| a_r1r2 | 0.3 | 5 | 0.2068 | 0.0329 | [-0.0199, 0.1331] | 0.00 | matches |
| a_r1r2 (baseline) | 1.0 | 5 | 0.1739 | 0.0000 | [-0.0528, 0.0528] | 0.00 | matches |
| b_cap | 0.005 | 5 | 1.4953 | 1.3215 | [0.5827, 3.1271] | 0.00 | loses |
| b_cap | 0.02 | 5 | 0.5653 | 0.3914 | [0.2979, 0.5239] | 0.00 | loses |
| b_cap | 0.1 | 5 | 0.4269 | 0.2530 | [0.1546, 0.2620] | 0.00 | loses |
| b_cap | 1.0 | 5 | 0.3673 | 0.1935 | [0.1407, 0.2318] | 0.00 | loses |
| c_eikonal | 0.005 | 5 | 1.1445 | 0.9706 | [0.7163, 2.0861] | 0.00 | loses |
| c_eikonal | 0.02 | 5 | 0.7467 | 0.5728 | [0.3699, 0.8266] | 0.00 | loses |
| c_eikonal | 0.1 | 5 | 0.4473 | 0.2735 | [0.1854, 0.3672] | 0.00 | loses |
| c_eikonal | 1.0 | 5 | 0.4715 | 0.2976 | [0.1368, 0.3624] | 0.00 | loses |
| d_asym | 0.005 | 5 | 1.3281 | 1.1542 | [0.5962, 3.1209] | 0.00 | loses |
| d_asym | 0.02 | 5 | 0.6259 | 0.4521 | [0.2699, 3.2070] | 0.00 | loses |
| d_asym | 0.1 | 5 | 0.4564 | 0.2826 | [0.1756, 0.4700] | 0.00 | loses |
| d_asym | 1.0 | 5 | 0.4155 | 0.2416 | [0.1888, 0.2874] | 0.00 | loses |
| e_interp | 0.005 | 5 | 0.6146 | 0.4408 | [0.3736, 0.5100] | 0.00 | loses |
| e_interp | 0.02 | 5 | 0.5072 | 0.3334 | [0.2398, 0.4440] | 0.00 | loses |
| e_interp | 0.1 | 5 | 0.4707 | 0.2969 | [0.2338, 0.3621] | 0.00 | loses |
| e_interp | 1.0 | 5 | 0.4547 | 0.2808 | [0.1657, 0.3672] | 0.00 | loses |
| f_none | 0.0 | 5 | 1.9804 | 1.8066 | [1.0060, 2.7622] | 0.00 | loses |

Note: the baseline row compares the baseline group against itself; its point estimate is 0 by construction and its CI reflects only seed noise, which is a useful scale reference for the other rows.


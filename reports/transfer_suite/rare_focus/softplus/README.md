# Rare-mode discriminator refinement, round 2

Ten architecture cards were frozen before this adaptive follow-up to the D96×2 / Fourier2 / Softplus β5 near-miss. Every failed architecture and all24 live/EMA observations are retained. All sustained rare-case winners receive the other five valid data tests without changes. These are inspected development cases, not held-out confirmation.

Original shared recipe: Rp logistic, b_cap coefficient3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. Only D Softplus β, width, and axis-Fourier resolution change; D depth2 is fixed. Every rare run uses1200 updates; any cross-check retains1200 except spiral1600. Seed0, one CPU thread; no seed sweeps.

PASS means unchanged live metric bounds, all24 expected observations, and at least five final passing observations. EMA never determines success. Missing cross-checks are unrun.

| Discriminator | D params | Rare sustained | Final passing suffix | Passing observations /24 | Final mean normalized shortfall | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| softplus6_d96_l2_f2 | 10465 | [FAIL](episodes/softplus6_d96_l2_f2__vector_unequal_mass.json.gz) | 1 | 4/24 | 0 | 6.72 |
| softplus5_d80_l2_f2 | 7441 | [FAIL](episodes/softplus5_d80_l2_f2__vector_unequal_mass.json.gz) | 0 | 1/24 | 0.0391024 | 6.33 |
| softplus4_d96_l2_f2 | 10465 | [FAIL](episodes/softplus4_d96_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.13545 | 6.61 |
| softplus4_d112_l2_f2 | 14001 | [FAIL](episodes/softplus4_d112_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.152637 | 6.99 |
| softplus5_d112_l2_f2 | 14001 | [FAIL](episodes/softplus5_d112_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.162658 | 7.20 |
| softplus8_d96_l2_f2 | 10465 | [FAIL](episodes/softplus8_d96_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.166667 | 6.70 |
| softplus2_d96_l2_f2 | 10465 | [FAIL](episodes/softplus2_d96_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.273264 | 7.59 |
| softplus6_d80_l2_f2 | 7441 | [FAIL](episodes/softplus6_d80_l2_f2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.328602 | 6.13 |
| softplus5_d96_l2_f3 | 10849 | [FAIL](episodes/softplus5_d96_l2_f3__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.333333 | 6.58 |
| softplus5_d96_l2_f1 | 10081 | [FAIL](episodes/softplus5_d96_l2_f1__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.491325 | 6.49 |

Rare sustained winners: **none**.

## Complete late curves for the three closest rare results

The five columns are steps1000,1050,1100,1150,1200. A good final value alone cannot pass. [All24 observations for every architecture](rare_curves.json.gz) include live and EMA values, measurement times, and recomputed per-check success.

| Discriminator | Metric / unchanged bound | 1000 | 1050 | 1100 | 1150 | 1200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| softplus6_d96_l2_f2 | sw1_normalized <= 0.18 | 0.0586813 | 0.0550347 | 0.0679851 | 0.0514019 | 0.0571733 |
| softplus6_d96_l2_f2 | mass_tv <= 0.15 | 0.0150391 | 0.0150391 | 0.0150391 | 0.0150391 | 0.0150391 |
| softplus6_d96_l2_f2 | hq >= 0.85 | 0.988281 | 0.989502 | 0.982422 | 0.993164 | 0.993164 |
| softplus6_d96_l2_f2 | component_covariance_error <= 0.85 | 0.577837 | 0.642904 | 0.661723 | 0.563801 | 0.557783 |
| softplus6_d96_l2_f2 | component_min_eigen_ratio >= 0.15 | 0.10595 | 0.132615 | 0.202776 | 0.135027 | 0.208003 |
| softplus6_d96_l2_f2 | min_mass_ratio >= 0.25 | 0.976119 | 0.976119 | 0.976119 | 0.976119 | 0.976119 |
| softplus6_d96_l2_f2 | ALL live bounds | FAIL | FAIL | PASS | FAIL | PASS |
| softplus5_d80_l2_f2 | sw1_normalized <= 0.18 | 0.078861 | 0.0787251 | 0.0727714 | 0.0830024 | 0.0698735 |
| softplus5_d80_l2_f2 | mass_tv <= 0.15 | 0.0612109 | 0.0612109 | 0.0612109 | 0.0612109 | 0.0612109 |
| softplus5_d80_l2_f2 | hq >= 0.85 | 0.982666 | 0.982666 | 0.987305 | 0.990479 | 0.987305 |
| softplus5_d80_l2_f2 | component_covariance_error <= 0.85 | 0.456987 | 0.417343 | 0.407416 | 0.398893 | 0.422698 |
| softplus5_d80_l2_f2 | component_min_eigen_ratio >= 0.15 | 0.12564 | 0.0190146 | 0.0239955 | 0.0903858 | 0.114808 |
| softplus5_d80_l2_f2 | min_mass_ratio >= 0.25 | 0.769043 | 0.769043 | 0.769043 | 0.769043 | 0.769043 |
| softplus5_d80_l2_f2 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |
| softplus4_d96_l2_f2 | sw1_normalized <= 0.18 | 0.078911 | 0.0723679 | 0.0902156 | 0.0809995 | 0.0811982 |
| softplus4_d96_l2_f2 | mass_tv <= 0.15 | 0.0568359 | 0.0568359 | 0.0568359 | 0.0568359 | 0.0568359 |
| softplus4_d96_l2_f2 | hq >= 0.85 | 0.988037 | 0.979736 | 0.971924 | 0.979004 | 0.977539 |
| softplus4_d96_l2_f2 | component_covariance_error <= 0.85 | 0.568185 | 0.622973 | 0.578166 | 0.579089 | 0.516897 |
| softplus4_d96_l2_f2 | component_min_eigen_ratio >= 0.15 | 0.0116859 | 0.000233321 | 0.00301053 | 0.00967237 | 0.0280954 |
| softplus4_d96_l2_f2 | min_mass_ratio >= 0.25 | 0.896662 | 0.896662 | 0.896662 | 0.896662 | 0.896662 |
| softplus4_d96_l2_f2 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |

All numerical sources, episode bytes, original targets/gates and complete observation schedules were audited. Source implementation is unchanged `SmoothFourierCritic`; static initialization/output/input-gradient/cap-penalty/parameter-gradient equality was checked against the earlier implementation. Architecture support across cases must be labeled separately from a single shared discriminator.

[Frozen plan](plan.json.gz) · [Exact task specs](task_specs.json.gz) · [Resolved records, metrics, EMA and hashes](index.json.gz) · [Audit](audit.json.gz) · [Static checks](architecture_checks.json.gz) · [Runtime/source hashes](protocol.json.gz) · [Exact source](source.tar.gz) · [Driver](run.py) · [Execution log](run.log).

Timings are observed CPU episode wall times including measurements and are not a controlled speed comparison. No required regression, image, stress or diagnostic cases were used to select these architectures.

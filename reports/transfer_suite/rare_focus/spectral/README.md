# Rare-mode spectral-balance refinement, round 3

Ten fixed feature-amplitude cards were frozen before this adaptive follow-up to the smooth-D near-misses. Forensics found native critic gradients contracted the rare minor axis despite a full-rank generator; gain changes are a hypothesis about the resulting spectral bias, not a demonstrated causal remedy. Every failed architecture and all24 live/EMA observations are retained. All sustained rare-case winners receive the other five valid data tests without changes. These are inspected development cases, not held-out confirmation.

Original shared recipe: Rp logistic, b_cap coefficient3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. Only fixed discriminator raw-coordinate/Fourier/harmonic amplitudes change around Softplus β5/6; width96, depth2, and axis-Fourier2 stay fixed. Gains are generic constants, independent of data, labels, target geometry, or component identities. Every rare run uses1200 updates; any cross-check retains1200 except spiral1600. Seed0, one CPU thread; no seed sweeps.

PASS means unchanged live metric bounds, all24 expected observations, and at least five final passing observations. EMA never determines success. Missing cross-checks are unrun.

Names encode Softplus beta (`b5` or `b6`), raw-coordinate gain, common Fourier gain, or second-harmonic gain. Unnamed gains equal1. All gains are nonzero, so these are different parameterizations of the same MLP function class; they add no target-aware features or representational capacity.

| Discriminator | D params | Rare sustained | Final passing suffix | Passing observations /24 | Final mean normalized shortfall | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| b5_fourier05 | 10465 | [FAIL](episodes/b5_fourier05__vector_unequal_mass.json.gz) | 0 | 1/24 | 0.0250353 | 6.79 |
| b5_raw4 | 10465 | [FAIL](episodes/b5_raw4__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.117932 | 6.88 |
| b6_fourier05 | 10465 | [FAIL](episodes/b6_fourier05__vector_unequal_mass.json.gz) | 0 | 2/24 | 0.126738 | 6.71 |
| b5_harmonic2_025 | 10465 | [FAIL](episodes/b5_harmonic2_025__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.142944 | 6.76 |
| b6_harmonic2_025 | 10465 | [FAIL](episodes/b6_harmonic2_025__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.148324 | 6.75 |
| b5_fourier025 | 10465 | [FAIL](episodes/b5_fourier025__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.151396 | 6.74 |
| b5_harmonic2_05 | 10465 | [FAIL](episodes/b5_harmonic2_05__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.157069 | 6.78 |
| b5_raw2_fourier05 | 10465 | [FAIL](episodes/b5_raw2_fourier05__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.159443 | 6.87 |
| b5_raw2 | 10465 | [FAIL](episodes/b5_raw2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.178711 | 7.95 |
| b6_raw2 | 10465 | [FAIL](episodes/b6_raw2__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.178711 | 6.78 |

Rare sustained winners: **none**.

## Complete late curves for the three closest rare results

The five columns are steps1000,1050,1100,1150,1200. A good final value alone cannot pass. [All24 observations for every architecture](rare_curves.json.gz) include live and EMA values, measurement times, and recomputed per-check success.

| Discriminator | Metric / unchanged bound | 1000 | 1050 | 1100 | 1150 | 1200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| b5_fourier05 | sw1_normalized <= 0.18 | 0.116116 | 0.109258 | 0.106721 | 0.110464 | 0.105292 |
| b5_fourier05 | mass_tv <= 0.15 | 0.0631152 | 0.0631152 | 0.0631152 | 0.0631152 | 0.0631152 |
| b5_fourier05 | hq >= 0.85 | 0.994385 | 0.994385 | 0.992188 | 0.980713 | 0.976318 |
| b5_fourier05 | component_covariance_error <= 0.85 | 0.408844 | 0.343828 | 0.422305 | 0.412049 | 0.382621 |
| b5_fourier05 | component_min_eigen_ratio >= 0.15 | 0.00309472 | 0.0065411 | 0.0144634 | 0.0785038 | 0.127468 |
| b5_fourier05 | min_mass_ratio >= 0.25 | 0.837402 | 0.837402 | 0.837402 | 0.837402 | 0.837402 |
| b5_fourier05 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |
| b5_raw4 | sw1_normalized <= 0.18 | 0.150935 | 0.159015 | 0.152965 | 0.160091 | 0.145901 |
| b5_raw4 | mass_tv <= 0.15 | 0.123584 | 0.123584 | 0.123584 | 0.123584 | 0.123584 |
| b5_raw4 | hq >= 0.85 | 0.972412 | 0.969727 | 0.962891 | 0.953369 | 0.958496 |
| b5_raw4 | component_covariance_error <= 0.85 | 3.16218 | 2.9254 | 2.53996 | 1.5553 | 1.45145 |
| b5_raw4 | component_min_eigen_ratio >= 0.15 | 0.0165703 | 0.0901491 | 0.0751774 | 0.232083 | 0.324313 |
| b5_raw4 | min_mass_ratio >= 0.25 | 0.614421 | 0.614421 | 0.614421 | 0.614421 | 0.614421 |
| b5_raw4 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |
| b6_fourier05 | sw1_normalized <= 0.18 | 0.0673639 | 0.0692359 | 0.0760425 | 0.0528096 | 0.078899 |
| b6_fourier05 | mass_tv <= 0.15 | 0.0318848 | 0.0318848 | 0.0318848 | 0.0318848 | 0.0318848 |
| b6_fourier05 | hq >= 0.85 | 0.994873 | 0.992188 | 1 | 0.980225 | 1 |
| b6_fourier05 | component_covariance_error <= 0.85 | 0.506331 | 0.494672 | 0.490449 | 0.441814 | 0.479452 |
| b6_fourier05 | component_min_eigen_ratio >= 0.15 | 0.135389 | 0.0581103 | 0.0394062 | 0.0368104 | 0.0359357 |
| b6_fourier05 | min_mass_ratio >= 0.25 | 0.945046 | 0.945046 | 0.945046 | 0.945046 | 0.945046 |
| b6_fourier05 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |

All numerical sources, episode bytes, original targets/gates and complete observation schedules were audited. The research `SpectralBalancedCritic` subclasses the exact existing smooth critic and changes only its native differentiable encoding amplitudes. Neutral gains exactly match baseline states, outputs, input/parameter gradients, and cap penalties. All ten cards pass finite input-Hessian and active cap double-backprop checks. Only those static checks multiply output weights100× to activate the cap; training uses normal initialization and the unchanged cap. Cap differentiation remains in original data coordinates. Architecture support across cases must be labeled separately from a single shared discriminator.

[Frozen plan](plan.json.gz) · [Exact task specs](task_specs.json.gz) · [Resolved records, metrics, EMA and hashes](index.json.gz) · [Audit](audit.json.gz) · [Static checks](architecture_checks.json.gz) · [Runtime/source hashes](protocol.json.gz) · [Exact source](source.tar.gz) · [Driver](run.py) · [Execution log](run.log).

Timings are observed CPU episode wall times including measurements and are not a controlled speed comparison. No required regression, image, stress or diagnostic cases were used to select these architectures.

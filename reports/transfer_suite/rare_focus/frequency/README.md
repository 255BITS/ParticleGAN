# Rare-mode frequency refinement, final round 4

Eight frequency-basis cards were frozen before this final adaptive architecture round. Frozen-critic forensics identified dominant contraction from the lowest pi harmonic. This follow-up tests lower generic frequencies and raw-only controls. It is a hypothesis about trainable feature bases, not a demonstrated causal remedy. Every failed architecture and all24 live/EMA observations are retained. All sustained rare-case winners receive the other five valid data tests without changes. These are inspected development cases, not held-out confirmation.

Original shared recipe: Rp logistic, b_cap coefficient3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. Only discriminator Fourier frequencies change around Softplus β5/6; width96 and depth2 stay fixed. Two axis bands use frequencies [pi,2pi] multiplied by .125/.25/.5; raw-only controls use zero Fourier bands. Frequency choices are generic constants, independent of data, labels, target geometry, or component identities. Feature amplitudes are unchanged. Every rare run uses1200 updates; any cross-check retains1200 except spiral1600. Seed0, one CPU thread; no seed sweeps.

PASS means unchanged live metric bounds, all24 expected observations, and at least five final passing observations. EMA never determines success. Missing cross-checks are unrun.

Names encode Softplus beta (`b5` or `b6`) and the global frequency multiplier; raw-only cards omit Fourier features. Prior architecture inventories and sibling confirmation excluded exact raw-only D96×2 Softplus5/6 duplicates before execution.

| Discriminator | D params | Rare sustained | Final passing suffix | Passing observations /24 | Final mean normalized shortfall | Seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| b5_raw_only | 9697 | [FAIL](episodes/b5_raw_only__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.133696 | 6.59 |
| b6_frequency0p125 | 10465 | [FAIL](episodes/b6_frequency0p125__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.147109 | 6.66 |
| b5_frequency0p125 | 10465 | [FAIL](episodes/b5_frequency0p125__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.158621 | 7.79 |
| b6_raw_only | 9697 | [FAIL](episodes/b6_raw_only__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.160668 | 6.71 |
| b5_frequency0p25 | 10465 | [FAIL](episodes/b5_frequency0p25__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.447264 | 6.63 |
| b6_frequency0p25 | 10465 | [FAIL](episodes/b6_frequency0p25__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.685366 | 6.73 |
| b5_frequency0p5 | 10465 | [FAIL](episodes/b5_frequency0p5__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.970851 | 6.56 |
| b6_frequency0p5 | 10465 | [FAIL](episodes/b6_frequency0p5__vector_unequal_mass.json.gz) | 0 | 0/24 | 0.971983 | 6.71 |

Rare sustained winners: **none**.

## Complete late curves for the three closest rare results

The five columns are steps1000,1050,1100,1150,1200. A good final value alone cannot pass. [All24 observations for every architecture](rare_curves.json.gz) include live and EMA values, measurement times, and recomputed per-check success.

| Discriminator | Metric / unchanged bound | 1000 | 1050 | 1100 | 1150 | 1200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| b5_raw_only | sw1_normalized <= 0.18 | 0.0660776 | 0.0672733 | 0.0704606 | 0.0626334 | 0.0621744 |
| b5_raw_only | mass_tv <= 0.15 | 0.0097168 | 0.0097168 | 0.0097168 | 0.0097168 | 0.0097168 |
| b5_raw_only | hq >= 0.85 | 1 | 1 | 0.980713 | 1 | 0.994873 |
| b5_raw_only | component_covariance_error <= 0.85 | 0.765243 | 0.790115 | 0.82586 | 0.84265 | 0.840732 |
| b5_raw_only | component_min_eigen_ratio >= 0.15 | 0.015038 | 0.0155863 | 0.0266837 | 0.0160378 | 0.0296732 |
| b5_raw_only | min_mass_ratio >= 0.25 | 0.967611 | 0.967611 | 0.967611 | 0.967611 | 0.967611 |
| b5_raw_only | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |
| b6_frequency0p125 | sw1_normalized <= 0.18 | 0.0823944 | 0.0642524 | 0.0603421 | 0.0652452 | 0.0762897 |
| b6_frequency0p125 | mass_tv <= 0.15 | 0.0118945 | 0.0118945 | 0.0118945 | 0.0118945 | 0.0118945 |
| b6_frequency0p125 | hq >= 0.85 | 0.942383 | 1 | 1 | 0.996582 | 0.992676 |
| b6_frequency0p125 | component_covariance_error <= 0.85 | 1.12334 | 0.749214 | 0.754951 | 0.73628 | 0.728805 |
| b6_frequency0p125 | component_min_eigen_ratio >= 0.15 | 0.0934192 | 0.0552548 | 0.00990941 | 0.0109358 | 0.0176015 |
| b6_frequency0p125 | min_mass_ratio >= 0.25 | 0.90332 | 0.90332 | 0.90332 | 0.90332 | 0.90332 |
| b6_frequency0p125 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |
| b5_frequency0p125 | sw1_normalized <= 0.18 | 0.0622823 | 0.0577064 | 0.0652084 | 0.0656486 | 0.0590642 |
| b5_frequency0p125 | mass_tv <= 0.15 | 0.0179394 | 0.0179394 | 0.0179394 | 0.0179394 | 0.0179394 |
| b5_frequency0p125 | hq >= 0.85 | 0.984375 | 0.987061 | 0.987305 | 0.993652 | 0.993652 |
| b5_frequency0p125 | component_covariance_error <= 0.85 | 0.66948 | 0.719733 | 0.732768 | 0.748148 | 0.736882 |
| b5_frequency0p125 | component_min_eigen_ratio >= 0.15 | 0.0124107 | 0.0250591 | 0.013603 | 0.00369334 | 0.00724133 |
| b5_frequency0p125 | min_mass_ratio >= 0.25 | 0.862004 | 0.862004 | 0.862004 | 0.862004 | 0.862004 |
| b5_frequency0p125 | ALL live bounds | FAIL | FAIL | FAIL | FAIL | FAIL |

All numerical sources, episode bytes, original targets/gates and complete observation schedules were audited. The research `FrequencyScaledCritic` subclasses the exact existing smooth critic and changes only its fixed Fourier frequency buffer. Neutral multiplier1 exactly matches baseline states, outputs, input/parameter gradients, and cap penalties for Fourier2 and raw/Fourier0. All eight cards pass finite input-Hessian and active cap double-backprop checks. Only those static checks multiply output weights1000× to activate the cap; training uses normal initialization and the unchanged cap. Cap differentiation remains in original data coordinates. Architecture support across cases must be labeled separately from a single shared discriminator.

[Frozen plan](plan.json.gz) · [Exact task specs](task_specs.json.gz) · [Resolved records, metrics, EMA and hashes](index.json.gz) · [Audit](audit.json.gz) · [Static checks](architecture_checks.json.gz) · [Runtime/source hashes](protocol.json.gz) · [Exact source](source.tar.gz) · [Driver](run.py) · [Execution log](run.log).

Timings are observed CPU episode wall times including measurements and are not a controlled speed comparison. No required regression, image, stress or diagnostic cases were used to select these architectures.

# Formulation × architecture results

One entry per formulation. PASS means at least one listed architecture sustains every live metric; each case counts once. Architecture failures remain visible. Architecture cells cannot mix numerical formulations, optimizer settings, data or training budgets.

Current PR scope: nine required regressions, six data toys and four image toys. The scope revision changed 7/16 to 7/10. Subsequent discriminator-only results improve b_cap3 to 9/10 with the same training recipe. [Longer training](LONG_TRAINING.md) is a separate toy; [imposed-setting diagnostics](DIAGNOSTICS.md) do not affect this comparison.

## rp_logistic_bcap3

Required: 9/9. Practical: 9/10. Eligible on required tests: True.

| Problem / condition | Supported | Passing / tested architectures | Earliest confirmed architecture |
| --- | --- | ---: | --- |
| vector_two_broad | PASS | 8/8 | softplus5_d128_l3_f2 at 250 |
| vector_unequal_mass | FAIL | 0/27 | — |
| vector_unequal_width | PASS | 2/27 | softplus10_d64_l2_f2 at 1050 |
| vector_anisotropic | PASS | 3/8 | original architecture at 700 |
| vector_overlap | PASS | 6/27 | d128_l3_f4 at 900 |
| vector_spiral | PASS | 8/8 | d128_l3_f3 at 334 |
| img_stripes2 | PASS | 2/5 | residual16 at 225 |
| img_bars4 | PASS | 1/5 | residual16 at 550 |
| img_blobs4 | PASS | 2/5 | baseline at 525 |
| img_intensity2 | PASS | 2/5 | residual12 at 550 |

<details><summary>Every architecture trial for rp_logistic_bcap3, including failures</summary>

### vector_two_broad

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | PASS | 19 | 500 | — |
| d128_l3_f3 | PASS | 21 | 400 | — |
| d128_l3_f4 | PASS | 22 | 350 | — |
| d256_l3_f3 | PASS | 23 | 300 | — |
| axis_softplus5 | PASS | 17 | 600 | — |
| axis_tanh | PASS | 17 | 600 | — |
| softplus10_d64_l2_f2 | PASS | 11 | 900 | — |
| softplus5_d128_l3_f2 | PASS | 24 | 250 | — |

### vector_unequal_mass

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | FAIL | 0 | — | component_covariance_error |
| d128_l2_f2 | FAIL | 0 | — | component_min_eigen_ratio |
| d128_l2_f3 | FAIL | 0 | — | component_covariance_error |
| d128_l2_f4 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| d128_l3_f2 | FAIL | 0 | — | component_min_eigen_ratio |
| d128_l3_f3 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| d128_l3_f4 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| d256_l2_f2 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| d256_l3_f3 | FAIL | 0 | — | hq, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| d64_l2_f3 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| d64_l2_f5 | FAIL | 0 | — | hq, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| d64_l3_f2 | FAIL | 0 | — | component_covariance_error |
| d64_l4_f3 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| axis_silu | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| axis_softplus1 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| axis_softplus5 | FAIL | 0 | — | sw1_normalized, mass_tv |
| axis_tanh | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| oriented4_silu | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| oriented8_silu | FAIL | 0 | — | component_min_eigen_ratio, min_mass_ratio |
| oriented8_softplus1 | FAIL | 0 | — | component_min_eigen_ratio |
| oriented8_tanh | FAIL | 0 | — | sw1_normalized, mass_tv, component_covariance_error |
| softplus10_d64_l2_f2 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| softplus3_d64_l2_f2 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| softplus5_d128_l2_f2 | FAIL | 0 | — | component_min_eigen_ratio |
| softplus5_d128_l3_f2 | FAIL | 0 | — | component_min_eigen_ratio |
| softplus5_d64_l3_f2 | FAIL | 0 | — | component_min_eigen_ratio |
| softplus5_d96_l2_f2 | FAIL | 0 | — | component_min_eigen_ratio |

### vector_unequal_width

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | FAIL | 0 | — | component_covariance_error |
| d128_l2_f2 | FAIL | 0 | — | component_covariance_error |
| d128_l2_f3 | FAIL | 0 | — | component_covariance_error |
| d128_l2_f4 | FAIL | 0 | — | sw1_normalized, hq, component_covariance_error |
| d128_l3_f2 | FAIL | 2 | — | Final bounds pass; insufficient final passing checks |
| d128_l3_f3 | FAIL | 0 | — | component_covariance_error |
| d128_l3_f4 | FAIL | 0 | — | component_covariance_error |
| d256_l2_f2 | FAIL | 0 | — | component_covariance_error |
| d256_l3_f3 | FAIL | 0 | — | component_covariance_error |
| d64_l2_f3 | FAIL | 0 | — | component_covariance_error |
| d64_l2_f5 | FAIL | 0 | — | hq, component_covariance_error |
| d64_l3_f2 | FAIL | 0 | — | mass_tv, component_covariance_error |
| d64_l4_f3 | FAIL | 0 | — | component_covariance_error |
| axis_silu | FAIL | 0 | — | component_covariance_error |
| axis_softplus1 | FAIL | 0 | — | hq, component_covariance_error |
| axis_softplus5 | PASS | 7 | 1100 | — |
| axis_tanh | FAIL | 0 | — | component_covariance_error |
| oriented4_silu | FAIL | 0 | — | component_covariance_error |
| oriented8_silu | FAIL | 0 | — | sw1_normalized, mass_tv, component_min_eigen_ratio |
| oriented8_softplus1 | FAIL | 0 | — | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio |
| oriented8_tanh | FAIL | 0 | — | sw1_normalized, mass_tv, component_covariance_error |
| softplus10_d64_l2_f2 | PASS | 8 | 1050 | — |
| softplus3_d64_l2_f2 | FAIL | 0 | — | component_covariance_error |
| softplus5_d128_l2_f2 | FAIL | 0 | — | component_covariance_error |
| softplus5_d128_l3_f2 | FAIL | 0 | — | component_covariance_error, component_min_eigen_ratio |
| softplus5_d64_l3_f2 | FAIL | 0 | — | component_covariance_error |
| softplus5_d96_l2_f2 | FAIL | 0 | — | component_covariance_error |

### vector_anisotropic

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | PASS | 15 | 700 | — |
| d128_l3_f3 | PASS | 13 | 800 | — |
| d128_l3_f4 | PASS | 13 | 800 | — |
| d256_l3_f3 | FAIL | 0 | — | component_covariance_error |
| axis_softplus5 | FAIL | 0 | — | sw1_normalized, mass_tv, component_covariance_error |
| axis_tanh | FAIL | 0 | — | sw1_normalized, mass_tv, component_covariance_error |
| softplus10_d64_l2_f2 | FAIL | 0 | — | sw1_normalized, mass_tv |
| softplus5_d128_l3_f2 | FAIL | 0 | — | component_covariance_error |

### vector_overlap

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | FAIL | 4 | — | Final bounds pass; insufficient final passing checks |
| d128_l2_f2 | FAIL | 2 | — | Final bounds pass; insufficient final passing checks |
| d128_l2_f3 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| d128_l2_f4 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| d128_l3_f2 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| d128_l3_f3 | PASS | 5 | 1200 | — |
| d128_l3_f4 | PASS | 11 | 900 | — |
| d256_l2_f2 | FAIL | 2 | — | Final bounds pass; insufficient final passing checks |
| d256_l3_f3 | PASS | 7 | 1100 | — |
| d64_l2_f3 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| d64_l2_f5 | PASS | 10 | 950 | — |
| d64_l3_f2 | FAIL | 3 | — | Final bounds pass; insufficient final passing checks |
| d64_l4_f3 | FAIL | 4 | — | Final bounds pass; insufficient final passing checks |
| axis_silu | FAIL | 4 | — | Final bounds pass; insufficient final passing checks |
| axis_softplus1 | FAIL | 0 | — | sw1_normalized, mean_error |
| axis_softplus5 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| axis_tanh | PASS | 6 | 1150 | — |
| oriented4_silu | FAIL | 0 | — | mean_error |
| oriented8_silu | FAIL | 0 | — | mean_error |
| oriented8_softplus1 | FAIL | 0 | — | sw1_normalized, mean_error |
| oriented8_tanh | PASS | 5 | 1200 | — |
| softplus10_d64_l2_f2 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| softplus3_d64_l2_f2 | FAIL | 0 | — | sw1_normalized, mean_error |
| softplus5_d128_l2_f2 | FAIL | 0 | — | sw1_normalized, mean_error, covariance_error |
| softplus5_d128_l3_f2 | FAIL | 4 | — | Final bounds pass; insufficient final passing checks |
| softplus5_d64_l3_f2 | FAIL | 1 | — | Final bounds pass; insufficient final passing checks |
| softplus5_d96_l2_f2 | FAIL | 2 | — | Final bounds pass; insufficient final passing checks |

### vector_spiral

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| original architecture | PASS | 23 | 400 | — |
| d128_l3_f3 | PASS | 24 | 334 | — |
| d128_l3_f4 | PASS | 22 | 467 | — |
| d256_l3_f3 | PASS | 23 | 400 | — |
| axis_softplus5 | PASS | 22 | 467 | — |
| axis_tanh | PASS | 15 | 934 | — |
| softplus10_d64_l2_f2 | PASS | 22 | 467 | — |
| softplus5_d128_l3_f2 | PASS | 24 | 334 | — |

### img_stripes2

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| baseline | FAIL | 0 | — | modes |
| residual16 | PASS | 20 | 225 | — |
| transpose24 | FAIL | 0 | — | modes |
| residual12 | PASS | 8 | 525 | — |
| transpose16 | FAIL | 0 | — | modes |

### img_bars4

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| baseline | FAIL | 0 | — | modes |
| residual16 | PASS | 7 | 550 | — |
| transpose24 | FAIL | 0 | — | modes |
| residual12 | FAIL | 0 | — | modes |
| transpose16 | FAIL | 0 | — | modes, hq |

### img_blobs4

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| baseline | PASS | 8 | 525 | — |
| residual16 | PASS | 7 | 550 | — |
| transpose24 | FAIL | 0 | — | modes, hq |
| residual12 | FAIL | 0 | — | modes, hq |
| transpose16 | FAIL | 0 | — | modes |

### img_intensity2

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| baseline | FAIL | 0 | — | modes, hq |
| residual16 | PASS | 6 | 575 | — |
| transpose24 | FAIL | 0 | — | modes, hq |
| residual12 | PASS | 7 | 550 | — |
| transpose16 | FAIL | 0 | — | modes, hq |

</details>

## rp_logistic_bcap10

Required: 8/9. Practical: 7/10. Eligible on required tests: False.

| Problem / condition | Supported | Passing / tested architectures | Earliest confirmed architecture |
| --- | --- | ---: | --- |
| vector_two_broad | PASS | 1/1 | cap10 original architecture at 450 |
| vector_unequal_mass | FAIL | 0/1 | — |
| vector_unequal_width | PASS | 1/1 | cap10 original architecture at 1200 |
| vector_anisotropic | PASS | 1/1 | cap10 original architecture at 850 |
| vector_overlap | FAIL | 0/1 | — |
| vector_spiral | PASS | 1/1 | cap10 original architecture at 400 |
| img_stripes2 | PASS | 1/1 | residual16_cap10 at 250 |
| img_bars4 | FAIL | 0/1 | — |
| img_blobs4 | PASS | 1/1 | residual16_cap10 at 550 |
| img_intensity2 | PASS | 1/1 | residual16_cap10 at 575 |

<details><summary>Every architecture trial for rp_logistic_bcap10, including failures</summary>

### vector_two_broad

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | PASS | 20 | 450 | — |

### vector_unequal_mass

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | FAIL | 0 | — | component_covariance_error |

### vector_unequal_width

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | PASS | 5 | 1200 | — |

### vector_anisotropic

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | PASS | 12 | 850 | — |

### vector_overlap

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | FAIL | 2 | — | Final bounds pass; insufficient final passing checks |

### vector_spiral

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| cap10 original architecture | PASS | 23 | 400 | — |

### img_stripes2

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| residual16_cap10 | PASS | 19 | 250 | — |

### img_bars4

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| residual16_cap10 | FAIL | 0 | — | hq |

### img_blobs4

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| residual16_cap10 | PASS | 7 | 550 | — |

### img_intensity2

| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | ---: | ---: | --- |
| residual16_cap10 | PASS | 6 | 575 | — |

</details>

The image variants include changes to G as well as D. Discriminator width changes alone, generator changes, and their combination are explicit in the machine-readable architecture records. These are inspected development cases with one initialization; architecture support is not a fresh-transfer result.

[Exact axes, metrics and source artifacts](leaderboard.json).

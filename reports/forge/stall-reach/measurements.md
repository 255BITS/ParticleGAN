# Stall-reach critic read: measurements

Executed source `ee0d05728ded8d912f577f206a2a874e82759e21`; protocol seed 0; ordinary revision 9. Every task metric and its frozen numerical gate. Unreached tasks remain unmeasured.

## base: gaussian1d_smoke — PASS

any scheduled full pass with independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 1 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.191566 | <= 0.2 | PASS |
| std_ratio | 1.13677 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.229587 | <= 0.05 | FAIL |
| mean | 2.09578 | No declared gate | DIAGNOSTIC |
| std | 0.568385 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 209 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 1 | No declared gate | DIAGNOSTIC |

## base: two_pole — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 17 | >= 5 | PASS |
| passing scheduled observations | 17 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| mean_abs | 0.958415 | >= 0.3 | PASS |
| grad_med | 0.953116 | <= 1 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_score | 1.43762 | No declared gate | DIAGNOSTIC |
| nearest | 0.0415846 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.5 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 40 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 27 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 17 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 17 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 27 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.658415 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.3 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.958415 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0468844 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 1 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.953116 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## base: unused_token_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 9 | >= 5 | PASS |
| passing scheduled observations | 15 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| unused_hold | 0.981873 | >= 0.85 | PASS |
| concept_move | 0.991633 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos | 0.997795 | No declared gate | DIAGNOSTIC |
| concept_delta_norm | 1.00618 | No declared gate | DIAGNOSTIC |
| scale0_err | 0 | No declared gate | DIAGNOSTIC |
| unused_dist | 0.0181265 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.835 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 167 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 50 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 15 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 9 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 134 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.131873 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.981873 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.141633 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.991633 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## base: ae_gan_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 22 | >= 5 | PASS |
| passing scheduled observations | 22 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00510909 | <= 0.05 | PASS |
| hold | 0.0252829 | <= 0.35 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| adv_steps | 250 | No declared gate | DIAGNOSTIC |
| init_recon_mse | 1.24921 | No declared gate | DIAGNOSTIC |
| penalty_applied | 250 | No declared gate | DIAGNOSTIC |
| steps | 250 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.292 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 73 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 32 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0448909 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00510909 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.324717 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0252829 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## base: ring16_acquisition — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 0 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 14 | >= 16 | FAIL |
| mass_tv | 0.0449219 | <= 0.15 | PASS |
| hq | 0.529541 | >= 0.85 | FAIL |
| component_covariance_error | 3.33305 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.467077 | >= 0.15 | PASS |
| component_core_covariance_error | 1.01455 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.738316 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.578022 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 1.92935 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 2.03395 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.844124 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.881827 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 1.71036 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.622761 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.327777 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.604698 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 1.52404 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.59309 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.119462 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.237033 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.64672 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.841348 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 1.06398 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.533026 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.725074 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.8907 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 1.33508 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.601936 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.577379 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.522865 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.556467 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.411845 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.704752 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.694795 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 1.02324 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.681327 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.963385 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.65738 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.411845 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 260 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 306 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 276 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 248 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 203 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 241 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 249 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 221 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 303 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 239 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 250 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 254 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 263 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 215 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 293 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 275 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 1.42762 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 5.08089 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 9.3789 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 8.15966 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 2.17682 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 3.51802 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 3.36358 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.44514 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.259013 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.919792 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 3.07243 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 2.44394 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 1.05024 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.637451 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 5.70009 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 4.69516 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0634766 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.074707 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0673828 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0605469 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0495605 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0588379 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.060791 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0539551 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0739746 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0583496 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0610352 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0620117 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.064209 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0524902 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0715332 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0671387 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | false | No declared gate | DIAGNOSTIC |
| component_resolved[1] | false | No declared gate | DIAGNOSTIC |
| component_resolved[2] | false | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_resolved[4] | false | No declared gate | DIAGNOSTIC |
| component_resolved[5] | false | No declared gate | DIAGNOSTIC |
| component_resolved[6] | false | No declared gate | DIAGNOSTIC |
| component_resolved[7] | false | No declared gate | DIAGNOSTIC |
| component_resolved[8] | false | No declared gate | DIAGNOSTIC |
| component_resolved[9] | false | No declared gate | DIAGNOSTIC |
| component_resolved[10] | false | No declared gate | DIAGNOSTIC |
| component_resolved[11] | false | No declared gate | DIAGNOSTIC |
| component_resolved[12] | false | No declared gate | DIAGNOSTIC |
| component_resolved[13] | false | No declared gate | DIAGNOSTIC |
| component_resolved[14] | false | No declared gate | DIAGNOSTIC |
| component_resolved[15] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.384615 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.787582 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.858696 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.782258 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.679803 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.456432 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.461847 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.321267 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.254125 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0585774 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.304 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.188976 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.254753 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.251163 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.604096 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.756364 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0652655 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 160 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 65 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 39 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 54 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 65 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 131 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 134 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 150 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 226 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 225 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 174 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 206 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 196 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 161 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 116 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 67 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.858696 | No declared gate | DIAGNOSTIC |
| mean_error | 0.102428 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.152344 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.792969 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.066854 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[4] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[5] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[6] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[7] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[8] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[9] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[10] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[11] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[12] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[13] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[14] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[15] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_scale | 3.00292 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 14 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.105078 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0449219 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.320459 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.529541 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -2.48305 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 3.33305 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.317077 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.467077 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.417002 | No declared gate | DIAGNOSTIC |

## base: five_word_joint_smoke — INCOMPLETE

task wall-time budget exhausted

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | unavailable | >= 1 | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | unavailable | >= 1024 | UNAVAILABLE |
| quality_fraction | unavailable | >= 0.95 | UNAVAILABLE |
| modes | unavailable | == 5 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| reconstruction_exact | unavailable | == 1 | UNAVAILABLE |
| minimum_reconstruction_token_probability | unavailable | >= 0.9 | UNAVAILABLE |

## control: gaussian1d_smoke — PASS

any scheduled full pass with independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 2 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.125898 | <= 0.2 | PASS |
| std_ratio | 1.06304 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0718422 | <= 0.05 | FAIL |
| mean | 2.06295 | No declared gate | DIAGNOSTIC |
| std | 0.531522 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 375 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 875 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 375 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 3 | No declared gate | DIAGNOSTIC |

## control: two_pole — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 17 | >= 5 | PASS |
| passing scheduled observations | 17 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| mean_abs | 0.958502 | >= 0.3 | PASS |
| grad_med | 0.952346 | <= 1 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_score | 1.43775 | No declared gate | DIAGNOSTIC |
| nearest | 0.0414975 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.5 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 40 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 27 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 17 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 17 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 27 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.658502 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.3 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.958502 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0476543 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 1 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.952346 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: unused_token_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 5 | >= 5 | PASS |
| passing scheduled observations | 14 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| unused_hold | 0.986702 | >= 0.85 | PASS |
| concept_move | 0.959325 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos | 0.999984 | No declared gate | DIAGNOSTIC |
| concept_delta_norm | 1.04066 | No declared gate | DIAGNOSTIC |
| scale0_err | 0 | No declared gate | DIAGNOSTIC |
| unused_dist | 0.013298 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 1 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 200 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 50 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 14 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 5 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 167 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.136702 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.986702 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.109325 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.959325 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: ae_gan_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 22 | >= 5 | PASS |
| passing scheduled observations | 22 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00435301 | <= 0.05 | PASS |
| hold | 0.015781 | <= 0.35 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| adv_steps | 250 | No declared gate | DIAGNOSTIC |
| init_recon_mse | 1.24921 | No declared gate | DIAGNOSTIC |
| penalty_applied | 250 | No declared gate | DIAGNOSTIC |
| steps | 250 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.292 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 73 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 32 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.045647 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00435301 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.334219 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.015781 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: ring16_acquisition — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 26 | >= 5 | PASS |
| passing scheduled observations | 26 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0651855 | <= 0.15 | PASS |
| hq | 0.960938 | >= 0.85 | PASS |
| component_covariance_error | 0.496567 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.242467 | >= 0.15 | PASS |
| component_core_covariance_error | 0.466545 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.315736 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.369424 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.594593 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.518116 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.432995 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.537865 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.575225 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.732785 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.312458 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.291613 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.620701 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.607636 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.362891 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.379506 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.314538 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.498637 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.565275 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.630006 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.423393 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.357952 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.441093 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.242708 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.273447 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.436765 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.624645 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.605689 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.46883 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.678126 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.662094 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.465717 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.574186 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.782796 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.242708 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 318 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 233 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 199 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 249 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 311 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 244 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 249 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 209 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 344 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 223 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 248 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 197 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 305 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 259 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 266 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 242 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.326795 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.447392 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.635604 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.52558 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.432995 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.536013 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.575225 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 0.82481 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.312458 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.291613 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.748908 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.607636 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.409289 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.378566 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.297045 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.595143 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0776367 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0568848 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.048584 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.060791 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0759277 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0595703 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.060791 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0510254 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0839844 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0544434 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0605469 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0480957 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0744629 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0632324 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0649414 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.059082 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | false | No declared gate | DIAGNOSTIC |
| component_resolved[1] | false | No declared gate | DIAGNOSTIC |
| component_resolved[2] | false | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_resolved[4] | false | No declared gate | DIAGNOSTIC |
| component_resolved[5] | false | No declared gate | DIAGNOSTIC |
| component_resolved[6] | false | No declared gate | DIAGNOSTIC |
| component_resolved[7] | false | No declared gate | DIAGNOSTIC |
| component_resolved[8] | false | No declared gate | DIAGNOSTIC |
| component_resolved[9] | false | No declared gate | DIAGNOSTIC |
| component_resolved[10] | false | No declared gate | DIAGNOSTIC |
| component_resolved[11] | false | No declared gate | DIAGNOSTIC |
| component_resolved[12] | false | No declared gate | DIAGNOSTIC |
| component_resolved[13] | false | No declared gate | DIAGNOSTIC |
| component_resolved[14] | false | No declared gate | DIAGNOSTIC |
| component_resolved[15] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0377358 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.055794 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.040201 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0200803 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0257235 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0245902 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0240964 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.062201 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0290698 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0134529 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.100806 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0406091 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0327869 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.015444 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0300752 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.0867769 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0610125 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 306 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 220 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 191 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 244 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 303 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 238 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 243 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 196 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 334 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 220 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 223 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 189 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 295 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 255 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 258 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 221 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.100806 | No declared gate | DIAGNOSTIC |
| mean_error | 0.00799623 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.738281 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.769531 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0269808 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[4] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[5] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[6] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[7] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[8] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[9] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[10] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[11] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[12] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[13] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[14] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_mass[15] | 0.0625 | No declared gate | DIAGNOSTIC |
| target_scale | 3.00292 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.78125 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1250 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 1184 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 26 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 26 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 1184 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0848145 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0651855 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.110938 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.960938 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.353433 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.496567 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.0924671 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.242467 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: five_word_joint_smoke — INCOMPLETE

task wall-time budget exhausted

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | unavailable | >= 1 | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | unavailable | >= 1024 | UNAVAILABLE |
| quality_fraction | unavailable | >= 0.95 | UNAVAILABLE |
| modes | unavailable | == 5 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| reconstruction_exact | unavailable | == 1 | UNAVAILABLE |
| minimum_reconstruction_token_probability | unavailable | >= 0.9 | UNAVAILABLE |

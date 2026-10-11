# BCAP prior coulomb search

Executed source `2ce01aec33c644ded0cf7f29b130bdb091e643ff`; protocol seed 0; ordinary revision 9.

Read the [interpretation and recommendation](readout.md), [identity and terminal-witness audit](audit.json), [saved prior geometry](prior-geometry.json), and [actual-training GIF receipt index](media/index.json).

Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.

| Strength | Tier 1 passes | Tier 2 passes | Repaired | Regressed | Unresolved control passes |
| --- | --- | --- | --- | --- | --- |
| {'prior_reg': 0.01} | 6/6 | 8/21 | none | five_word_joint_hold | none |
| {'prior_reg': 0.001} | 5/6 | 0/21 | none | none | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |
| {'prior_reg': 0.1} | 6/6 | 8/21 | none | vector_two_broad | none |

Every task metric and its frozen numerical gate are retained below and in [results.json](results.json). Unreached tasks remain unmeasured.

The goal's [current leaderboard](../../technique-inventory.md) remains the sole generated leaderboard; this table is a source-bound experimental comparison.

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

## control: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 23 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999617 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.208008 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 3.4861e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 23 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 1667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 834 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 23 | No declared gate | DIAGNOSTIC |

## control: gaussian1d_stability — FAIL

stationary hold, deadline reacquisition and shifted hold

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | 2 | == 72 | FAIL |
| deadline reacquisition | false | == true | FAIL |
| shifted hold passing checks | 0 | == 24 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.255238 | <= 0.2 | FAIL |
| std_ratio | 0.66233 | >= 0.8 and <= 1.2 | FAIL |
| cdf_ks | 0.320623 | <= 0.05 | FAIL |
| mean | 3.12762 | No declared gate | DIAGNOSTIC |
| std | 0.331165 | No declared gate | DIAGNOSTIC |
| step | 6000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| frozen_checks | 48 | No declared gate | DIAGNOSTIC |
| frozen_passes | 0 | No declared gate | DIAGNOSTIC |
| shift_hold_checks | 24 | No declared gate | DIAGNOSTIC |
| shift_hold_passes | 0 | No declared gate | DIAGNOSTIC |
| stationary_checks | 72 | No declared gate | DIAGNOSTIC |
| stationary_passes | 2 | No declared gate | DIAGNOSTIC |

## control: five_word_joint_hold — PASS

every joint generation/inverse check throughout the fixed continuation window

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passing primary checks | 25 | == 25 | PASS |
| confirmed checks | 25 | == 25 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0208984 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.951635 | >= 0.9 | PASS |
| completed_steps | 4834 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.21582 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.191406 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.205078 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.193359 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.194336 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 0.00218637 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 4834 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 25 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 1001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 1168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 1334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 1501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 1668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 1834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 2001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 2168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 2334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 2668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 2834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 3001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 3168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 3501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 3668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 3834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 4001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 4168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 4334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 4501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[23] | 4668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[24] | 4834 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 834 | No declared gate | DIAGNOSTIC |
| observations | 25 | No declared gate | DIAGNOSTIC |
| passing_observations | 25 | No declared gate | DIAGNOSTIC |

## control: trajectory — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 19 | >= 5 | PASS |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000216867 | <= 0.02 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| critic_gradient_max | 1.04656 | No declared gate | DIAGNOSTIC |
| critic_gradient_median | 0.767898 | No declared gate | DIAGNOSTIC |
| own_nearest_fraction | 1 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000216867 | No declared gate | DIAGNOSTIC |
| particle_mean_square | 0.22033 | No declared gate | DIAGNOSTIC |
| particle_std_mean | 0.481338 | No declared gate | DIAGNOSTIC |
| set_cover | 0.00346987 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.4175 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 167 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 19 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0197831 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000216867 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: residual_student — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000159604 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| both_land_rows | 12 | No declared gate | DIAGNOSTIC |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| endpoint_l2 | 0.0270144 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| land_tol | 0.25 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000159604 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| pass_identity_mse | 0.02 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| residual_weight | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| slow_impact_max | 0.1 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |
| success_min | 1 | No declared gate | DIAGNOSTIC |
| vicreg_weight | 0 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 150 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 84 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 84 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0198404 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000159604 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 1 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: unipolar — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 18 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover | 0.9938 | >= 0.85 | PASS |
| off_caption | 6.61748e-05 | <= 0.05 | PASS |
| neu_hold | 0.989787 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cos_plus | 0.999908 | No declared gate | DIAGNOSTIC |
| cover_weight | 0 | No declared gate | DIAGNOSTIC |
| even_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| half_norm_ratio | 0.500457 | No declared gate | DIAGNOSTIC |
| hit | true | No declared gate | DIAGNOSTIC |
| mag_ratio | 0.993892 | No declared gate | DIAGNOSTIC |
| odd_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| origin_norm | 0.0102133 | No declared gate | DIAGNOSTIC |
| polarity | 1 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.46 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 184 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 18 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 117 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.1438 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.9938 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0499338 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 6.61748e-05 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.139787 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.989787 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: cover_leftover — PASS

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
| u_kept | 0.986895 | >= 0.85 | PASS |
| content_kept | 0.986414 | >= 0.75 | PASS |
| leak_ratio | 0.016776 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0159148 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0150323 | <= 0.2 | PASS |
| same_dir | 0.00243978 | <= 0.25 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| covered | true | No declared gate | DIAGNOSTIC |
| even_norm | 0.00127678 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| kappa | 1 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.999998 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| odd_norm | 1.11419 | No declared gate | DIAGNOSTIC |
| on_content | 0.537246 | No declared gate | DIAGNOSTIC |
| on_e | -0.00127874 | No declared gate | DIAGNOSTIC |
| on_u | 0.976864 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| particle_rms | 0.217452 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.2925 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 234 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.136895 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.986895 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.236414 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.986414 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.183224 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.016776 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.184085 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.0159148 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.184968 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0150323 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.24756 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.00243978 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: mid_scale_identity — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos_plus | 0.999944 | >= 0.85 | PASS |
| concept_cos_minus | 0.999962 | >= 0.85 | PASS |
| concept_mag_plus | 1.00204 | >= 0.75 and <= 1.25 | PASS |
| concept_mag_minus | 1.00013 | >= 0.75 and <= 1.25 | PASS |
| identity_at_0 | 0.984797 | >= 0.85 | PASS |
| identity_at_mid | 0.992997 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.99985 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_calls | 3200 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| same_dir | 0.00864692 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 300 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 167 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.149944 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.999944 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.149962 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.999962 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.25204 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.24796 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.250134 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.249866 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[6].margin | 0.134797 | No declared gate | DIAGNOSTIC |
| metrics[6].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[6].value | 0.984797 | No declared gate | DIAGNOSTIC |
| metrics[7].margin | 0.142997 | No declared gate | DIAGNOSTIC |
| metrics[7].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[7].value | 0.992997 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: mode_hold — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 3 | >= 5 | FAIL |
| passing scheduled observations | 10 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 8 | >= 8 | PASS |
| hq | 0.989258 | >= 0.9 | PASS |
| cover | 1 | No declared gate | DIAGNOSTIC |
| effective_modes | 7.15797 | No declared gate | DIAGNOSTIC |
| n_modes | 8 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 700 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 10 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 3 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 8 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 8 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0892578 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.989258 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: vector_two_broad — PASS

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
| sw1_normalized | 0.13459 | <= 0.18 | PASS |
| mass_tv | 0.0629883 | <= 0.15 | PASS |
| hq | 0.98877 | >= 0.85 | PASS |
| component_covariance_error | 0.385581 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.400066 | >= 0.15 | PASS |
| component_core_covariance_error | 0.385581 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.455045 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.316117 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.400066 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.553021 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.400066 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2306 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1790 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.455045 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.316117 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.562988 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.437012 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00563747 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0184358 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.111471 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0184358 | No declared gate | DIAGNOSTIC |
| mean_error | 0.191464 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.874023 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.385581 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.400066 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0184358 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.5 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.5 | No declared gate | DIAGNOSTIC |
| target_scale | 1.05556 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.291667 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 350 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 150 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 150 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0454105 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.13459 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0870117 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0629883 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.13877 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.98877 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.464419 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.385581 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.250066 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.400066 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: vector_unequal_mass — FAIL

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
| sw1_normalized | 0.154638 | <= 0.18 | PASS |
| mass_tv | 0.0706543 | <= 0.15 | PASS |
| hq | 0.95459 | >= 0.85 | PASS |
| component_covariance_error | 3.69165 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.00907291 | >= 0.15 | FAIL |
| min_mass_ratio | 0.20752 | >= 0.25 | FAIL |
| component_core_covariance_error | 0.530393 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.245608 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.463451 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.454114 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.958399 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.734128 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.380592 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.415286 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.00907291 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.00907291 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2273 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1498 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 308 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 17 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 1.02757 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.463451 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 12.3172 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.958399 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.554932 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.365723 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0751953 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.00415039 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0184778 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0493992 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.227273 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0803272 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.227273 | No declared gate | DIAGNOSTIC |
| mean_error | 0.211882 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.387724 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.380592 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.227273 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.55 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.3 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.13 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.02 | No declared gate | DIAGNOSTIC |
| target_scale | 1.75933 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0253616 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.154638 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0793457 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0706543 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.10459 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.95459 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -2.84165 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 3.69165 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.140927 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.00907291 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | -0.0424805 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.20752 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.518239 | No declared gate | DIAGNOSTIC |

## control: vector_unequal_width — FAIL

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
| sw1_normalized | 0.295865 | <= 0.18 | FAIL |
| mass_tv | 0.291504 | <= 0.15 | FAIL |
| hq | 0.97876 | >= 0.85 | PASS |
| component_covariance_error | 6.28756 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.307472 | >= 0.15 | PASS |
| component_core_covariance_error | 0.444178 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.630566 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.551118 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.412632 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.182396 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.307472 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.308032 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.432674 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 1.04737 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.307472 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1475 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 197 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1767 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 657 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.630566 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 23.9278 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.409488 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.182396 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.360107 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0480957 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.431396 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.1604 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0862944 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0294284 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0273973 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.25809 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0862944 | No declared gate | DIAGNOSTIC |
| mean_error | 0.409949 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.192383 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.444178 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.307472 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0862944 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.25 | No declared gate | DIAGNOSTIC |
| target_scale | 2.1344 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.115865 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.295865 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.141504 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.291504 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.12876 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.97876 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -5.43756 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 6.28756 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.157472 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.307472 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.717411 | No declared gate | DIAGNOSTIC |

## control: vector_anisotropic — FAIL

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
| sw1_normalized | 0.197952 | <= 0.18 | FAIL |
| mass_tv | 0.195964 | <= 0.15 | FAIL |
| hq | 0.97998 | >= 0.85 | PASS |
| component_covariance_error | 0.449625 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.204013 | >= 0.15 | PASS |
| component_core_covariance_error | 0.344346 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.0422367 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.493619 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.497183 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.436122 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.204013 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.412602 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.204013 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1276 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 2168 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 652 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.146985 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.493619 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.708271 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.311523 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.529297 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.15918 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0125392 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.00461255 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0858896 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.338893 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0858896 | No declared gate | DIAGNOSTIC |
| mean_error | 0.236065 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.477539 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.344346 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.204013 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0858896 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_scale | 2.03229 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0179519 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.197952 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.0459635 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.195964 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.12998 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.97998 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.400375 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.449625 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.0540131 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.204013 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0812313 | No declared gate | DIAGNOSTIC |

## control: vector_overlap — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 5 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.255769 | <= 0.18 | FAIL |
| mean_error | 0.388766 | <= 0.15 | FAIL |
| covariance_error | 0.43849 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_scale | 0.850144 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 5 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0757691 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.255769 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.238766 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.388766 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0115097 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.43849 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.670905 | No declared gate | DIAGNOSTIC |

## control: vector_spiral — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 23 | >= 5 | PASS |
| passing scheduled observations | 23 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.0587562 | <= 0.18 | PASS |
| mean_error | 0.0824902 | <= 0.15 | PASS |
| covariance_error | 0.0718014 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| target_scale | 1.22418 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.25 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 400 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 134 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 23 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 23 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 134 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.121244 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0587562 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0675098 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0824902 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.378199 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0718014 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: img_stripes2 — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 12 | >= 5 | PASS |
| passing scheduled observations | 12 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 2 | >= 2 | PASS |
| hq | 1 | >= 0.9 | PASS |
| distribution_tv | 0.125 | No declared gate | DIAGNOSTIC |
| mean_rmse | 5.96898e-05 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.625 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.375 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.708333 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 425 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 325 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 12 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 12 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 325 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.1 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## control: img_bars4 — FAIL

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
| modes | 2 | >= 4 | FAIL |
| hq | 0.96875 | >= 0.9 | PASS |
| distribution_tv | 0.40625 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0139485 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.03125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.06875 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.96875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.25 | No declared gate | DIAGNOSTIC |

## control: img_blobs4 — FAIL

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
| modes | 2 | >= 4 | FAIL |
| hq | 0.75 | >= 0.9 | FAIL |
| distribution_tv | 0.1875 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0562021 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.0625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0.03125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0.375 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.75 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.333333 | No declared gate | DIAGNOSTIC |

## control: img_intensity2 — FAIL

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
| modes | 2 | >= 2 | PASS |
| hq | 0.6875 | >= 0.9 | FAIL |
| distribution_tv | 0.03125 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0601416 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.53125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.46875 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.2125 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.6875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.118056 | No declared gate | DIAGNOSTIC |

## control: grid100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.24435 | >= 0.97 | FAIL |
| modes | 9 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.148 | <= 0.1 | FAIL |
| max_mode_mass | 0.0228 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 4.14073 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.50679 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.24435 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2444 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.148 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.744491 | No declared gate | DIAGNOSTIC |
| within_radius_n | 4888 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.24072 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.371385 | <= 0.1 | FAIL |
| center_rms_sigma | 1.52031 | <= 0.2 | FAIL |
| mass_tv | 0.14718 | <= 0.06 | FAIL |
| radial_ks | 0.490883 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.51012 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.49686 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.662866 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.371385 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.748171 | No declared gate | DIAGNOSTIC |
| within_radius_n | 24072 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.24072 | >= 0.97 | FAIL |
| modes | 8 | == 100 | FAIL |
| min_hq_mode_mass | 2e-05 | >= 0.005 | FAIL |
| mass_tv | 0.14718 | <= 0.1 | FAIL |
| max_mode_mass | 0.02201 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.62235 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.5768 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.29117 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.24072 | No declared gate | DIAGNOSTIC |
| min_hq_count | 2 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.24072 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.371385 | <= 0.1 | FAIL |
| center_rms_sigma | 1.52031 | <= 0.2 | FAIL |
| mass_tv | 0.14718 | <= 0.06 | FAIL |
| radial_ks | 0.490883 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.51012 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.49686 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.662866 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.371385 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.748171 | No declared gate | DIAGNOSTIC |
| within_radius_n | 24072 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279445 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417376 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151412 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113417 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279445 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## control: rotated100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2478 | >= 0.97 | FAIL |
| modes | 13 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.1397 | <= 0.1 | FAIL |
| max_mode_mass | 0.02485 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.97754 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.18165 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.2478 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.24785 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.1397 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.741041 | No declared gate | DIAGNOSTIC |
| within_radius_n | 4957 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.25552 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.14325 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.733371 | No declared gate | DIAGNOSTIC |
| within_radius_n | 25552 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.25552 | >= 0.97 | FAIL |
| modes | 14 | == 100 | FAIL |
| min_hq_mode_mass | 1e-05 | >= 0.005 | FAIL |
| mass_tv | 0.14325 | <= 0.1 | FAIL |
| max_mode_mass | 0.0252 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.58847 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.37206 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.488 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.25552 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.25552 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.14325 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.733371 | No declared gate | DIAGNOSTIC |
| within_radius_n | 25552 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.00027944 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151327 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113412 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.100913 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.00027944 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## control: staggered100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.29765 | >= 0.97 | FAIL |
| modes | 20 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.12455 | <= 0.1 | FAIL |
| max_mode_mass | 0.0225 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.07548 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.35846 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.29765 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.29765 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.12455 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.691241 | No declared gate | DIAGNOSTIC |
| within_radius_n | 5953 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.30168 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.12277 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.687211 | No declared gate | DIAGNOSTIC |
| within_radius_n | 30168 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.30168 | >= 0.97 | FAIL |
| modes | 19 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.12277 | <= 0.1 | FAIL |
| max_mode_mass | 0.02277 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.25366 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.26078 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.30168 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.30168 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.12277 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.687211 | No declared gate | DIAGNOSTIC |
| within_radius_n | 30168 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279439 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151475 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113421 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279439 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## control: clockfree_audit_measurement_v1 — PASS

declared state/horizon/cadence/restart comparisons agree; source audit bound

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| step_label: exact learning-state parity | true | == true | PASS |
| horizon: exact learning-state parity | true | == true | PASS |
| evaluation_cadence: exact learning-state parity | true | == true | PASS |
| restart: exact learning-state parity | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| parity_comparisons | 4 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.0330389 | <= 0.2 | PASS |
| std_ratio | 0.981187 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0562667 | <= 0.05 | FAIL |
| mean | 2.01652 | No declared gate | DIAGNOSTIC |
| std | 0.490594 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 542 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 917 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 542 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 3 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: two_pole — PASS

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

## {'prior_reg': 0.01}: unused_token_hold — PASS

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

## {'prior_reg': 0.01}: ae_gan_hold — PASS

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
| recon_mse | 0.00557108 | <= 0.05 | PASS |
| hold | 0.0131855 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0444289 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00557108 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.336815 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0131855 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: ring16_acquisition — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 38 | >= 5 | PASS |
| passing scheduled observations | 39 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.076416 | <= 0.15 | PASS |
| hq | 0.947998 | >= 0.85 | PASS |
| component_covariance_error | 0.529464 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.328701 | >= 0.15 | PASS |
| component_core_covariance_error | 0.449681 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.490618 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.757454 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.526247 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.489744 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.192739 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.421654 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.750601 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.838981 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.269049 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.355381 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.405524 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.263741 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.23039 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.33846 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.396288 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.468019 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.650579 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.470862 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.561219 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.398342 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.758856 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.487137 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.318079 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.661501 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.63728 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.527447 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.426785 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.702524 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.676054 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.536537 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.44754 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.412033 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.318079 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 340 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 237 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 224 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 273 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 315 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 257 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 204 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 208 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 336 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 229 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 206 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 200 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 300 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 267 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 227 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 273 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.836231 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.01654 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.551078 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.553553 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.436944 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.421654 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.781998 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.01152 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.269049 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.370811 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.388483 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.315405 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.23039 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.33023 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.474686 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.482847 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0830078 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0578613 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0546875 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0666504 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0769043 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0627441 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0498047 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0507812 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0820312 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0559082 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.050293 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0488281 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0732422 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0651855 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0554199 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0666504 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.102941 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.092827 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0491071 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.032967 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0507937 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0294118 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.0865385 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0178571 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0436681 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0582524 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.08 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0566667 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0524345 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0528634 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.032967 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0692176 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 305 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 215 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 213 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 264 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 299 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 257 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 198 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 330 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 219 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 194 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 184 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 283 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 253 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 215 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 264 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.102941 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0474595 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.71875 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.78125 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0385498 | No declared gate | DIAGNOSTIC |
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
| confirmation_fraction | 0.65625 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1050 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 867 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 39 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 38 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 984 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.073584 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.076416 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.097998 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.947998 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.320536 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.529464 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.178701 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.328701 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 19 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999105 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.208008 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 5.40826e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 19 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 4167 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 19 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: gaussian1d_stability — FAIL

stationary hold, deadline reacquisition and shifted hold

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | 9 | == 72 | FAIL |
| deadline reacquisition | false | == true | FAIL |
| shifted hold passing checks | 1 | == 24 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.370123 | <= 0.2 | FAIL |
| std_ratio | 1.06479 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.125184 | <= 0.05 | FAIL |
| mean | 3.18506 | No declared gate | DIAGNOSTIC |
| std | 0.532397 | No declared gate | DIAGNOSTIC |
| step | 6000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| frozen_checks | 48 | No declared gate | DIAGNOSTIC |
| frozen_passes | 0 | No declared gate | DIAGNOSTIC |
| shift_hold_checks | 24 | No declared gate | DIAGNOSTIC |
| shift_hold_passes | 1 | No declared gate | DIAGNOSTIC |
| stationary_checks | 72 | No declared gate | DIAGNOSTIC |
| stationary_passes | 9 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: five_word_joint_hold — FAIL

every joint generation/inverse check throughout the fixed continuation window

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passing primary checks | 24 | == 25 | FAIL |
| confirmed checks | 24 | == 25 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0226563 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999907 | >= 0.9 | PASS |
| completed_steps | 8167 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.189453 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.199219 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.222656 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.193359 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.52632e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 8167 | No declared gate | DIAGNOSTIC |

Worst mandatory word-hold primary checks (saved failure witnesses)

1/25 mandatory checks failed numerical bounds. Original cohort SHA256 `8c618134062b34e50e9355ce098d00e919d028179f132452f3583e89086184b7`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 4167 | `a30a0840a9ceaae775187b3b4155e187b474d7bd48884c9fc948409a89164666` |
| quality_fraction | 0.782227 | >= 0.95 | FAIL | 5001 | `f73d1a8e2f37352c13493e7f00d678412fd4c254b07e151765a527786f5f9317` |
| modes | 4 | == 5 | FAIL | 5001 | `f73d1a8e2f37352c13493e7f00d678412fd4c254b07e151765a527786f5f9317` |
| mass_tv | 0.235742 | <= 0.1 | FAIL | 5001 | `f73d1a8e2f37352c13493e7f00d678412fd4c254b07e151765a527786f5f9317` |
| reconstruction_exact | 0 | == 1 | FAIL | 5001 | `f73d1a8e2f37352c13493e7f00d678412fd4c254b07e151765a527786f5f9317` |
| minimum_reconstruction_token_probability | 0.345344 | >= 0.9 | FAIL | 5001 | `f73d1a8e2f37352c13493e7f00d678412fd4c254b07e151765a527786f5f9317` |

Worst mandatory word-hold independent confirmation checks (saved failure witnesses)

1/25 mandatory checks failed numerical bounds. Original cohort SHA256 `3f9fed8e6dbec4cd4ad533c46b927f9e89b9e572d9500046940c68c347c35a4d`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 4167 | `53b34242974b54b974150b84c5124f4ba1bbd977fe2c51598b88ff3cf580172c` |
| quality_fraction | 0.796875 | >= 0.95 | FAIL | 5001 | `5e8ed52a022cfad7483a8570517da19d38932a4170c61509a87247a8195d1bab` |
| modes | 4 | == 5 | FAIL | 5001 | `5e8ed52a022cfad7483a8570517da19d38932a4170c61509a87247a8195d1bab` |
| mass_tv | 0.209375 | <= 0.1 | FAIL | 5001 | `5e8ed52a022cfad7483a8570517da19d38932a4170c61509a87247a8195d1bab` |
| reconstruction_exact | 0 | == 1 | FAIL | 5001 | `5e8ed52a022cfad7483a8570517da19d38932a4170c61509a87247a8195d1bab` |
| minimum_reconstruction_token_probability | 0.345344 | >= 0.9 | FAIL | 5001 | `5e8ed52a022cfad7483a8570517da19d38932a4170c61509a87247a8195d1bab` |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 24 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 4334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 4501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 4667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 4834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 5167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 5334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 5501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 5667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 6001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 6167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 6334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 6501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 6834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 7001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 7167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 7334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 7667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 7834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 8001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[23] | 8167 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 4167 | No declared gate | DIAGNOSTIC |
| observations | 25 | No declared gate | DIAGNOSTIC |
| passing_observations | 24 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: trajectory — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 19 | >= 5 | PASS |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000238842 | <= 0.02 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| critic_gradient_max | 0.826099 | No declared gate | DIAGNOSTIC |
| critic_gradient_median | 0.352886 | No declared gate | DIAGNOSTIC |
| own_nearest_fraction | 1 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000238842 | No declared gate | DIAGNOSTIC |
| particle_mean_square | 0.0180828 | No declared gate | DIAGNOSTIC |
| particle_std_mean | 0.1386 | No declared gate | DIAGNOSTIC |
| set_cover | 0.00382148 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.4175 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 167 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 19 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0197612 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000238842 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: residual_student — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 19 | >= 5 | PASS |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000174489 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| both_land_rows | 12 | No declared gate | DIAGNOSTIC |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| endpoint_l2 | 0.0273374 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| land_tol | 0.25 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000174489 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| pass_identity_mse | 0.02 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| residual_weight | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| slow_impact_max | 0.1 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |
| success_min | 1 | No declared gate | DIAGNOSTIC |
| vicreg_weight | 0.01 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.4175 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 167 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 19 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0198255 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000174489 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 1 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: unipolar — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 18 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover | 0.9938 | >= 0.85 | PASS |
| off_caption | 6.61748e-05 | <= 0.05 | PASS |
| neu_hold | 0.989787 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cos_plus | 0.999908 | No declared gate | DIAGNOSTIC |
| cover_weight | 0 | No declared gate | DIAGNOSTIC |
| even_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| half_norm_ratio | 0.500457 | No declared gate | DIAGNOSTIC |
| hit | true | No declared gate | DIAGNOSTIC |
| mag_ratio | 0.993892 | No declared gate | DIAGNOSTIC |
| odd_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| origin_norm | 0.0102133 | No declared gate | DIAGNOSTIC |
| polarity | 1 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.46 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 184 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 18 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 117 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.1438 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.9938 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0499338 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 6.61748e-05 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.139787 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.989787 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: cover_leftover — PASS

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
| u_kept | 0.987543 | >= 0.85 | PASS |
| content_kept | 0.976899 | >= 0.75 | PASS |
| leak_ratio | 0.0107198 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0161375 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0123548 | <= 0.2 | PASS |
| same_dir | 0.0163114 | <= 0.25 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| covered | true | No declared gate | DIAGNOSTIC |
| even_norm | 0.0034113 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| kappa | 1 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.999982 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| odd_norm | 1.111 | No declared gate | DIAGNOSTIC |
| on_content | 0.533405 | No declared gate | DIAGNOSTIC |
| on_e | 0.00215077 | No declared gate | DIAGNOSTIC |
| on_u | 0.973743 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| particle_rms | 0.0567921 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.2925 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 234 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.137543 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.987543 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.226899 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.976899 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.18928 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0107198 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.183862 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.0161375 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.187645 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0123548 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.233689 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.0163114 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: mid_scale_identity — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos_plus | 0.999944 | >= 0.85 | PASS |
| concept_cos_minus | 0.999962 | >= 0.85 | PASS |
| concept_mag_plus | 1.00204 | >= 0.75 and <= 1.25 | PASS |
| concept_mag_minus | 1.00013 | >= 0.75 and <= 1.25 | PASS |
| identity_at_0 | 0.984797 | >= 0.85 | PASS |
| identity_at_mid | 0.992997 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.99985 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_calls | 3200 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| same_dir | 0.00864692 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 300 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 167 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.149944 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.999944 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.149962 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.999962 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.25204 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.24796 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.250134 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.249866 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[6].margin | 0.134797 | No declared gate | DIAGNOSTIC |
| metrics[6].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[6].value | 0.984797 | No declared gate | DIAGNOSTIC |
| metrics[7].margin | 0.142997 | No declared gate | DIAGNOSTIC |
| metrics[7].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[7].value | 0.992997 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: mode_hold — FAIL

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
| modes | 7 | >= 8 | FAIL |
| hq | 0.932617 | >= 0.9 | PASS |
| cover | 0.875 | No declared gate | DIAGNOSTIC |
| effective_modes | 6.64566 | No declared gate | DIAGNOSTIC |
| n_modes | 8 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -1 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 8 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 7 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0326172 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.932617 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0625 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_two_broad — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.17234 | <= 0.18 | PASS |
| mass_tv | 0.0407715 | <= 0.15 | PASS |
| hq | 0.967285 | >= 0.85 | PASS |
| component_covariance_error | 0.527838 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.319753 | >= 0.15 | PASS |
| component_core_covariance_error | 0.527838 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.572686 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.48299 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.353325 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.319753 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.319753 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2215 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1881 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.572686 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.48299 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.540771 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.459229 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0604966 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.37411 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0604966 | No declared gate | DIAGNOSTIC |
| mean_error | 0.20776 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.918457 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.527838 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.319753 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0604966 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.5 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.5 | No declared gate | DIAGNOSTIC |
| target_scale | 1.05556 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 450 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 250 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 250 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.00765954 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.17234 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.109229 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0407715 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.117285 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.967285 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.322162 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.527838 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.169753 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.319753 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_unequal_mass — FAIL

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
| sw1_normalized | 0.159922 | <= 0.18 | PASS |
| mass_tv | 0.0789844 | <= 0.15 | PASS |
| hq | 0.983398 | >= 0.85 | PASS |
| component_covariance_error | 1.21768 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0635782 | >= 0.15 | FAIL |
| min_mass_ratio | 0.805664 | >= 0.25 | PASS |
| component_core_covariance_error | 0.525658 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.230278 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.607411 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.428949 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.835994 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.695998 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.426322 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.46315 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.0635782 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.0635782 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2171 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1003 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 856 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 66 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.230278 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 3.3755 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.428949 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.835994 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.530029 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.244873 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.208984 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0161133 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0133579 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0388834 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.216622 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0388834 | No declared gate | DIAGNOSTIC |
| mean_error | 0.224236 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.422213 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.426322 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0388834 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.55 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.3 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.13 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.02 | No declared gate | DIAGNOSTIC |
| target_scale | 1.75933 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0200779 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.159922 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0710156 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0789844 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.133398 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.983398 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.367681 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.21768 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.0864218 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0635782 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.555664 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.805664 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.168119 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_unequal_width — FAIL

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
| sw1_normalized | 0.285169 | <= 0.18 | FAIL |
| mass_tv | 0.301025 | <= 0.15 | FAIL |
| hq | 0.970703 | >= 0.85 | PASS |
| component_covariance_error | 9.37526 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.371002 | >= 0.15 | PASS |
| component_core_covariance_error | 0.531838 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.574413 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.523992 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.654526 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.374421 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.371002 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.25899 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.569152 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.482781 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.25899 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1539 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 253 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1742 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 562 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.574413 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 35.7669 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.785325 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.374421 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.375732 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0617676 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.425293 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.137207 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00194932 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.150198 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0287026 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0516014 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.166048 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.150198 | No declared gate | DIAGNOSTIC |
| mean_error | 0.377922 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.24707 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.531838 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.25899 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.150198 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.25 | No declared gate | DIAGNOSTIC |
| target_scale | 2.1344 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.105169 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.285169 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.151025 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.301025 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.120703 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.970703 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -8.52526 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 9.37526 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.221002 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.371002 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.718222 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_anisotropic — FAIL

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
| sw1_normalized | 0.348019 | <= 0.18 | FAIL |
| mass_tv | 0.289876 | <= 0.15 | FAIL |
| hq | 0.882568 | >= 0.85 | PASS |
| component_covariance_error | 17.7193 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.962194 | >= 0.15 | PASS |
| component_core_covariance_error | 0.328679 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.0872271 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.120839 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.777971 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.565752 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.677804 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.074661 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.074661 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1427 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 2491 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 178 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 31.6163 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.459261 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 21.0825 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.348389 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.608154 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.043457 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.161177 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0473705 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.747191 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.770039 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.747191 | No declared gate | DIAGNOSTIC |
| mean_error | 0.430046 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.130371 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.328679 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.074661 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.747191 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_scale | 2.03229 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.168019 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.348019 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.139876 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.289876 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0325684 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.882568 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -16.8693 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 17.7193 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.812194 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.962194 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.77319 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_overlap — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 6 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.25137 | <= 0.18 | FAIL |
| mean_error | 0.411998 | <= 0.15 | FAIL |
| covariance_error | 0.122998 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_scale | 0.850144 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 6 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0713695 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.25137 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.261998 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.411998 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.327002 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.122998 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.714383 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_spiral — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 23 | >= 5 | PASS |
| passing scheduled observations | 23 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.0452011 | <= 0.18 | PASS |
| mean_error | 0.050108 | <= 0.15 | PASS |
| covariance_error | 0.0677141 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| target_scale | 1.22418 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.25 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 400 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 134 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 23 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 23 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 134 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.134799 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0452011 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.099892 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.050108 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.382286 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0677141 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: img_stripes2 — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 12 | >= 5 | PASS |
| passing scheduled observations | 12 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 2 | >= 2 | PASS |
| hq | 1 | >= 0.9 | PASS |
| distribution_tv | 0.09375 | No declared gate | DIAGNOSTIC |
| mean_rmse | 6.85506e-05 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.59375 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.59375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.40625 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.708333 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 425 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 325 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 12 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 12 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 325 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.1 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: img_bars4 — FAIL

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
| modes | 2 | >= 4 | FAIL |
| hq | 0.9375 | >= 0.9 | PASS |
| distribution_tv | 0.4375 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0232418 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.03125 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.03125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0375 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.9375 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.25 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: img_blobs4 — FAIL

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
| modes | 3 | >= 4 | FAIL |
| hq | 0.9375 | >= 0.9 | PASS |
| distribution_tv | 0.1875 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0229626 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.1875 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0.40625 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -1 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 3 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0375 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.9375 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.125 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: img_intensity2 — FAIL

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
| modes | 2 | >= 2 | PASS |
| hq | 0.5625 | >= 0.9 | FAIL |
| distribution_tv | 0 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0599538 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.5 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.5 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.3375 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.5625 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.1875 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: grid100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.28515 | >= 0.97 | FAIL |
| modes | 13 | == 100 | FAIL |
| min_hq_mode_mass | 5e-05 | >= 0.005 | FAIL |
| mass_tv | 0.13975 | <= 0.1 | FAIL |
| max_mode_mass | 0.0212 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 4.02049 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.30121 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.35205 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28515 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.28515 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.13975 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.703741 | No declared gate | DIAGNOSTIC |
| within_radius_n | 5703 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.2803 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.475674 | <= 0.1 | FAIL |
| center_rms_sigma | 1.29296 | <= 0.2 | FAIL |
| mass_tv | 0.1423 | <= 0.06 | FAIL |
| radial_ks | 0.417246 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.00609 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.36693 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.670115 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.475674 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.708591 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28030 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28029 | >= 0.97 | FAIL |
| modes | 14 | == 100 | FAIL |
| min_hq_mode_mass | 4e-05 | >= 0.005 | FAIL |
| mass_tv | 0.1423 | <= 0.1 | FAIL |
| max_mode_mass | 0.02146 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0937262 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.29697 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.3915 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.26782 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28029 | No declared gate | DIAGNOSTIC |
| min_hq_count | 4 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.2803 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.475674 | <= 0.1 | FAIL |
| center_rms_sigma | 1.29296 | <= 0.2 | FAIL |
| mass_tv | 0.1423 | <= 0.06 | FAIL |
| radial_ks | 0.417246 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.00609 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.36693 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.670115 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.475674 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.708591 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28030 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279445 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417376 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151412 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113417 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279445 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: rotated100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.28055 | >= 0.97 | FAIL |
| modes | 16 | == 100 | FAIL |
| min_hq_mode_mass | 5e-05 | >= 0.005 | FAIL |
| mass_tv | 0.1343 | <= 0.1 | FAIL |
| max_mode_mass | 0.0233 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.34597 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.10522 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.52485 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28055 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2805 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.1343 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.708391 | No declared gate | DIAGNOSTIC |
| within_radius_n | 5610 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28116 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.506447 | <= 0.1 | FAIL |
| center_rms_sigma | 1.30582 | <= 0.2 | FAIL |
| mass_tv | 0.13708 | <= 0.06 | FAIL |
| radial_ks | 0.39392 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.93156 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.7449 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.769774 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.506447 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.707731 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28116 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28117 | >= 0.97 | FAIL |
| modes | 16 | == 100 | FAIL |
| min_hq_mode_mass | 0.00013 | >= 0.005 | FAIL |
| mass_tv | 0.13708 | <= 0.1 | FAIL |
| max_mode_mass | 0.02221 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.00404388 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.63969 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.33455 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.43032 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28117 | No declared gate | DIAGNOSTIC |
| min_hq_count | 13 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28116 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.506447 | <= 0.1 | FAIL |
| center_rms_sigma | 1.30582 | <= 0.2 | FAIL |
| mass_tv | 0.13708 | <= 0.06 | FAIL |
| radial_ks | 0.39392 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.93156 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.7449 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.769774 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.506447 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.707731 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28116 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.00027944 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151327 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113412 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.100913 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.00027944 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: staggered100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.32065 | >= 0.97 | FAIL |
| modes | 16 | == 100 | FAIL |
| min_hq_mode_mass | 0.00015 | >= 0.005 | FAIL |
| mass_tv | 0.13065 | <= 0.1 | FAIL |
| max_mode_mass | 0.0246 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.000770933 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.85381 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.21141 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.33324 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.32065 | No declared gate | DIAGNOSTIC |
| min_hq_count | 3 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.32065 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.458409 | <= 0.1 | FAIL |
| center_rms_sigma | 1.25392 | <= 0.2 | FAIL |
| mass_tv | 0.13065 | <= 0.06 | FAIL |
| radial_ks | 0.389148 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.68998 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.22208 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.743069 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.458409 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.668241 | No declared gate | DIAGNOSTIC |
| within_radius_n | 6413 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.32204 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.498203 | <= 0.1 | FAIL |
| center_rms_sigma | 1.2218 | <= 0.2 | FAIL |
| mass_tv | 0.12469 | <= 0.06 | FAIL |
| radial_ks | 0.391154 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.73701 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.02351 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.678683 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.498203 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.666851 | No declared gate | DIAGNOSTIC |
| within_radius_n | 32204 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.32204 | >= 0.97 | FAIL |
| modes | 16 | == 100 | FAIL |
| min_hq_mode_mass | 0.00024 | >= 0.005 | FAIL |
| mass_tv | 0.12469 | <= 0.1 | FAIL |
| max_mode_mass | 0.02449 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.219406 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.64344 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.39293 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.2044 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.32204 | No declared gate | DIAGNOSTIC |
| min_hq_count | 24 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.32204 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.498203 | <= 0.1 | FAIL |
| center_rms_sigma | 1.2218 | <= 0.2 | FAIL |
| mass_tv | 0.12469 | <= 0.06 | FAIL |
| radial_ks | 0.391154 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.73701 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.02351 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.678683 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.498203 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.666851 | No declared gate | DIAGNOSTIC |
| within_radius_n | 32204 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279439 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151475 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113421 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279439 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: clockfree_audit_measurement_v1 — PASS

declared state/horizon/cadence/restart comparisons agree; source audit bound

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| step_label: exact learning-state parity | true | == true | PASS |
| horizon: exact learning-state parity | true | == true | PASS |
| evaluation_cadence: exact learning-state parity | true | == true | PASS |
| restart: exact learning-state parity | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| parity_comparisons | 4 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.0277494 | <= 0.2 | PASS |
| std_ratio | 0.799439 | >= 0.8 and <= 1.2 | FAIL |
| cdf_ks | 0.106707 | <= 0.05 | FAIL |
| mean | 1.98613 | No declared gate | DIAGNOSTIC |
| std | 0.399719 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 292 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 792 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 292 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 3 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: two_pole — PASS

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

## {'prior_reg': 0.001}: unused_token_hold — PASS

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

## {'prior_reg': 0.001}: ae_gan_hold — PASS

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
| recon_mse | 0.00442903 | <= 0.05 | PASS |
| hold | 0.0119835 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.045571 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00442903 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.338017 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0119835 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: ring16_acquisition — FAIL

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
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0563965 | <= 0.15 | PASS |
| hq | 0.949707 | >= 0.85 | PASS |
| component_covariance_error | 1.45461 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.358408 | >= 0.15 | PASS |
| component_core_covariance_error | 0.474356 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.345863 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.692019 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.677371 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.47037 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.335206 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.513488 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.50582 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.654875 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.39039 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.503284 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.432626 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.458832 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.282026 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.315384 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.406751 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.605395 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.584402 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.424268 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.612166 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.367099 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.533891 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.358408 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.416743 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.494465 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.501257 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.438356 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.399844 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.502933 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.601724 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.576721 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.455038 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.571193 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.358408 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 316 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 220 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 226 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 265 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 319 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 257 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 256 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 225 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 299 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 258 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 209 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 192 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 284 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 281 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 243 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 3.03807 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 13.3503 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.793924 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.486091 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.343787 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.513488 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.525391 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 0.709186 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.372006 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.568994 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.397462 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.502996 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.281157 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.303814 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.481604 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.605395 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0771484 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0537109 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0551758 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0646973 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0778809 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0627441 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0625 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0549316 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.072998 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0629883 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0510254 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.046875 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0693359 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0686035 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0593262 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.117089 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.136364 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0575221 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0415094 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0250784 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.00389105 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.015625 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.0488889 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.00334448 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0775194 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0382775 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0625 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0528169 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0177936 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0731707 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.0493827 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0738987 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 279 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 213 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 254 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 311 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 256 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 252 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 214 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 298 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 238 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 201 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 180 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 269 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 276 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 228 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 231 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.136364 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0378383 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.703125 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.75 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0346611 | No declared gate | DIAGNOSTIC |
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
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0936035 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0563965 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.099707 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.949707 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.604606 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.45461 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.208408 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.358408 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.11855 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 19 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.99929 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.198242 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 5.70247e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 19 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 5001 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 19 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: gaussian1d_stability — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | unavailable | == 72 | UNAVAILABLE |
| deadline reacquisition | unavailable | == true | UNAVAILABLE |
| shifted hold passing checks | unavailable | == 24 | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | unavailable | >= 4096 | UNAVAILABLE |
| finite_fraction | unavailable | == 1 | UNAVAILABLE |
| mean_error_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| std_ratio | unavailable | >= 0.8 and <= 1.2 | UNAVAILABLE |
| cdf_ks | unavailable | <= 0.05 | UNAVAILABLE |

## {'prior_reg': 0.001}: five_word_joint_hold — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passing primary checks | unavailable | == 25 | UNAVAILABLE |
| confirmed checks | unavailable | == 25 | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | unavailable | >= 1024 | UNAVAILABLE |
| quality_fraction | unavailable | >= 0.95 | UNAVAILABLE |
| modes | unavailable | == 5 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| reconstruction_exact | unavailable | == 1 | UNAVAILABLE |
| minimum_reconstruction_token_probability | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: trajectory — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | unavailable | <= 0.02 | UNAVAILABLE |

## {'prior_reg': 0.001}: residual_student — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | unavailable | <= 0.02 | UNAVAILABLE |
| success_rate | unavailable | >= 1 | UNAVAILABLE |
| wrong_pad_rate | unavailable | <= 0 | UNAVAILABLE |

## {'prior_reg': 0.001}: unipolar — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover | unavailable | >= 0.85 | UNAVAILABLE |
| off_caption | unavailable | <= 0.05 | UNAVAILABLE |
| neu_hold | unavailable | >= 0.85 | UNAVAILABLE |

## {'prior_reg': 0.001}: cover_leftover — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| u_kept | unavailable | >= 0.85 | UNAVAILABLE |
| content_kept | unavailable | >= 0.75 | UNAVAILABLE |
| leak_ratio | unavailable | <= 0.2 | UNAVAILABLE |
| pole_rel_err_plus | unavailable | <= 0.2 | UNAVAILABLE |
| pole_rel_err_minus | unavailable | <= 0.2 | UNAVAILABLE |
| same_dir | unavailable | <= 0.25 | UNAVAILABLE |

## {'prior_reg': 0.001}: mid_scale_identity — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos_plus | unavailable | >= 0.85 | UNAVAILABLE |
| concept_cos_minus | unavailable | >= 0.85 | UNAVAILABLE |
| concept_mag_plus | unavailable | >= 0.75 and <= 1.25 | UNAVAILABLE |
| concept_mag_minus | unavailable | >= 0.75 and <= 1.25 | UNAVAILABLE |
| identity_at_0 | unavailable | >= 0.85 | UNAVAILABLE |
| identity_at_mid | unavailable | >= 0.85 | UNAVAILABLE |

## {'prior_reg': 0.001}: mode_hold — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | unavailable | >= 8 | UNAVAILABLE |
| hq | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_two_broad — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.15 | UNAVAILABLE |
| hq | unavailable | >= 0.85 | UNAVAILABLE |
| component_covariance_error | unavailable | <= 0.85 | UNAVAILABLE |
| component_min_eigen_ratio | unavailable | >= 0.15 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_unequal_mass — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.15 | UNAVAILABLE |
| hq | unavailable | >= 0.85 | UNAVAILABLE |
| component_covariance_error | unavailable | <= 0.85 | UNAVAILABLE |
| component_min_eigen_ratio | unavailable | >= 0.15 | UNAVAILABLE |
| min_mass_ratio | unavailable | >= 0.25 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_unequal_width — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.15 | UNAVAILABLE |
| hq | unavailable | >= 0.85 | UNAVAILABLE |
| component_covariance_error | unavailable | <= 0.85 | UNAVAILABLE |
| component_min_eigen_ratio | unavailable | >= 0.15 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_anisotropic — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.15 | UNAVAILABLE |
| hq | unavailable | >= 0.85 | UNAVAILABLE |
| component_covariance_error | unavailable | <= 0.85 | UNAVAILABLE |
| component_min_eigen_ratio | unavailable | >= 0.15 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_overlap — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mean_error | unavailable | <= 0.15 | UNAVAILABLE |
| covariance_error | unavailable | <= 0.45 | UNAVAILABLE |

## {'prior_reg': 0.001}: vector_spiral — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | unavailable | <= 0.18 | UNAVAILABLE |
| mean_error | unavailable | <= 0.15 | UNAVAILABLE |
| covariance_error | unavailable | <= 0.45 | UNAVAILABLE |

## {'prior_reg': 0.001}: img_stripes2 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | unavailable | >= 2 | UNAVAILABLE |
| hq | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: img_bars4 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | unavailable | >= 4 | UNAVAILABLE |
| hq | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: img_blobs4 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | unavailable | >= 4 | UNAVAILABLE |
| hq | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: img_intensity2 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | unavailable | >= 5 | UNAVAILABLE |
| passing scheduled observations | unavailable | No declared gate | UNAVAILABLE |
| complete declared execution | unavailable | == true | UNAVAILABLE |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | unavailable | >= 2 | UNAVAILABLE |
| hq | unavailable | >= 0.9 | UNAVAILABLE |

## {'prior_reg': 0.001}: grid100 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | unavailable | == true | UNAVAILABLE |
| terminal live accuracy passing checks | unavailable | == 5 | UNAVAILABLE |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent holdout coverage from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

## {'prior_reg': 0.001}: rotated100 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | unavailable | == true | UNAVAILABLE |
| terminal live accuracy passing checks | unavailable | == 5 | UNAVAILABLE |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent holdout coverage from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

## {'prior_reg': 0.001}: staggered100 — BLOCKED

tier 1: ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | unavailable | == true | UNAVAILABLE |
| terminal live accuracy passing checks | unavailable | == 5 | UNAVAILABLE |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 20000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| all_finite | unavailable | == true | UNAVAILABLE |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.06 | UNAVAILABLE |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | unavailable | == true | UNAVAILABLE |
| frozen_pass | unavailable | == true | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

Independent holdout coverage from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | unavailable | >= 20000 | UNAVAILABLE |
| valid_n | unavailable | == 100000 | UNAVAILABLE |
| precision | unavailable | >= 0.97 | UNAVAILABLE |
| modes | unavailable | == 100 | UNAVAILABLE |
| min_hq_mode_mass | unavailable | >= 0.005 | UNAVAILABLE |
| mass_tv | unavailable | <= 0.1 | UNAVAILABLE |
| max_mode_mass | unavailable | <= 0.02 | UNAVAILABLE |
| min_cov_eig_ratio | unavailable | >= 0.4 | UNAVAILABLE |
| max_cov_eig_ratio | unavailable | <= 1.7 | UNAVAILABLE |
| min_radial_median_ratio | unavailable | >= 0.65 | UNAVAILABLE |
| max_radial_median_ratio | unavailable | <= 1.4 | UNAVAILABLE |
| passed | unavailable | == true | UNAVAILABLE |

## {'prior_reg': 0.001}: clockfree_audit_measurement_v1 — PASS

declared state/horizon/cadence/restart comparisons agree; source audit bound

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| step_label: exact learning-state parity | true | == true | PASS |
| horizon: exact learning-state parity | true | == true | PASS |
| evaluation_cadence: exact learning-state parity | true | == true | PASS |
| restart: exact learning-state parity | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| parity_comparisons | 4 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.30185 | <= 0.2 | FAIL |
| std_ratio | 1.06616 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.134367 | <= 0.05 | FAIL |
| mean | 2.15093 | No declared gate | DIAGNOSTIC |
| std | 0.533082 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 542 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 542 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 2 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: two_pole — PASS

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

## {'prior_reg': 0.1}: unused_token_hold — PASS

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

## {'prior_reg': 0.1}: ae_gan_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 16 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00965968 | <= 0.05 | PASS |
| hold | 0.030768 | <= 0.35 | PASS |

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
| confirmation_fraction | 0.544 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 136 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 16 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 94 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0403403 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00965968 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.319232 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.030768 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: ring16_acquisition — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 35 | >= 5 | PASS |
| passing scheduled observations | 35 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0681152 | <= 0.15 | PASS |
| hq | 0.932617 | >= 0.85 | PASS |
| component_covariance_error | 0.54007 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.218786 | >= 0.15 | PASS |
| component_core_covariance_error | 0.463036 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.243269 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.907607 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.533903 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.399158 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.216432 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.281647 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.594802 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.716825 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.377187 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.434206 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.418549 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.326233 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.274448 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.657832 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.514901 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.511576 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.721534 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.470145 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.544067 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.492833 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.836574 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.702796 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.214802 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.518112 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.506222 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.443212 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.654046 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.729432 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.612864 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.650414 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.481694 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.516836 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.214802 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 268 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 220 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 187 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 223 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 355 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 251 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 303 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 238 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 306 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 255 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 267 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 222 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 303 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 269 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 207 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 222 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.342478 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.27414 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.799986 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.41673 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.225138 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.281647 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.617333 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 0.793958 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.356688 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.434206 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.444542 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.361504 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.253362 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.895887 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.608052 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.535467 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0654297 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0537109 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0456543 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0544434 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0866699 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0612793 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0739746 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0581055 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.074707 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0622559 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0651855 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0541992 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0739746 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0656738 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0505371 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0541992 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.0671642 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.168182 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.181818 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.058296 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0591549 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0517928 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0627063 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.105042 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0163399 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.027451 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0299625 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0585586 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0231023 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0966543 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.101449 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.0405405 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.065696 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 250 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 183 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 153 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 210 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 334 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 238 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 284 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 213 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 301 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 248 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 259 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 209 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 296 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 243 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 186 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 213 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.181818 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0351764 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.597656 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.730469 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0370284 | No declared gate | DIAGNOSTIC |
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
| confirmation_fraction | 0.6875 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1100 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 1034 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 35 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 35 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 1034 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0818848 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0681152 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0826172 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.932617 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.30993 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.54007 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.0687857 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.218786 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 24 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999989 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.198242 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 2.24114e-06 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 24 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 1667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[23] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 834 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 24 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: gaussian1d_stability — FAIL

stationary hold, deadline reacquisition and shifted hold

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | 2 | == 72 | FAIL |
| deadline reacquisition | false | == true | FAIL |
| shifted hold passing checks | 1 | == 24 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0319453 | <= 0.2 | PASS |
| std_ratio | 0.680289 | >= 0.8 and <= 1.2 | FAIL |
| cdf_ks | 0.130572 | <= 0.05 | FAIL |
| mean | 2.98403 | No declared gate | DIAGNOSTIC |
| std | 0.340144 | No declared gate | DIAGNOSTIC |
| step | 6000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| frozen_checks | 48 | No declared gate | DIAGNOSTIC |
| frozen_passes | 0 | No declared gate | DIAGNOSTIC |
| shift_hold_checks | 24 | No declared gate | DIAGNOSTIC |
| shift_hold_passes | 1 | No declared gate | DIAGNOSTIC |
| stationary_checks | 72 | No declared gate | DIAGNOSTIC |
| stationary_passes | 2 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: five_word_joint_hold — PASS

every joint generation/inverse check throughout the fixed continuation window

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passing primary checks | 25 | == 25 | PASS |
| confirmed checks | 25 | == 25 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0208984 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999995 | >= 0.9 | PASS |
| completed_steps | 4834 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.191406 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.193359 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.205078 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.21582 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.194336 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.14838e-06 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 4834 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 25 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 1001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 1168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 1334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 1501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 1668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 1834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 2001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 2168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 2334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 2668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 2834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 3001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 3168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 3501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 3668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 3834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 4001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 4168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 4334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 4501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[23] | 4668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[24] | 4834 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 834 | No declared gate | DIAGNOSTIC |
| observations | 25 | No declared gate | DIAGNOSTIC |
| passing_observations | 25 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: trajectory — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 18 | >= 5 | PASS |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000365531 | <= 0.02 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| critic_gradient_max | 0.23609 | No declared gate | DIAGNOSTIC |
| critic_gradient_median | 0.086614 | No declared gate | DIAGNOSTIC |
| own_nearest_fraction | 1 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000365531 | No declared gate | DIAGNOSTIC |
| particle_mean_square | 0.0150094 | No declared gate | DIAGNOSTIC |
| particle_std_mean | 0.127132 | No declared gate | DIAGNOSTIC |
| set_cover | 0.0058485 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.46 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 184 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 84 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 18 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 117 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0196345 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000365531 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: residual_student — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| identity_mse | 0.000211003 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| both_land_rows | 12 | No declared gate | DIAGNOSTIC |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| endpoint_l2 | 0.0299815 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| land_tol | 0.25 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000211003 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| pass_identity_mse | 0.02 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| residual_weight | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| slow_impact_max | 0.1 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |
| success_min | 1 | No declared gate | DIAGNOSTIC |
| vicreg_weight | 0.1 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 150 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 84 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 84 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.019789 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000211003 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 1 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: unipolar — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 18 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover | 0.9938 | >= 0.85 | PASS |
| off_caption | 6.61748e-05 | <= 0.05 | PASS |
| neu_hold | 0.989787 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cos_plus | 0.999908 | No declared gate | DIAGNOSTIC |
| cover_weight | 0 | No declared gate | DIAGNOSTIC |
| even_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| half_norm_ratio | 0.500457 | No declared gate | DIAGNOSTIC |
| hit | true | No declared gate | DIAGNOSTIC |
| mag_ratio | 0.993892 | No declared gate | DIAGNOSTIC |
| odd_norm | 0.496544 | No declared gate | DIAGNOSTIC |
| origin_norm | 0.0102133 | No declared gate | DIAGNOSTIC |
| polarity | 1 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 400 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.46 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 184 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 18 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 117 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.1438 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.9938 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0499338 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 6.61748e-05 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.139787 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.989787 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: cover_leftover — PASS

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
| u_kept | 0.980064 | >= 0.85 | PASS |
| content_kept | 0.971363 | >= 0.75 | PASS |
| leak_ratio | 0.02874 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0326644 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.023579 | <= 0.2 | PASS |
| same_dir | 0.012627 | <= 0.25 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| covered | true | No declared gate | DIAGNOSTIC |
| even_norm | 0.00819606 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| kappa | 1 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.999892 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| odd_norm | 1.09936 | No declared gate | DIAGNOSTIC |
| on_content | 0.525484 | No declared gate | DIAGNOSTIC |
| on_e | 0.00558371 | No declared gate | DIAGNOSTIC |
| on_u | 0.964045 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| particle_rms | 0.0657945 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.2925 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 234 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 22 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 22 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 100 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.130064 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.980064 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.221363 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.971363 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.17126 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.02874 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.167336 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.0326644 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.176421 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.023579 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.237373 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.012627 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: mid_scale_identity — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 20 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| concept_cos_plus | 0.999944 | >= 0.85 | PASS |
| concept_cos_minus | 0.999962 | >= 0.85 | PASS |
| concept_mag_plus | 1.00204 | >= 0.75 and <= 1.25 | PASS |
| concept_mag_minus | 1.00013 | >= 0.75 and <= 1.25 | PASS |
| identity_at_0 | 0.984797 | >= 0.85 | PASS |
| identity_at_mid | 0.992997 | >= 0.85 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.99985 | No declared gate | DIAGNOSTIC |
| pass | true | No declared gate | DIAGNOSTIC |
| reg_calls | 3200 | No declared gate | DIAGNOSTIC |
| reg_coeff | 1 | No declared gate | DIAGNOSTIC |
| reg_is_gradient_penalty | false | No declared gate | DIAGNOSTIC |
| reg_kappa | 1 | No declared gate | DIAGNOSTIC |
| reg_lazy | 1 | No declared gate | DIAGNOSTIC |
| same_dir | 0.00864692 | No declared gate | DIAGNOSTIC |
| seed | 0 | No declared gate | DIAGNOSTIC |
| steps | 800 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 300 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 67 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 20 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 167 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.149944 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.999944 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.149962 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.999962 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.25204 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.24796 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.00204 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.250134 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.249866 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 1.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 1.00013 | No declared gate | DIAGNOSTIC |
| metrics[6].margin | 0.134797 | No declared gate | DIAGNOSTIC |
| metrics[6].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[6].value | 0.984797 | No declared gate | DIAGNOSTIC |
| metrics[7].margin | 0.142997 | No declared gate | DIAGNOSTIC |
| metrics[7].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[7].value | 0.992997 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: mode_hold — FAIL

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
| modes | 6 | >= 8 | FAIL |
| hq | 0.901611 | >= 0.9 | PASS |
| cover | 0.75 | No declared gate | DIAGNOSTIC |
| effective_modes | 5.11401 | No declared gate | DIAGNOSTIC |
| n_modes | 8 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 8 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 6 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.00161133 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.901611 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.125 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_two_broad — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 3 | >= 5 | FAIL |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.175025 | <= 0.18 | PASS |
| mass_tv | 0.0605469 | <= 0.15 | PASS |
| hq | 0.991211 | >= 0.85 | PASS |
| component_covariance_error | 0.478607 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.374263 | >= 0.15 | PASS |
| component_core_covariance_error | 0.478607 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.44506 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.512154 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.374263 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.464946 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.374263 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2296 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1800 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.44506 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.512154 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.560547 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.439453 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00740418 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0105556 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.311362 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0105556 | No declared gate | DIAGNOSTIC |
| mean_error | 0.162559 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.878906 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.478607 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.374263 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0105556 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.5 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.5 | No declared gate | DIAGNOSTIC |
| target_scale | 1.05556 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 150 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 3 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.00497454 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.175025 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0894531 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0605469 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.141211 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.991211 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.371393 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.478607 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.224263 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.374263 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_unequal_mass — FAIL

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
| sw1_normalized | 0.163341 | <= 0.18 | PASS |
| mass_tv | 0.0712207 | <= 0.15 | PASS |
| hq | 0.964844 | >= 0.85 | PASS |
| component_covariance_error | 0.548615 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.0130056 | >= 0.15 | FAIL |
| min_mass_ratio | 0.770671 | >= 0.25 | PASS |
| component_core_covariance_error | 0.549254 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.512679 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.354297 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.418192 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.911851 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.413769 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.517862 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.43922 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.0130056 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.0130056 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2276 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 947 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 801 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 72 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.512679 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.351738 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.418192 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.911851 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.555664 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.231201 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.195557 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0175781 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0434974 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0475185 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.270039 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0475185 | No declared gate | DIAGNOSTIC |
| mean_error | 0.196076 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.428389 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.413769 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0475185 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.55 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.3 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.13 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.02 | No declared gate | DIAGNOSTIC |
| target_scale | 1.75933 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0166593 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.163341 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0787793 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0712207 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.114844 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.964844 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.301385 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.548615 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.136994 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0130056 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.520671 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.770671 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.152216 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_unequal_width — FAIL

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
| sw1_normalized | 0.278834 | <= 0.18 | FAIL |
| mass_tv | 0.281494 | <= 0.15 | FAIL |
| hq | 0.990967 | >= 0.85 | PASS |
| component_covariance_error | 0.382541 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.21351 | >= 0.15 | PASS |
| component_core_covariance_error | 0.382541 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.309832 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.579398 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.360973 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.279961 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.570848 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.21351 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.517925 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.675711 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.21351 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1517 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 259 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1684 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 636 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.309832 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.579398 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.360973 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.279961 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.370361 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0632324 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.411133 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.155273 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00593276 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0166271 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.254585 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0166271 | No declared gate | DIAGNOSTIC |
| mean_error | 0.40041 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.25293 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.382541 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.21351 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0166271 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.25 | No declared gate | DIAGNOSTIC |
| target_mass[3] | 0.25 | No declared gate | DIAGNOSTIC |
| target_scale | 2.1344 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0988336 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.278834 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.131494 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.281494 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.140967 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.990967 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.467459 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.382541 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.0635099 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.21351 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.285141 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_anisotropic — FAIL

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
| sw1_normalized | 0.250814 | <= 0.18 | FAIL |
| mass_tv | 0.237874 | <= 0.15 | FAIL |
| hq | 0.985596 | >= 0.85 | PASS |
| component_covariance_error | 1.06774 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.223326 | >= 0.15 | PASS |
| component_core_covariance_error | 0.167302 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.283758 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.0483772 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.169771 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.420146 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.223326 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.306799 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.223326 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 391 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 2146 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1559 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 2.71841 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.0483772 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.436427 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.095459 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.523926 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.380615 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0358056 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0288647 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.477001 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0358056 | No declared gate | DIAGNOSTIC |
| mean_error | 0.358422 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.286377 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.167302 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.223326 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0358056 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_mass[2] | 0.333333 | No declared gate | DIAGNOSTIC |
| target_scale | 2.03229 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0708139 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.250814 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.0878743 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.237874 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.135596 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.985596 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.217738 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.06774 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.0733264 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.223326 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.24708 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_overlap — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 5 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.207525 | <= 0.18 | FAIL |
| mean_error | 0.300184 | <= 0.15 | FAIL |
| covariance_error | 0.319299 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_scale | 0.850144 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 100 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 5 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -0.0275249 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.207525 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.150184 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.300184 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.130701 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.319299 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.384713 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_spiral — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 23 | >= 5 | PASS |
| passing scheduled observations | 23 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.0469044 | <= 0.18 | PASS |
| mean_error | 0.062042 | <= 0.15 | PASS |
| covariance_error | 0.0634385 | <= 0.45 | PASS |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| target_scale | 1.22418 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.25 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 400 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 134 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 23 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 23 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 134 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.133096 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0469044 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.087958 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.062042 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.386562 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0634385 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: img_stripes2 — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 12 | >= 5 | PASS |
| passing scheduled observations | 12 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 2 | >= 2 | PASS |
| hq | 1 | >= 0.9 | PASS |
| distribution_tv | 0.125 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.000286472 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.625 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.375 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.708333 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 425 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 325 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 12 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 12 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 325 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.1 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: img_bars4 — FAIL

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
| modes | 2 | >= 4 | FAIL |
| hq | 0.875 | >= 0.9 | FAIL |
| distribution_tv | 0.375 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0420645 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.625 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.125 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | -2 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.025 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.263889 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: img_blobs4 — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 2 | >= 5 | FAIL |
| passing scheduled observations | 2 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 4 | >= 4 | PASS |
| hq | 1 | >= 0.9 | PASS |
| distribution_tv | 0.15625 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.00283533 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.21875 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.125 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.21875 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[2] | 0.125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[3] | 0.375 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 575 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 2 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 2 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.1 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: img_intensity2 — FAIL

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
| modes | 2 | >= 2 | PASS |
| hq | 0.75 | >= 0.9 | FAIL |
| distribution_tv | 0 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0538732 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.5 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.5 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.34375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.40625 | No declared gate | DIAGNOSTIC |
| step | 600 | No declared gate | DIAGNOSTIC |

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
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 0 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 2 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 2 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.75 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0833333 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: grid100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2745 | >= 0.97 | FAIL |
| modes | 19 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.16515 | <= 0.1 | FAIL |
| max_mode_mass | 0.0268 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.84269 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.46395 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.2745 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2745 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.16515 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.714391 | No declared gate | DIAGNOSTIC |
| within_radius_n | 5490 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.27582 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.265095 | <= 0.1 | FAIL |
| center_rms_sigma | 1.55346 | <= 0.2 | FAIL |
| mass_tv | 0.16482 | <= 0.06 | FAIL |
| radial_ks | 0.419203 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.91133 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.44822 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.584463 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.265095 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.713071 | No declared gate | DIAGNOSTIC |
| within_radius_n | 27582 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.27582 | >= 0.97 | FAIL |
| modes | 16 | == 100 | FAIL |
| min_hq_mode_mass | 2e-05 | >= 0.005 | FAIL |
| mass_tv | 0.16482 | <= 0.1 | FAIL |
| max_mode_mass | 0.02737 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.23255 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.52115 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.34404 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.27582 | No declared gate | DIAGNOSTIC |
| min_hq_count | 2 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.27582 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.265095 | <= 0.1 | FAIL |
| center_rms_sigma | 1.55346 | <= 0.2 | FAIL |
| mass_tv | 0.16482 | <= 0.06 | FAIL |
| radial_ks | 0.419203 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.91133 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.44822 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.584463 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.265095 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.713071 | No declared gate | DIAGNOSTIC |
| within_radius_n | 27582 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279445 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417376 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151412 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113417 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279445 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: rotated100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.28315 | >= 0.97 | FAIL |
| modes | 15 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.13485 | <= 0.1 | FAIL |
| max_mode_mass | 0.02235 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.54995 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.3082 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28315 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2832 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.13485 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.705691 | No declared gate | DIAGNOSTIC |
| within_radius_n | 5664 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28306 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.13717 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.705831 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28306 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28306 | >= 0.97 | FAIL |
| modes | 14 | == 100 | FAIL |
| min_hq_mode_mass | 0 | >= 0.005 | FAIL |
| mass_tv | 0.13717 | <= 0.1 | FAIL |
| max_mode_mass | 0.02253 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.48584 | <= 1.7 | FAIL |
| min_radial_median_ratio | 0 | >= 0.65 | FAIL |
| max_radial_median_ratio | 2.26739 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.28306 | No declared gate | DIAGNOSTIC |
| min_hq_count | 0 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.28306 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.13717 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.705831 | No declared gate | DIAGNOSTIC |
| within_radius_n | 28306 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.00027944 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151327 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113412 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.100913 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.00027944 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: staggered100 — FAIL

coverage, sustained live accuracy, or independent holdout failed

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| coverage decision | false | == true | FAIL |
| terminal live accuracy passing checks | 0 | == 5 | FAIL |

Live endpoint coverage, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.3403 | >= 0.97 | FAIL |
| modes | 23 | == 100 | FAIL |
| min_hq_mode_mass | 5e-05 | >= 0.005 | FAIL |
| mass_tv | 0.1197 | <= 0.1 | FAIL |
| max_mode_mass | 0.02245 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.46725 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.16749 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.29964 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.3403 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.3403 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.1197 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.648591 | No declared gate | DIAGNOSTIC |
| within_radius_n | 6806 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34003 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.463829 | <= 0.1 | FAIL |
| center_rms_sigma | 1.20381 | <= 0.2 | FAIL |
| mass_tv | 0.11494 | <= 0.06 | FAIL |
| radial_ks | 0.381505 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.52766 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.30517 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.640839 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.463829 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.648861 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34003 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34 | >= 0.97 | FAIL |
| modes | 22 | == 100 | FAIL |
| min_hq_mode_mass | 0.00016 | >= 0.005 | FAIL |
| mass_tv | 0.11494 | <= 0.1 | FAIL |
| max_mode_mass | 0.02191 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.296312 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.24059 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.45364 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.31046 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.34 | No declared gate | DIAGNOSTIC |
| min_hq_count | 16 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passed | false | No declared gate | DIAGNOSTIC |
| terminal_summary.checks | 5 | No declared gate | DIAGNOSTIC |
| terminal_summary.passing_checks | 0 | No declared gate | DIAGNOSTIC |

EMA holdout diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34003 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.463829 | <= 0.1 | FAIL |
| center_rms_sigma | 1.20381 | <= 0.2 | FAIL |
| mass_tv | 0.11494 | <= 0.06 | FAIL |
| radial_ks | 0.381505 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.52766 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.30517 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.640839 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.463829 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.648861 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34003 | No declared gate | DIAGNOSTIC |

Oracle diagnostic cohort, 100000 saved draws (no qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.98914 | >= 0.97 | PASS |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.000279439 | <= 0.1 | PASS |
| center_rms_sigma | 0.0417374 | <= 0.2 | PASS |
| mass_tv | 0.01226 | <= 0.06 | PASS |
| radial_ks | 0.00151475 | <= 0.04 | PASS |
| accuracy_pass | true | == true | PASS |
| frozen_pass | true | == true | PASS |
| passed | true | == true | PASS |
| accuracy_score | 0.113421 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 0.10091 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.0561027 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.000279439 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.000248997 | No declared gate | DIAGNOSTIC |
| within_radius_n | 98914 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: clockfree_audit_measurement_v1 — PASS

declared state/horizon/cadence/restart comparisons agree; source audit bound

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| step_label: exact learning-state parity | true | == true | PASS |
| horizon: exact learning-state parity | true | == true | PASS |
| evaluation_cadence: exact learning-state parity | true | == true | PASS |
| restart: exact learning-state parity | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| parity_comparisons | 4 | No declared gate | DIAGNOSTIC |

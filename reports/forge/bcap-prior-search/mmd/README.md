# BCAP prior mmd search

Executed source `e85cabc484140cd8afa3c44ce2abe1b93daf0443`; protocol seed 0; ordinary revision 9.

Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.

No strength repaired a Tier 2 failure. Weight `.1` regressed mandatory word hold;
weights `1` and `10` were blocked before Tier 2. Keep leader defaults. See the
[conclusion, provenance and recommendation](interpretation.md),
[original sustained failure witnesses](failure-witnesses.md),
[saved-evidence audit](audit.json), and [all saved training GIFs](media/README.md).

| Strength | Tier 1 passes | Tier 2 passes | Repaired | Regressed | Unresolved control passes |
| --- | --- | --- | --- | --- | --- |
| Matched leader, prior_reg=0 | 6/6 | 9/21, all measured | — | — | none |
| {'prior_reg': 1.0} | 5/6 | Unmeasured: Tier 1 blocked | none | none measured | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |
| {'prior_reg': 0.1} | 6/6 | 8/21, all measured | none | five_word_joint_hold | none |
| {'prior_reg': 10.0} | 4/6 | Unmeasured: Tier 1 blocked | none | none measured | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |

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

## {'prior_reg': 1.0}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.261873 | <= 0.2 | FAIL |
| std_ratio | 1.22691 | >= 0.8 and <= 1.2 | FAIL |
| cdf_ks | 0.0996805 | <= 0.05 | FAIL |
| mean | 1.86906 | No declared gate | DIAGNOSTIC |
| std | 0.613455 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 209 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 542 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 2 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 1.0}: two_pole — PASS

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

## {'prior_reg': 1.0}: unused_token_hold — PASS

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

## {'prior_reg': 1.0}: ae_gan_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 6 | >= 5 | PASS |
| passing scheduled observations | 20 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00439154 | <= 0.05 | PASS |
| hold | 0.0171925 | <= 0.35 | PASS |

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
| confirmation_fraction | 0.96 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 240 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 20 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 6 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 198 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0456085 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00439154 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.332807 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0171925 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 1.0}: ring16_acquisition — FAIL

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
| mass_tv | 0.104492 | <= 0.15 | PASS |
| hq | 0.874756 | >= 0.85 | PASS |
| component_covariance_error | 1.4348 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.331045 | >= 0.15 | PASS |
| component_core_covariance_error | 0.64287 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.660634 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.919849 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.983433 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.596706 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.449783 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.181737 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.91666 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.859551 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.313032 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.541312 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.833612 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.34266 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.434832 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.25813 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.549813 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.444174 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 1.00887 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.852304 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.712973 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.534448 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.691206 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.745799 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.29683 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.445847 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.582703 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.564093 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.520115 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.523702 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.919807 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.742397 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.553957 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.876026 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.29683 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 287 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 178 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 208 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 243 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 368 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 251 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 183 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 197 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 362 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 349 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 202 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 168 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 292 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 286 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 276 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 3.46789 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 2.89267 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 2.50606 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 1.00293 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.588923 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.198731 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 1.67785 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.30743 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 3.41123 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.752052 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 1.25266 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 1.86 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.563963 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.35808 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.683097 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.433208 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0700684 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.043457 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0507812 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0593262 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0898438 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0612793 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0446777 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0480957 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0883789 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0852051 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0493164 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0410156 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0712891 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0698242 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0673828 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.170732 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.297753 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.25 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.17284 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.125 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0438247 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.240437 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.243655 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0745856 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0515759 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.10396 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.160714 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0821918 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0664336 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0650406 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.057971 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0438358 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 238 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 125 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 156 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 201 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 322 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 240 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 139 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 149 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 335 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 331 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 181 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 141 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 268 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 267 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 230 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 260 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.297753 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0280478 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.488281 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.65625 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0356759 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0455078 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.104492 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0247559 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.874756 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.584798 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.4348 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.181045 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.331045 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.114666 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 1.0}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 22 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999877 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.44049e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 22 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 2501 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 22 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 1.0}: gaussian1d_stability — BLOCKED

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

## {'prior_reg': 1.0}: five_word_joint_hold — BLOCKED

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

## {'prior_reg': 1.0}: trajectory — BLOCKED

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

## {'prior_reg': 1.0}: residual_student — BLOCKED

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

## {'prior_reg': 1.0}: unipolar — BLOCKED

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

## {'prior_reg': 1.0}: cover_leftover — BLOCKED

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

## {'prior_reg': 1.0}: mid_scale_identity — BLOCKED

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

## {'prior_reg': 1.0}: mode_hold — BLOCKED

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

## {'prior_reg': 1.0}: vector_two_broad — BLOCKED

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

## {'prior_reg': 1.0}: vector_unequal_mass — BLOCKED

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

## {'prior_reg': 1.0}: vector_unequal_width — BLOCKED

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

## {'prior_reg': 1.0}: vector_anisotropic — BLOCKED

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

## {'prior_reg': 1.0}: vector_overlap — BLOCKED

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

## {'prior_reg': 1.0}: vector_spiral — BLOCKED

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

## {'prior_reg': 1.0}: img_stripes2 — BLOCKED

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

## {'prior_reg': 1.0}: img_bars4 — BLOCKED

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

## {'prior_reg': 1.0}: img_blobs4 — BLOCKED

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

## {'prior_reg': 1.0}: img_intensity2 — BLOCKED

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

## {'prior_reg': 1.0}: grid100 — BLOCKED

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

## {'prior_reg': 1.0}: rotated100 — BLOCKED

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

## {'prior_reg': 1.0}: staggered100 — BLOCKED

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

## {'prior_reg': 1.0}: clockfree_audit_measurement_v1 — PASS

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
| mean_error_sigma | 0.812054 | <= 0.2 | FAIL |
| std_ratio | 1.01607 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.346762 | <= 0.05 | FAIL |
| mean | 2.40603 | No declared gate | DIAGNOSTIC |
| std | 0.508036 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 250 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 250 | No declared gate | DIAGNOSTIC |
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
| terminal passing suffix | 22 | >= 5 | PASS |
| passing scheduled observations | 22 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00404515 | <= 0.05 | PASS |
| hold | 0.00144375 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0459548 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00404515 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.348556 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.00144375 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: ring16_acquisition — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 11 | >= 5 | PASS |
| passing scheduled observations | 11 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.072998 | <= 0.15 | PASS |
| hq | 0.944092 | >= 0.85 | PASS |
| component_covariance_error | 0.546139 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.296102 | >= 0.15 | PASS |
| component_core_covariance_error | 0.469421 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.137621 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.875808 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.700678 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.431083 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.244481 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.457958 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.625743 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.899087 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.319462 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.379265 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.387373 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.494677 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.27616 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.270738 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.525017 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.485591 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.810197 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.39722 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.454551 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.519341 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.665352 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.416222 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.291078 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.542689 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.587418 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.748962 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.535615 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.736632 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.714667 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.620489 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.621798 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.517802 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.291078 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 301 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 244 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 212 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 232 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 367 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 280 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 227 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 212 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 333 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 244 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 235 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 172 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 255 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 274 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 228 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 280 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.153707 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.07418 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.765256 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.472615 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.543956 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.310021 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.653449 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.18485 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.319462 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.402686 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.495542 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.530119 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.335342 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.258702 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.713798 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.524542 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0734863 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0595703 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0517578 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0566406 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0895996 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0683594 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0554199 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0517578 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0812988 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0595703 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.057373 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0419922 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0622559 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0668945 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0556641 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0683594 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.089701 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.139344 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0377358 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0215517 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0572207 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0178571 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0220264 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.103774 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.012012 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.045082 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0510638 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0406977 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0431373 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0328467 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.105263 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.0857143 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0608724 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 274 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 210 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 204 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 227 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 346 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 275 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 222 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 329 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 233 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 223 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 165 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 244 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 265 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 204 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 256 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.139344 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0412623 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.644531 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.671875 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0364281 | No declared gate | DIAGNOSTIC |
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
| confirmation_fraction | 0.9375 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1500 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 1434 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 11 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 11 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 1434 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.077002 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.072998 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0940918 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.944092 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.303861 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.546139 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.146102 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.296102 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.999277 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 6.27753e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 23 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 1667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 1667 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 23 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: gaussian1d_stability — FAIL

stationary hold, deadline reacquisition and shifted hold

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | 3 | == 72 | FAIL |
| deadline reacquisition | false | == true | FAIL |
| shifted hold passing checks | 0 | == 24 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0630491 | <= 0.2 | PASS |
| std_ratio | 0.831685 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0733304 | <= 0.05 | FAIL |
| mean | 3.03152 | No declared gate | DIAGNOSTIC |
| std | 0.415842 | No declared gate | DIAGNOSTIC |
| step | 6000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| frozen_checks | 48 | No declared gate | DIAGNOSTIC |
| frozen_passes | 0 | No declared gate | DIAGNOSTIC |
| shift_hold_checks | 24 | No declared gate | DIAGNOSTIC |
| shift_hold_passes | 0 | No declared gate | DIAGNOSTIC |
| stationary_checks | 72 | No declared gate | DIAGNOSTIC |
| stationary_passes | 3 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: five_word_joint_hold — FAIL

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
| mass_tv | 0.0345703 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999956 | >= 0.9 | PASS |
| completed_steps | 5667 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.223633 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.194336 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.188477 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.182617 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.210938 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 5.50752e-06 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 5667 | No declared gate | DIAGNOSTIC |

Worst mandatory word-hold primary checks (saved failure witnesses)

1/25 mandatory checks failed numerical bounds. Original cohort SHA256 `9ba1f7f0e557eea5a27e2c18355d736c432170a1270ba6a1ee05bcdca0f81565`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 1667 | `4d7360baed48cef6fc18d0cc784a94707c15588d499a08f2ff012910eaf308bc` |
| quality_fraction | 1 | >= 0.95 | PASS | 1667 | `4d7360baed48cef6fc18d0cc784a94707c15588d499a08f2ff012910eaf308bc` |
| modes | 5 | == 5 | PASS | 1667 | `4d7360baed48cef6fc18d0cc784a94707c15588d499a08f2ff012910eaf308bc` |
| mass_tv | 0.0492188 | <= 0.1 | PASS | 4334 | `0abf951a438d6a6e5f376452c3dd07517260977041dd5c98a643eb1d17bd7191` |
| reconstruction_exact | 0 | == 1 | FAIL | 2001 | `e5f82f0556805e04bd37bc68f3f270ee8726f07b72689ebed7e63280f89d43fd` |
| minimum_reconstruction_token_probability | 0.19466 | >= 0.9 | FAIL | 2001 | `e5f82f0556805e04bd37bc68f3f270ee8726f07b72689ebed7e63280f89d43fd` |

Worst mandatory word-hold independent confirmation checks (saved failure witnesses)

1/25 mandatory checks failed numerical bounds. Original cohort SHA256 `63598f96863eda74d98a6c5bc1c743fa790796e11e6217ab5dfc85626162d637`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 1667 | `1308dcc988012d8d6edec230c2cf878a925570e208f3f44a20f9b4e02bb1325c` |
| quality_fraction | 1 | >= 0.95 | PASS | 1667 | `1308dcc988012d8d6edec230c2cf878a925570e208f3f44a20f9b4e02bb1325c` |
| modes | 5 | == 5 | PASS | 1667 | `1308dcc988012d8d6edec230c2cf878a925570e208f3f44a20f9b4e02bb1325c` |
| mass_tv | 0.0464844 | <= 0.1 | PASS | 5167 | `0c7c200c681f8617206d3f38899aecdeeb2da66c822ceb0b941035b086241e0b` |
| reconstruction_exact | 0 | == 1 | FAIL | 2001 | `e9f91a95fcfd8a34047cb355c7c8cf88d30fd3450b1cd0d772453f3949e89ef1` |
| minimum_reconstruction_token_probability | 0.19466 | >= 0.9 | FAIL | 2001 | `e9f91a95fcfd8a34047cb355c7c8cf88d30fd3450b1cd0d772453f3949e89ef1` |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 24 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 1667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 1834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 2167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 2334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 2667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 2834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 3001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 3167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 3501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 3667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 3834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 4001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 4334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 4501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 4667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 4834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 5167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 5334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 5501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[23] | 5667 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 1667 | No declared gate | DIAGNOSTIC |
| observations | 25 | No declared gate | DIAGNOSTIC |
| passing_observations | 24 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: trajectory — PASS

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
| identity_mse | 0.000241508 | <= 0.02 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| critic_gradient_max | 0.185154 | No declared gate | DIAGNOSTIC |
| critic_gradient_median | 0.0514865 | No declared gate | DIAGNOSTIC |
| own_nearest_fraction | 1 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000241508 | No declared gate | DIAGNOSTIC |
| particle_mean_square | 0.0106534 | No declared gate | DIAGNOSTIC |
| particle_std_mean | 0.106339 | No declared gate | DIAGNOSTIC |
| set_cover | 0.00386413 | No declared gate | DIAGNOSTIC |

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
| metrics[0].margin | 0.0197585 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000241508 | No declared gate | DIAGNOSTIC |
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
| identity_mse | 0.000202646 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| both_land_rows | 12 | No declared gate | DIAGNOSTIC |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| endpoint_l2 | 0.0311323 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| land_tol | 0.25 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000202646 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | 0.0197974 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000202646 | No declared gate | DIAGNOSTIC |
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
| u_kept | 1.02767 | >= 0.85 | PASS |
| content_kept | 1.05055 | >= 0.75 | PASS |
| leak_ratio | 0.0205896 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0297713 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0410239 | <= 0.2 | PASS |
| same_dir | 0.0146274 | <= 0.25 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| covered | true | No declared gate | DIAGNOSTIC |
| even_norm | 0.00596995 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| kappa | 1 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.999943 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| odd_norm | 1.11217 | No declared gate | DIAGNOSTIC |
| on_content | 0.54438 | No declared gate | DIAGNOSTIC |
| on_e | 0.0137316 | No declared gate | DIAGNOSTIC |
| on_u | 0.968992 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| particle_rms | 0.0662586 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | 0.177667 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 1.02767 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.300555 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 1.05055 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.17941 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0205896 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.170229 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.0297713 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.158976 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0410239 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.235373 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.0146274 | No declared gate | DIAGNOSTIC |
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
| passing scheduled observations | 4 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 8 | >= 8 | PASS |
| hq | 0.881836 | >= 0.9 | FAIL |
| cover | 1 | No declared gate | DIAGNOSTIC |
| effective_modes | 7.28092 | No declared gate | DIAGNOSTIC |
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
| convergence.first_pass_step | 900 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 4 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 8 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 8 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.0181641 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.881836 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0100911 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_two_broad — PASS

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
| sw1_normalized | 0.0993846 | <= 0.18 | PASS |
| mass_tv | 0.0209961 | <= 0.15 | PASS |
| hq | 0.998047 | >= 0.85 | PASS |
| component_covariance_error | 0.390363 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.431228 | >= 0.15 | PASS |
| component_core_covariance_error | 0.390363 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.378131 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.402594 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.520896 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.431228 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.431228 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2134 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1962 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.378131 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.402594 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.520996 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.479004 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.00407747 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0969057 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.00407747 | No declared gate | DIAGNOSTIC |
| mean_error | 0.119598 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.958008 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.390363 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.431228 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.00407747 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.5 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.5 | No declared gate | DIAGNOSTIC |
| target_scale | 1.05556 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.5 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 600 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 400 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 17 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 17 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 400 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0806154 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0993846 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.129004 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0209961 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.148047 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.998047 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.459637 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.390363 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.281228 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.431228 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_unequal_mass — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 1 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.121971 | <= 0.18 | PASS |
| mass_tv | 0.0792773 | <= 0.15 | PASS |
| hq | 0.990479 | >= 0.85 | PASS |
| component_covariance_error | 0.557519 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.0138622 | >= 0.15 | FAIL |
| min_mass_ratio | 0.748698 | >= 0.25 | PASS |
| component_core_covariance_error | 0.490575 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.304659 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.519836 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.197054 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.94075 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.570661 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.443383 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.762294 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.0138622 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.0138622 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2322 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 920 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 788 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 66 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.544767 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.519836 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.224721 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.94075 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.566895 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.224609 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.192383 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0161133 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00861326 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0241117 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.288776 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0241117 | No declared gate | DIAGNOSTIC |
| mean_error | 0.158585 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.340516 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.443383 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0241117 | No declared gate | DIAGNOSTIC |
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
| convergence.first_pass_step | 650 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 1 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0580294 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.121971 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0707227 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0792773 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.140479 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.990479 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.292481 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.557519 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.136138 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0138622 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.498698 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.748698 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.151264 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.264519 | <= 0.18 | FAIL |
| mass_tv | 0.258789 | <= 0.15 | FAIL |
| hq | 0.986328 | >= 0.85 | PASS |
| component_covariance_error | 7.86101 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.287994 | >= 0.15 | PASS |
| component_core_covariance_error | 0.612312 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.701377 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.773497 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.581005 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.393368 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.287994 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.150743 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.354462 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.507677 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.150743 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1468 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 252 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1640 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 736 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.701377 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 29.7683 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.581005 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.393368 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.358398 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0615234 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.400391 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.179688 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00817439 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.170635 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.000609756 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.269706 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.170635 | No declared gate | DIAGNOSTIC |
| mean_error | 0.355349 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.246094 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.612312 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.150743 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.170635 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | -0.0845188 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.264519 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.108789 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.258789 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.136328 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.986328 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -7.01101 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 7.86101 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.137994 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.287994 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.638962 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.274875 | <= 0.18 | FAIL |
| mass_tv | 0.264323 | <= 0.15 | FAIL |
| hq | 0.973389 | >= 0.85 | PASS |
| component_covariance_error | 2.6919 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.684525 | >= 0.15 | PASS |
| component_core_covariance_error | 0.160588 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.17579 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.0406148 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.26536 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.296171 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.70211 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.711313 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.296171 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1313 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 2448 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 335 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.181921 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.0603832 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 7.8334 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.320557 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.597656 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0817871 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0213252 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0106209 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.164179 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.457405 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.164179 | No declared gate | DIAGNOSTIC |
| mean_error | 0.371441 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.245361 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.160588 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.296171 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.164179 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | -0.094875 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.274875 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.114323 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.264323 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.123389 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.973389 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -1.8419 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 2.6919 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.534525 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.684525 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.657847 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: vector_overlap — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 0 | >= 5 | FAIL |
| passing scheduled observations | 4 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.109318 | <= 0.18 | PASS |
| mean_error | 0.15257 | <= 0.15 | FAIL |
| covariance_error | 0.074916 | <= 0.45 | PASS |
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
| convergence.passing_observations | 4 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 0 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0706819 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.109318 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.00256996 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.15257 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.375084 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.074916 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.00571103 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.0547795 | <= 0.18 | PASS |
| mean_error | 0.0783417 | <= 0.15 | PASS |
| covariance_error | 0.0485619 | <= 0.45 | PASS |
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
| metrics[0].margin | 0.125221 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0547795 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0716583 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0783417 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.401438 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0485619 | No declared gate | DIAGNOSTIC |
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
| distribution_tv | 0.09375 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.000151767 | No declared gate | DIAGNOSTIC |
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
| hq | 0.90625 | >= 0.9 | PASS |
| distribution_tv | 0.40625 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0186519 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.09375 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
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
| metrics[1].margin | 0.00625 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.90625 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.25 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: img_blobs4 — FAIL

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
| hq | 0.96875 | >= 0.9 | PASS |
| distribution_tv | 0.1875 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.00993652 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.21875 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
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
| metrics[1].margin | 0.06875 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.96875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.125 | No declared gate | DIAGNOSTIC |

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
| hq | 0.875 | >= 0.9 | FAIL |
| distribution_tv | 0 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.045735 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.5 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.5 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.46875 | No declared gate | DIAGNOSTIC |
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
| metrics[1].margin | -0.025 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0138889 | No declared gate | DIAGNOSTIC |

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
| precision | 0.2477 | >= 0.97 | FAIL |
| modes | 12 | == 100 | FAIL |
| min_hq_mode_mass | 5e-05 | >= 0.005 | FAIL |
| mass_tv | 0.1493 | <= 0.1 | FAIL |
| max_mode_mass | 0.0227 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.05298 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.1273 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.47669 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.2477 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2477 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.1493 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.741191 | No declared gate | DIAGNOSTIC |
| within_radius_n | 4954 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.24413 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.393986 | <= 0.1 | FAIL |
| center_rms_sigma | 1.45937 | <= 0.2 | FAIL |
| mass_tv | 0.14798 | <= 0.06 | FAIL |
| radial_ks | 0.454284 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.26504 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.50863 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.642949 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.393986 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.744761 | No declared gate | DIAGNOSTIC |
| within_radius_n | 24413 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.24409 | >= 0.97 | FAIL |
| modes | 11 | == 100 | FAIL |
| min_hq_mode_mass | 7e-05 | >= 0.005 | FAIL |
| mass_tv | 0.14798 | <= 0.1 | FAIL |
| max_mode_mass | 0.02196 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0252797 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.62537 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.50762 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.35915 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.24409 | No declared gate | DIAGNOSTIC |
| min_hq_count | 7 | No declared gate | DIAGNOSTIC |

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
| precision | 0.24413 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.393986 | <= 0.1 | FAIL |
| center_rms_sigma | 1.45937 | <= 0.2 | FAIL |
| mass_tv | 0.14798 | <= 0.06 | FAIL |
| radial_ks | 0.454284 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.26504 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.50863 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.642949 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.393986 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.744761 | No declared gate | DIAGNOSTIC |
| within_radius_n | 24413 | No declared gate | DIAGNOSTIC |

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
| precision | 0.23825 | >= 0.97 | FAIL |
| modes | 12 | == 100 | FAIL |
| min_hq_mode_mass | 5e-05 | >= 0.005 | FAIL |
| mass_tv | 0.13245 | <= 0.1 | FAIL |
| max_mode_mass | 0.02455 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.89194 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.37746 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.50343 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.23825 | No declared gate | DIAGNOSTIC |
| min_hq_count | 1 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2383 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | unavailable | <= 0.1 | UNAVAILABLE |
| center_rms_sigma | unavailable | <= 0.2 | UNAVAILABLE |
| mass_tv | 0.13245 | <= 0.06 | FAIL |
| radial_ks | unavailable | <= 0.04 | UNAVAILABLE |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | unavailable | No declared gate | UNAVAILABLE |
| center_max_sigma | unavailable | No declared gate | UNAVAILABLE |
| cov_frob_rms | unavailable | No declared gate | UNAVAILABLE |
| cov_trace_bias | unavailable | No declared gate | UNAVAILABLE |
| precision_gap | 0.750591 | No declared gate | DIAGNOSTIC |
| within_radius_n | 4766 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.23907 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.466392 | <= 0.1 | FAIL |
| center_rms_sigma | 1.39095 | <= 0.2 | FAIL |
| mass_tv | 0.13479 | <= 0.06 | FAIL |
| radial_ks | 0.464625 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.3702 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.62778 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.732202 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.466392 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.749821 | No declared gate | DIAGNOSTIC |
| within_radius_n | 23907 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.23905 | >= 0.97 | FAIL |
| modes | 12 | == 100 | FAIL |
| min_hq_mode_mass | 0.00017 | >= 0.005 | FAIL |
| mass_tv | 0.13479 | <= 0.1 | FAIL |
| max_mode_mass | 0.02337 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.02394 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.3305 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.47079 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.36499 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.23905 | No declared gate | DIAGNOSTIC |
| min_hq_count | 17 | No declared gate | DIAGNOSTIC |

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
| precision | 0.23907 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.466392 | <= 0.1 | FAIL |
| center_rms_sigma | 1.39095 | <= 0.2 | FAIL |
| mass_tv | 0.13479 | <= 0.06 | FAIL |
| radial_ks | 0.464625 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.3702 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.62778 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.732202 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.466392 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.749821 | No declared gate | DIAGNOSTIC |
| within_radius_n | 23907 | No declared gate | DIAGNOSTIC |

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
| precision | 0.34405 | >= 0.97 | FAIL |
| modes | 21 | == 100 | FAIL |
| min_hq_mode_mass | 0.0006 | >= 0.005 | FAIL |
| mass_tv | 0.1303 | <= 0.1 | FAIL |
| max_mode_mass | 0.02395 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.243241 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.01105 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.04604 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.32797 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.34405 | No declared gate | DIAGNOSTIC |
| min_hq_count | 12 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.344 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.448737 | <= 0.1 | FAIL |
| center_rms_sigma | 1.16117 | <= 0.2 | FAIL |
| mass_tv | 0.1303 | <= 0.06 | FAIL |
| radial_ks | 0.371038 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.43521 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.05208 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.687069 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.448737 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.644891 | No declared gate | DIAGNOSTIC |
| within_radius_n | 6880 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34425 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.53531 | <= 0.1 | FAIL |
| center_rms_sigma | 1.10354 | <= 0.2 | FAIL |
| mass_tv | 0.12414 | <= 0.06 | FAIL |
| radial_ks | 0.366605 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.52623 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 1.74823 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.701054 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.53531 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.644641 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34425 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34425 | >= 0.97 | FAIL |
| modes | 22 | == 100 | FAIL |
| min_hq_mode_mass | 0.00039 | >= 0.005 | FAIL |
| mass_tv | 0.12414 | <= 0.1 | FAIL |
| max_mode_mass | 0.02389 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.44043 | >= 0.4 | PASS |
| max_cov_eig_ratio | 2.47099 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.30261 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.0228 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.34425 | No declared gate | DIAGNOSTIC |
| min_hq_count | 39 | No declared gate | DIAGNOSTIC |

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
| precision | 0.34425 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.53531 | <= 0.1 | FAIL |
| center_rms_sigma | 1.10354 | <= 0.2 | FAIL |
| mass_tv | 0.12414 | <= 0.06 | FAIL |
| radial_ks | 0.366605 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.52623 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 1.74823 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.701054 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.53531 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.644641 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34425 | No declared gate | DIAGNOSTIC |

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

## {'prior_reg': 10.0}: gaussian1d_smoke — PASS

any scheduled full pass with independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 4 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.134615 | <= 0.2 | PASS |
| std_ratio | 0.913585 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0660693 | <= 0.05 | FAIL |
| mean | 1.93269 | No declared gate | DIAGNOSTIC |
| std | 0.456793 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 209 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 292 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 500 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 667 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 6 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 10.0}: two_pole — PASS

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

## {'prior_reg': 10.0}: unused_token_hold — PASS

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

## {'prior_reg': 10.0}: ae_gan_hold — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 3 | >= 5 | FAIL |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00511108 | <= 0.05 | PASS |
| hold | 0.026898 | <= 0.35 | PASS |

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
| confirmation_fraction | 2 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| convergence.first_pass_step | 42 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 3 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0448889 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00511108 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.323102 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.026898 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 10.0}: ring16_acquisition — FAIL

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
| mass_tv | 0.0297852 | <= 0.15 | PASS |
| hq | 0.762207 | >= 0.85 | FAIL |
| component_covariance_error | 5.33207 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.396108 | >= 0.15 | PASS |
| component_core_covariance_error | 1.38837 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 1.64728 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.70729 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 1.8375 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 1.84602 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 1.84914 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.919015 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 1.36287 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.19094 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.708903 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 1.39583 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 1.42806 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.18823 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 1.4349 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.752215 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.16729 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 1.77843 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 1.55986 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.765052 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.602608 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.801793 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.887041 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.825765 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.149206 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.34716 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.313628 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.44002 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.413784 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.845077 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.690672 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.864583 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.822449 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.987632 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.149206 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 263 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 235 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 254 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 249 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 250 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 275 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 285 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 268 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 291 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 255 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 223 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 236 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 271 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 261 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 234 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 22.0298 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 3.95697 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 3.08489 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 3.56303 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 3.37792 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 2.64456 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 2.15723 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 2.23843 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 1.15224 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 2.08132 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 2.7562 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 2.09454 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 2.63441 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 1.48002 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 2.08346 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 27.9781 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.064209 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.057373 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0620117 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.060791 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0610352 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0671387 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0695801 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0654297 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0710449 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0622559 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0544434 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0576172 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0661621 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0637207 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0571289 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.368821 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.297872 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.251969 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.284553 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.301205 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.28 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.2 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.294737 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.130597 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.189003 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.196078 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.174888 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.190678 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.140221 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.16092 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.363248 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0493635 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 166 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 165 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 176 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 174 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 180 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 220 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 201 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 233 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 236 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 205 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 184 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 191 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 233 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 219 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 149 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.368821 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0180297 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.582031 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.871094 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0285897 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.120215 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0297852 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.087793 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.762207 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -4.48207 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 5.33207 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.246108 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.396108 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.350548 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 10.0}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.995146 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 0.000169097 | No declared gate | DIAGNOSTIC |
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

## {'prior_reg': 10.0}: gaussian1d_stability — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: five_word_joint_hold — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: trajectory — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: residual_student — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: unipolar — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: cover_leftover — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: mid_scale_identity — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: mode_hold — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_two_broad — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_unequal_mass — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_unequal_width — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_anisotropic — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_overlap — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: vector_spiral — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: img_stripes2 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: img_bars4 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: img_blobs4 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: img_intensity2 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: grid100 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: rotated100 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: staggered100 — BLOCKED

tier 1: ae_gan_hold: FAIL recomputed complete live curve and terminal suffix; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 10.0}: clockfree_audit_measurement_v1 — PASS

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

Saved-only reproduction commands require the [mmd feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/e85cabc484140cd8afa3c44ce2abe1b93daf0443) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

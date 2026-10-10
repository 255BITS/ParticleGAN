# BCAP prior sigreg search

Executed source `a2a9183f042586968fe93daad1db5984c4621ddf`; protocol seed 0; ordinary revision 9.

Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.

| Strength | Tier 1 passes | Tier 2 measured | Repaired | Regressed | Unresolved control passes |
| --- | --- | --- | --- | --- | --- |
| {'prior_reg': 0.01} | 4/6 | BLOCKED (0/21 measured) | none measured | none measured | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |
| {'prior_reg': 0.0001} | 4/6 | BLOCKED (0/21 measured) | none measured | none measured | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |
| {'prior_reg': 0.001} | 5/6 | BLOCKED (0/21 measured) | none measured | none measured | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |

Every task metric and its frozen numerical gate are retained below and in [results.json](results.json). Unreached tasks remain unmeasured.

Empty Tier 2 repair/regression lists supply no evidence that the nine passing control tasks were preserved. The matched control passed 6/6 Tier 1 and 9/21 Tier 2. See the [interpretation](INTERPRETATION.md), [protocol audit](protocol-audit.json), and [49 saved actual-training GIFs with receipts](media/README.md). These links are report-only additions after the original executed source; the immutable certifications and numerical display cells remain intact.

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

## {'prior_reg': 0.01}: gaussian1d_smoke — FAIL

any scheduled full pass with independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 0 | >= 1 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.498709 | <= 0.2 | FAIL |
| std_ratio | 1.05062 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.234117 | <= 0.05 | FAIL |
| mean | 1.75065 | No declared gate | DIAGNOSTIC |
| std | 0.525312 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| first_confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 0 | No declared gate | DIAGNOSTIC |

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
| terminal passing suffix | 19 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00966191 | <= 0.05 | PASS |
| hold | 0.0587801 | <= 0.35 | PASS |

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
| confirmation_fraction | 0.42 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 105 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 19 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 63 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0403381 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00966191 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.29122 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0587801 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: ring16_acquisition — FAIL

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
| mass_tv | 0.0541992 | <= 0.15 | PASS |
| hq | 0.854492 | >= 0.85 | PASS |
| component_covariance_error | 1.80633 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.355756 | >= 0.15 | PASS |
| component_core_covariance_error | 0.99163 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 1.63546 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.15877 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.989701 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.859002 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 1.14313 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.511571 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 1.06866 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.30982 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.787715 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.305418 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.94883 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.523525 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.861878 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.599161 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.96362 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 1.19982 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.803423 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.923503 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.465486 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.388985 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.57472 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.35025 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.237493 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.842637 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.770401 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.800296 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.599439 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 1.00122 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 1.04126 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.68811 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.555059 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.609845 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.237493 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 258 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 274 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 215 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 223 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 322 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 268 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 229 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 284 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 278 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 296 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 235 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 238 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 222 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 290 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 238 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 226 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 3.01265 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 2.07328 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 1.6594 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 1.52836 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 1.72968 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.693306 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 1.89334 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 2.92301 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 1.78544 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.768746 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 1.63609 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 1.10757 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 1.68304 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 1.13308 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 3.13679 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 2.1375 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0629883 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0668945 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0524902 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0544434 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0786133 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0654297 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0559082 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0693359 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0678711 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0722656 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.057373 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0581055 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0541992 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0708008 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0581055 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0551758 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.232558 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.175182 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.125581 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.103139 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.127329 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0746269 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.117904 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.232394 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.219424 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0675676 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.13617 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0840336 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.157658 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0827586 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.239496 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.154867 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0308237 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 198 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 226 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 188 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 200 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 281 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 248 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 202 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 218 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 217 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 276 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 203 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 218 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 187 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 266 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 181 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 191 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.239496 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0022776 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.707031 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.839844 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0199274 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0958008 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0541992 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.00449219 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.854492 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.95633 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 1.80633 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.205756 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.355756 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.187516 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 21 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999977 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.198242 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.37092e-06 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 21 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 3334 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 21 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: gaussian1d_stability — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: five_word_joint_hold — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: trajectory — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: residual_student — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: unipolar — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: cover_leftover — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: mid_scale_identity — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: mode_hold — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_two_broad — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_unequal_mass — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_unequal_width — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_anisotropic — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_overlap — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: vector_spiral — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: img_stripes2 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: img_bars4 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: img_blobs4 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: img_intensity2 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: grid100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: rotated100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.01}: staggered100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: gaussian1d_smoke — FAIL

any scheduled full pass with independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 0 | >= 1 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.125682 | <= 0.2 | PASS |
| std_ratio | 1.07872 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0665886 | <= 0.05 | FAIL |
| mean | 1.93716 | No declared gate | DIAGNOSTIC |
| std | 0.53936 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| first_confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 2 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.0001}: two_pole — PASS

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

## {'prior_reg': 0.0001}: unused_token_hold — PASS

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

## {'prior_reg': 0.0001}: ae_gan_hold — PASS

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
| recon_mse | 0.0068059 | <= 0.05 | PASS |
| hold | 0.00683075 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0431941 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0068059 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.343169 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.00683075 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.0001}: ring16_acquisition — FAIL

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
| mass_tv | 0.0795898 | <= 0.15 | PASS |
| hq | 0.944336 | >= 0.85 | PASS |
| component_covariance_error | 2.12804 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.343701 | >= 0.15 | PASS |
| component_core_covariance_error | 0.537345 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.33487 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.851045 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.740472 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.60241 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.350552 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.378305 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.589131 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.00812 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.440598 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.439158 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.593064 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.851515 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.213251 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.367023 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.368556 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.469451 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.972205 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.444063 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.570203 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.7259 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.529335 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.551584 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.357686 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.333763 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.402181 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.410029 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.475571 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.815771 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.698718 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.536525 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.523114 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.595749 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.333763 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 320 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 215 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 186 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 256 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 351 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 266 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 279 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 243 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 314 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 302 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 188 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 206 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 235 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 251 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 286 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 198 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 8.51578 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.07109 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.846177 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.751139 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.366317 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.378305 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.60366 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.1281 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.421179 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.42753 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.593064 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.977512 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.213251 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.406 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.385632 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 16.9639 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.078125 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0524902 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0454102 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0625 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0856934 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0649414 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0681152 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0593262 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0766602 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0737305 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0458984 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.050293 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.057373 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0612793 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0698242 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0483398 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.065625 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0837209 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0645161 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.117188 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.045584 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0225564 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0430108 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.090535 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0541401 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0298013 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.037234 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.101942 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.012766 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0318725 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.020979 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.10101 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.06956 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 299 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 197 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 174 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 226 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 335 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 260 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 267 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 221 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 297 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 293 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 181 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 185 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 232 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 243 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 280 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 178 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.117188 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0422896 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.679688 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.726562 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0389587 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0704102 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0795898 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0943359 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.944336 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -1.27804 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 2.12804 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.193701 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.343701 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.250596 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.0001}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 14 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.995024 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.198242 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 0.000489143 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 14 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 9168 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 14 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.0001}: gaussian1d_stability — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: five_word_joint_hold — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: trajectory — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: residual_student — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: unipolar — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: cover_leftover — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: mid_scale_identity — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: mode_hold — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_two_broad — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_unequal_mass — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_unequal_width — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_anisotropic — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_overlap — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: vector_spiral — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: img_stripes2 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: img_bars4 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: img_blobs4 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: img_intensity2 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: grid100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: rotated100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: staggered100 — BLOCKED

tier 1: gaussian1d_smoke: FAIL any scheduled full pass with independent same-state confirmation; ring16_acquisition: FAIL recomputed complete live curve and terminal suffix

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

## {'prior_reg': 0.0001}: clockfree_audit_measurement_v1 — PASS

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
| independently confirmed scheduled states | 4 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.132182 | <= 0.2 | PASS |
| std_ratio | 1.00929 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0780312 | <= 0.05 | FAIL |
| mean | 1.93391 | No declared gate | DIAGNOSTIC |
| std | 0.504643 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 125 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 209 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 709 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 834 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 125 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 5 | No declared gate | DIAGNOSTIC |

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
| recon_mse | 0.00535378 | <= 0.05 | PASS |
| hold | 0.0156361 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0446462 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00535378 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.334364 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0156361 | No declared gate | DIAGNOSTIC |
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
| mass_tv | 0.0695801 | <= 0.15 | PASS |
| hq | 0.891113 | >= 0.85 | PASS |
| component_covariance_error | 0.884537 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.363003 | >= 0.15 | PASS |
| component_core_covariance_error | 0.634048 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.657676 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.07793 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.586057 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.649955 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.0218171 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.234365 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.868779 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.29526 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.310594 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.545709 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.789935 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.659689 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.365582 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.254966 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.911335 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.91512 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.43638 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.425439 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.399207 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.768297 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.969304 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.929046 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.349955 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.430482 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.561365 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.541491 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.607933 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.881077 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.913936 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.805118 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.595068 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.891299 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.349955 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 298 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 223 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 229 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 260 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 390 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 305 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 226 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 247 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 286 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 244 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 253 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 152 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 282 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 231 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 224 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.724793 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.71052 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.856013 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.853574 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.0526671 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.264981 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 1.31309 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 2.07628 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.325362 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.618563 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 1.12977 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.676742 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.587058 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.269107 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 1.45268 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 1.24138 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0727539 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0544434 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0559082 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0634766 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0952148 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0744629 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0551758 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0603027 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0698242 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0595703 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0617676 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0371094 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0688477 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0563965 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0546875 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.0604027 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.273543 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.113537 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.130769 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0692308 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0491803 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.150442 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.238866 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.034965 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0737705 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0909091 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0657895 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.102837 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0519481 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.134146 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.165179 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0476466 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 280 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 162 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 203 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 226 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 363 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 290 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 192 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 188 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 276 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 226 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 230 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 142 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 253 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 219 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 213 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 187 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.273543 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0456654 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.554688 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.59375 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0428751 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0804199 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0695801 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0411133 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.891113 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.0345368 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.884537 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.213003 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.363003 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.00677192 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.99961 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.208008 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 2.57862e-05 | No declared gate | DIAGNOSTIC |
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

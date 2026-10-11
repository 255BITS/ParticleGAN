# BCAP prior covariance search

[Readout and recommendation](readout.md): global strength 0.01 repairs
`mode_hold` but regresses word hold; 0.001 and 0.1 lose the ring prerequisite.
See the [saved-state audit](audit.json), [original repair/regression witnesses](repair-witness.json),
[full-location moment diagnostics](moment-diagnostics.md) and
[all 70 actual-training GIFs](media/README.md). Defaults remain opt-in.

Executed source `18e2aa1018f25de8c447a59287025df6f6aded3e`; protocol seed 0; ordinary revision 9.

Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.

| Strength | Tier 1 passes | Tier 2 passes | Repaired | Regressed | Unresolved control passes |
| --- | --- | --- | --- | --- | --- |
| {'prior_reg': 0.1} | 5/6 | 0/21 | none | none | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |
| {'prior_reg': 0.01} | 6/6 | 9/21 | mode_hold | five_word_joint_hold | none |
| {'prior_reg': 0.001} | 5/6 | 0/21 | none | none | five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2 |

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
| mean_error_sigma | 0.213004 | <= 0.2 | FAIL |
| std_ratio | 0.841545 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.109906 | <= 0.05 | FAIL |
| mean | 1.8935 | No declared gate | DIAGNOSTIC |
| std | 0.420773 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 875 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 875 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 1 | No declared gate | DIAGNOSTIC |

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
| recon_mse | 0.00558663 | <= 0.05 | PASS |
| hold | 0.0349411 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0444134 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00558663 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.315059 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0349411 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.1}: ring16_acquisition — FAIL

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
| mass_tv | 0.0761719 | <= 0.15 | PASS |
| hq | 0.916992 | >= 0.85 | PASS |
| component_covariance_error | 0.945122 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.402359 | >= 0.15 | PASS |
| component_core_covariance_error | 0.712761 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.469138 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.896543 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 1.10855 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.662567 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.211525 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.193372 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.641989 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.41186 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.424886 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.570128 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.813533 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.05634 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.41154 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.626633 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.03567 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.869906 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.660524 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.464107 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.457303 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.508711 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.7961 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 1.17074 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.395972 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.816883 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.613438 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.520716 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.576056 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.957008 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.908101 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.726489 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.666076 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 1.03913 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.395972 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 309 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 233 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 203 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 264 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 351 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 335 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 239 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 212 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 292 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 259 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 288 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 189 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 262 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 242 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 198 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 220 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.556562 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.29747 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 1.46535 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.689403 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.283222 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.193372 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.718818 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.98983 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.656086 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.599367 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 1.13815 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 1.67303 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.458273 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.840108 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 1.39104 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 1.17188 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0754395 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0568848 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0495605 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0644531 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0856934 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0817871 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0583496 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0517578 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0712891 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0632324 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0703125 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0461426 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0639648 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.059082 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0483398 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0537109 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.0550162 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.137339 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.167488 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0492424 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.02849 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0328358 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0460251 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.179245 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0650685 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0694981 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0798611 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.142857 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0572519 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.053719 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.136364 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.145455 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0497765 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 292 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 201 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 169 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 251 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 341 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 324 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 228 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 174 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 273 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 241 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 265 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 162 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 247 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 229 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 171 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 188 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.179245 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0280912 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.632812 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.738281 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0325975 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0738281 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0761719 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0669922 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.916992 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.0951225 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.945122 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.252359 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.402359 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.0186515 | No declared gate | DIAGNOSTIC |

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
| minimum_reconstruction_token_probability | 0.999868 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.32924e-05 | No declared gate | DIAGNOSTIC |
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

## {'prior_reg': 0.1}: gaussian1d_stability — BLOCKED

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

## {'prior_reg': 0.1}: five_word_joint_hold — BLOCKED

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

## {'prior_reg': 0.1}: trajectory — BLOCKED

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

## {'prior_reg': 0.1}: residual_student — BLOCKED

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

## {'prior_reg': 0.1}: unipolar — BLOCKED

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

## {'prior_reg': 0.1}: cover_leftover — BLOCKED

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

## {'prior_reg': 0.1}: mid_scale_identity — BLOCKED

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

## {'prior_reg': 0.1}: mode_hold — BLOCKED

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

## {'prior_reg': 0.1}: vector_two_broad — BLOCKED

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

## {'prior_reg': 0.1}: vector_unequal_mass — BLOCKED

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

## {'prior_reg': 0.1}: vector_unequal_width — BLOCKED

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

## {'prior_reg': 0.1}: vector_anisotropic — BLOCKED

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

## {'prior_reg': 0.1}: vector_overlap — BLOCKED

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

## {'prior_reg': 0.1}: vector_spiral — BLOCKED

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

## {'prior_reg': 0.1}: img_stripes2 — BLOCKED

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

## {'prior_reg': 0.1}: img_bars4 — BLOCKED

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

## {'prior_reg': 0.1}: img_blobs4 — BLOCKED

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

## {'prior_reg': 0.1}: img_intensity2 — BLOCKED

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

## {'prior_reg': 0.1}: grid100 — BLOCKED

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

## {'prior_reg': 0.1}: rotated100 — BLOCKED

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

## {'prior_reg': 0.1}: staggered100 — BLOCKED

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

## {'prior_reg': 0.01}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.702792 | <= 0.2 | FAIL |
| std_ratio | 1.01379 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.330528 | <= 0.05 | FAIL |
| mean | 2.3514 | No declared gate | DIAGNOSTIC |
| std | 0.506893 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 209 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 459 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 542 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 959 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 4 | No declared gate | DIAGNOSTIC |

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
| recon_mse | 0.00444648 | <= 0.05 | PASS |
| hold | 0.0284711 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0455535 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00444648 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.321529 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0284711 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: ring16_acquisition — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 9 | >= 5 | PASS |
| passing scheduled observations | 10 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0905762 | <= 0.15 | PASS |
| hq | 0.947021 | >= 0.85 | PASS |
| component_covariance_error | 0.726276 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.325237 | >= 0.15 | PASS |
| component_core_covariance_error | 0.533935 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.749843 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.974269 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.721399 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.664339 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.345945 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.354546 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.681086 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.860036 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.306585 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.37224 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.538902 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.707595 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.272901 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.291045 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.345321 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.356903 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.450135 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.464731 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.610467 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.565859 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.614253 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.521216 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.324326 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.462896 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.56949 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.473884 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.45078 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.473722 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.618748 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.618107 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.511686 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.638079 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.324326 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 279 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 193 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 202 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 182 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 380 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 259 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 222 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 251 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 327 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 217 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 206 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 204 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 313 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 276 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 287 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 298 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.925564 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 1.43096 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.78147 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.927202 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 2.40011 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.276544 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.724697 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 0.929827 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.306585 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.37224 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.538902 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.707595 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.265231 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.2738 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.342384 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 0.417301 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0681152 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0471191 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0493164 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0444336 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0927734 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0632324 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0541992 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0612793 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.079834 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0529785 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.050293 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0498047 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.076416 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0673828 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0700684 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0727539 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.114695 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.150259 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0792079 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0879121 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0578947 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.023166 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0405405 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.0796813 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0152905 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0138249 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0242718 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.0441176 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0223642 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0362319 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0557491 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.0402685 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.106599 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 247 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 164 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 186 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 166 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 358 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 253 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 213 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 231 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 322 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 214 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 201 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 195 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 306 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 266 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 271 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 286 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.150259 | No declared gate | DIAGNOSTIC |
| mean_error | 0.00824019 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.640625 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.710938 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0385957 | No declared gate | DIAGNOSTIC |
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
| confirmation_fraction | 0.95875 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1534 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 1434 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 96 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 10 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 9 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 1467 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4096 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 4096 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 16 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 16 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.0594238 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0905762 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0970215 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.947021 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.123724 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.726276 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.175237 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.325237 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 20 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999481 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.91555e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 20 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 10834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 13334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 15001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 16668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 4167 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 20 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: gaussian1d_stability — FAIL

stationary hold, deadline reacquisition and shifted hold

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| stationary passing checks | 0 | == 72 | FAIL |
| deadline reacquisition | false | == true | FAIL |
| shifted hold passing checks | 0 | == 24 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.117416 | <= 0.2 | PASS |
| std_ratio | 1.22242 | >= 0.8 and <= 1.2 | FAIL |
| cdf_ks | 0.13339 | <= 0.05 | FAIL |
| mean | 3.05871 | No declared gate | DIAGNOSTIC |
| std | 0.611208 | No declared gate | DIAGNOSTIC |
| step | 6000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| frozen_checks | 48 | No declared gate | DIAGNOSTIC |
| frozen_passes | 0 | No declared gate | DIAGNOSTIC |
| shift_hold_checks | 24 | No declared gate | DIAGNOSTIC |
| shift_hold_passes | 0 | No declared gate | DIAGNOSTIC |
| stationary_checks | 72 | No declared gate | DIAGNOSTIC |
| stationary_passes | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: five_word_joint_hold — FAIL

every joint generation/inverse check throughout the fixed continuation window

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| passing primary checks | 23 | == 25 | FAIL |
| confirmed checks | 23 | == 25 | FAIL |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0226563 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999655 | >= 0.9 | PASS |
| completed_steps | 8167 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.193359 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.199219 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.189453 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.222656 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 2.20187e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 8167 | No declared gate | DIAGNOSTIC |

Worst mandatory word-hold primary checks (saved failure witnesses)

2/25 mandatory checks failed numerical bounds. Original cohort SHA256 `83574b3a663534b0f489ab31a2dea875280a66ef6024b0a8d4e4e8e3ccf6ac35`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 4167 | `f6c7a4db292f846fb5c65a38bdacee5c31ea8a625a5afd77032c9b787e7fb6bb` |
| quality_fraction | 1 | >= 0.95 | PASS | 4167 | `f6c7a4db292f846fb5c65a38bdacee5c31ea8a625a5afd77032c9b787e7fb6bb` |
| modes | 4 | == 5 | FAIL | 5167 | `16738db111e18a8bba56934bdfdec90cdbfbf732f62cf00c9ce49fbf051d92f9` |
| mass_tv | 0.203516 | <= 0.1 | FAIL | 5167 | `16738db111e18a8bba56934bdfdec90cdbfbf732f62cf00c9ce49fbf051d92f9` |
| reconstruction_exact | 1 | == 1 | PASS | 4167 | `f6c7a4db292f846fb5c65a38bdacee5c31ea8a625a5afd77032c9b787e7fb6bb` |
| minimum_reconstruction_token_probability | 0.796768 | >= 0.9 | FAIL | 6167 | `f6bcfd6c751e57833caae9cb12d63f474abb25ecd89a3b75f23353ad79612290` |

Worst mandatory word-hold independent confirmation checks (saved failure witnesses)

2/25 mandatory checks failed numerical bounds. Original cohort SHA256 `ac483f0be2f4ea99ab972e93c41e5a4b609d90465d7c8d6438113715065cbf5b`; no observations rescored for qualification.

| Measurement | Score | Target | Status | Witness step | Original observation SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| sample_count | 1024 | >= 1024 | PASS | 4167 | `45171b2d4076a2b8efebd0a767cec6c2c5a97d0a753f6b86e09a576f20df314b` |
| quality_fraction | 1 | >= 0.95 | PASS | 4167 | `45171b2d4076a2b8efebd0a767cec6c2c5a97d0a753f6b86e09a576f20df314b` |
| modes | 4 | == 5 | FAIL | 5167 | `cc6814fad481499f9fd7992e4f8d9c9181cef268fba5ad82b867941308124df9` |
| mass_tv | 0.204687 | <= 0.1 | FAIL | 5167 | `cc6814fad481499f9fd7992e4f8d9c9181cef268fba5ad82b867941308124df9` |
| reconstruction_exact | 1 | == 1 | PASS | 4167 | `45171b2d4076a2b8efebd0a767cec6c2c5a97d0a753f6b86e09a576f20df314b` |
| minimum_reconstruction_token_probability | 0.796768 | >= 0.9 | FAIL | 6167 | `4aa6fd3ad033aead3e9e81f0a3773891782765d958a0a68af8b5268d05cdee90` |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 23 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 4167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 4334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 4501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 4667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 4834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 5334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 5501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 5667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 6001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 6334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 6501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 6834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 7001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[16] | 7167 | No declared gate | DIAGNOSTIC |
| confirmed_steps[17] | 7334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[18] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[19] | 7667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 7834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 8001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[22] | 8167 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 4167 | No declared gate | DIAGNOSTIC |
| observations | 25 | No declared gate | DIAGNOSTIC |
| passing_observations | 23 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: trajectory — PASS

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
| identity_mse | 0.000288721 | <= 0.02 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| critic_gradient_max | 0.149001 | No declared gate | DIAGNOSTIC |
| critic_gradient_median | 0.0561815 | No declared gate | DIAGNOSTIC |
| own_nearest_fraction | 1 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000288721 | No declared gate | DIAGNOSTIC |
| particle_mean_square | 0.00988272 | No declared gate | DIAGNOSTIC |
| particle_std_mean | 0.10272 | No declared gate | DIAGNOSTIC |
| set_cover | 0.00461953 | No declared gate | DIAGNOSTIC |

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
| metrics[0].margin | 0.0197113 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000288721 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: residual_student — PASS

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
| identity_mse | 0.000262156 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| both_land_rows | 12 | No declared gate | DIAGNOSTIC |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| endpoint_l2 | 0.032518 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| land_tol | 0.25 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| paired_target_mse | 0.000262156 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | 0.0197378 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.02 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.000262156 | No declared gate | DIAGNOSTIC |
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
| u_kept | 1.02279 | >= 0.85 | PASS |
| content_kept | 0.962276 | >= 0.75 | PASS |
| leak_ratio | 0.0261118 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0268752 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0583194 | <= 0.2 | PASS |
| same_dir | 0.0438744 | <= 0.25 | PASS |

Additional scalar endpoint diagnostics (no declared gate)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| cover_weight | 1.5 | No declared gate | DIAGNOSTIC |
| covered | true | No declared gate | DIAGNOSTIC |
| even_norm | 0.00746989 | No declared gate | DIAGNOSTIC |
| fm_weight | 0 | No declared gate | DIAGNOSTIC |
| kappa | 1 | No declared gate | DIAGNOSTIC |
| leak_frac | -0.99993 | No declared gate | DIAGNOSTIC |
| n_particles | 12 | No declared gate | DIAGNOSTIC |
| odd_norm | 1.08967 | No declared gate | DIAGNOSTIC |
| on_content | 0.519241 | No declared gate | DIAGNOSTIC |
| on_e | 0.00556048 | No declared gate | DIAGNOSTIC |
| on_u | 0.953676 | No declared gate | DIAGNOSTIC |
| particle_l2 | 0 | No declared gate | DIAGNOSTIC |
| particle_rms | 0.0523514 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | 0.172788 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 1.02279 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.212276 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.75 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.962276 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.173888 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0261118 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.173125 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.0268752 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.141681 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.2 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.0583194 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.206126 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.0438744 | No declared gate | DIAGNOSTIC |
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

## {'prior_reg': 0.01}: mode_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 7 | >= 5 | PASS |
| passing scheduled observations | 10 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 8 | >= 8 | PASS |
| hq | 0.999023 | >= 0.9 | PASS |
| cover | 1 | No declared gate | DIAGNOSTIC |
| effective_modes | 7.27086 | No declared gate | DIAGNOSTIC |
| n_modes | 8 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.916667 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1100 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 700 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 10 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 7 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 900 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 8 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 8 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0990234 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.999023 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_two_broad — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 8 | >= 5 | PASS |
| passing scheduled observations | 21 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.0989545 | <= 0.18 | PASS |
| mass_tv | 0.0581055 | <= 0.15 | PASS |
| hq | 0.987549 | >= 0.85 | PASS |
| component_covariance_error | 0.298793 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.445288 | >= 0.15 | PASS |
| component_core_covariance_error | 0.298793 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.394678 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.202909 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.445288 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.73929 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.445288 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2286 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 1810 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.394678 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.202909 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.558105 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.441895 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0223097 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.23627 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0223097 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0689523 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.883789 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.298793 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.445288 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0223097 | No declared gate | DIAGNOSTIC |
| sample_count | 4096 | No declared gate | DIAGNOSTIC |
| step | 1200 | No declared gate | DIAGNOSTIC |
| target_mass[0] | 0.5 | No declared gate | DIAGNOSTIC |
| target_mass[1] | 0.5 | No declared gate | DIAGNOSTIC |
| target_scale | 1.05556 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| attempted | true | No declared gate | DIAGNOSTIC |
| confirmation_fraction | 0.875 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 1050 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 150 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 21 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 8 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 850 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.0810455 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0989545 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0918945 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0581055 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.137549 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.987549 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.551207 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.298793 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.295288 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.445288 | No declared gate | DIAGNOSTIC |
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
| sw1_normalized | 0.176138 | <= 0.18 | PASS |
| mass_tv | 0.0870898 | <= 0.15 | PASS |
| hq | 0.953857 | >= 0.85 | PASS |
| component_covariance_error | 4.10357 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.00366343 | >= 0.15 | FAIL |
| min_mass_ratio | 0.32959 | >= 0.25 | PASS |
| component_core_covariance_error | 0.684383 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.835885 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.414653 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.556626 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.930369 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.777524 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.413882 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.323844 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.00366343 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.00366343 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 2318 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 927 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 824 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 27 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 1.73305 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.414653 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 13.3362 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.930369 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.565918 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.226318 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.201172 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0065918 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | false | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.0353753 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0463862 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0728155 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.148148 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.382766 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.148148 | No declared gate | DIAGNOSTIC |
| mean_error | 0.248616 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.602388 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.323844 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0728155 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | 0.00386206 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.176138 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0629102 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0870898 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.103857 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.953857 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -3.25357 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 4.10357 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.146337 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.00366343 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.0795898 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.25 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.32959 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.49593 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.308444 | <= 0.18 | FAIL |
| mass_tv | 0.300049 | <= 0.15 | FAIL |
| hq | 0.987061 | >= 0.85 | PASS |
| component_covariance_error | 6.73499 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.108822 | >= 0.15 | FAIL |
| component_core_covariance_error | 0.590539 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.789673 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.775007 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.44032 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.357155 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.108822 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.190284 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.55267 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.593769 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.108822 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1505 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 253 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 1772 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 566 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.789673 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 25.3528 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.44032 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.357155 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.367432 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0617676 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.432617 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.138184 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_resolved[3] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.00863787 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.0671937 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0129797 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.247582 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0671937 | No declared gate | DIAGNOSTIC |
| mean_error | 0.437655 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.24707 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.590539 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.108822 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0671937 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | -0.128444 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.308444 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.150049 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.300049 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.137061 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.987061 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -5.88499 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 6.73499 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -0.0411784 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.108822 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.797685 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.291984 | <= 0.18 | FAIL |
| mass_tv | 0.299967 | <= 0.15 | FAIL |
| hq | 0.980957 | >= 0.85 | PASS |
| component_covariance_error | 1.04532 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.334534 | >= 0.15 | PASS |
| component_core_covariance_error | 0.305221 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.373744 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.231104 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.310814 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.415391 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.279583 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.186672 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.186672 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 1177 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 2594 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 325 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 0.804604 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 0.228224 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 2.10314 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.287354 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.633301 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0793457 | No declared gate | DIAGNOSTIC |
| component_resolved[0] | true | No declared gate | DIAGNOSTIC |
| component_resolved[1] | true | No declared gate | DIAGNOSTIC |
| component_resolved[2] | true | No declared gate | DIAGNOSTIC |
| component_spill[0] | 0.039932 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.00424056 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.0615385 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.522375 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.0615385 | No declared gate | DIAGNOSTIC |
| mean_error | 0.37391 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.238037 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0.305221 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 0.186672 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0.0615385 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | -0.111984 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.291984 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.149967 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.299967 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.130957 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.980957 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.195322 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 1.04532 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | 0.184534 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 0.334534 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.370341 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.01}: vector_overlap — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 1 | >= 5 | FAIL |
| passing scheduled observations | 11 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.0899424 | <= 0.18 | PASS |
| mean_error | 0.0959597 | <= 0.15 | PASS |
| covariance_error | 0.302039 | <= 0.45 | PASS |
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
| convergence.passing_observations | 11 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 1 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0900576 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0899424 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.0540403 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0959597 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.147961 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.302039 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

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
| sw1_normalized | 0.0367942 | <= 0.18 | PASS |
| mean_error | 0.0281733 | <= 0.15 | PASS |
| covariance_error | 0.0648453 | <= 0.45 | PASS |
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
| metrics[0].margin | 0.143206 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.18 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.0367942 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.121827 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0281733 | No declared gate | DIAGNOSTIC |
| metrics[2].margin | 0.385155 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.45 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0648453 | No declared gate | DIAGNOSTIC |
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
| mean_rmse | 1.03932e-09 | No declared gate | DIAGNOSTIC |
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
| hq | 0.90625 | >= 0.9 | PASS |
| distribution_tv | 0.40625 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.018518 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.25 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.65625 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.03125 | No declared gate | DIAGNOSTIC |
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
| hq | 0.8125 | >= 0.9 | FAIL |
| distribution_tv | 0.1875 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0439276 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.28125 | No declared gate | DIAGNOSTIC |
| mode_fractions[2] | 0.0625 | No declared gate | DIAGNOSTIC |
| mode_fractions[3] | 0.375 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.28125 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.125 | No declared gate | DIAGNOSTIC |
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
| metrics[0].margin | -1 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 4 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 3 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | -0.0875 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.8125 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.173611 | No declared gate | DIAGNOSTIC |

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
| hq | 0.71875 | >= 0.9 | FAIL |
| distribution_tv | 0 | No declared gate | DIAGNOSTIC |
| mean_rmse | 0.0522922 | No declared gate | DIAGNOSTIC |
| mode_fractions[0] | 0.5 | No declared gate | DIAGNOSTIC |
| mode_fractions[1] | 0.5 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[0] | 0.40625 | No declared gate | DIAGNOSTIC |
| quality_mode_fractions[1] | 0.3125 | No declared gate | DIAGNOSTIC |
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
| metrics[1].margin | -0.18125 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.9 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.71875 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.100694 | No declared gate | DIAGNOSTIC |

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
| precision | 0.30345 | >= 0.97 | FAIL |
| modes | 17 | == 100 | FAIL |
| min_hq_mode_mass | 0.0002 | >= 0.005 | FAIL |
| mass_tv | 0.16115 | <= 0.1 | FAIL |
| max_mode_mass | 0.02975 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0429869 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.67322 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.32121 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.37525 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.30345 | No declared gate | DIAGNOSTIC |
| min_hq_count | 4 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.30345 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.43602 | <= 0.1 | FAIL |
| center_rms_sigma | 1.32115 | <= 0.2 | FAIL |
| mass_tv | 0.16115 | <= 0.06 | FAIL |
| radial_ks | 0.408031 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.96313 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.49105 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.755954 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.43602 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.685441 | No declared gate | DIAGNOSTIC |
| within_radius_n | 6069 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.29935 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.462706 | <= 0.1 | FAIL |
| center_rms_sigma | 1.31193 | <= 0.2 | FAIL |
| mass_tv | 0.15976 | <= 0.06 | FAIL |
| radial_ks | 0.411098 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.03171 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.35044 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.683237 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.462706 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.689541 | No declared gate | DIAGNOSTIC |
| within_radius_n | 29935 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.29934 | >= 0.97 | FAIL |
| modes | 17 | == 100 | FAIL |
| min_hq_mode_mass | 0.00032 | >= 0.005 | FAIL |
| mass_tv | 0.15976 | <= 0.1 | FAIL |
| max_mode_mass | 0.02966 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0927582 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.37681 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.40554 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.2375 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.29934 | No declared gate | DIAGNOSTIC |
| min_hq_count | 32 | No declared gate | DIAGNOSTIC |

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
| precision | 0.29935 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.462706 | <= 0.1 | FAIL |
| center_rms_sigma | 1.31193 | <= 0.2 | FAIL |
| mass_tv | 0.15976 | <= 0.06 | FAIL |
| radial_ks | 0.411098 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.03171 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.35044 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.683237 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.462706 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.689541 | No declared gate | DIAGNOSTIC |
| within_radius_n | 29935 | No declared gate | DIAGNOSTIC |

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
| precision | 0.2246 | >= 0.97 | FAIL |
| modes | 8 | == 100 | FAIL |
| min_hq_mode_mass | 0.0001 | >= 0.005 | FAIL |
| mass_tv | 0.13485 | <= 0.1 | FAIL |
| max_mode_mass | 0.02365 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.99274 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.33466 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.37765 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.2246 | No declared gate | DIAGNOSTIC |
| min_hq_count | 2 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.2246 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.40832 | <= 0.1 | FAIL |
| center_rms_sigma | 1.41948 | <= 0.2 | FAIL |
| mass_tv | 0.13485 | <= 0.06 | FAIL |
| radial_ks | 0.474401 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.32203 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.55212 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.797771 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.40832 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.764291 | No declared gate | DIAGNOSTIC |
| within_radius_n | 4492 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.2257 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.498164 | <= 0.1 | FAIL |
| center_rms_sigma | 1.3965 | <= 0.2 | FAIL |
| mass_tv | 0.13697 | <= 0.06 | FAIL |
| radial_ks | 0.479056 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.55584 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.55705 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.760195 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.498164 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.763191 | No declared gate | DIAGNOSTIC |
| within_radius_n | 22570 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.22569 | >= 0.97 | FAIL |
| modes | 10 | == 100 | FAIL |
| min_hq_mode_mass | 0.00014 | >= 0.005 | FAIL |
| mass_tv | 0.13697 | <= 0.1 | FAIL |
| max_mode_mass | 0.02335 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.056588 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.42662 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.64338 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.32489 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.22569 | No declared gate | DIAGNOSTIC |
| min_hq_count | 14 | No declared gate | DIAGNOSTIC |

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
| precision | 0.2257 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.498164 | <= 0.1 | FAIL |
| center_rms_sigma | 1.3965 | <= 0.2 | FAIL |
| mass_tv | 0.13697 | <= 0.06 | FAIL |
| radial_ks | 0.479056 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 6.55584 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.55705 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.760195 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.498164 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.763191 | No declared gate | DIAGNOSTIC |
| within_radius_n | 22570 | No declared gate | DIAGNOSTIC |

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
| precision | 0.34075 | >= 0.97 | FAIL |
| modes | 18 | == 100 | FAIL |
| min_hq_mode_mass | 0.0002 | >= 0.005 | FAIL |
| mass_tv | 0.12895 | <= 0.1 | FAIL |
| max_mode_mass | 0.02295 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0159732 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 3.35805 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.38552 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.14426 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.34075 | No declared gate | DIAGNOSTIC |
| min_hq_count | 4 | No declared gate | DIAGNOSTIC |

Live endpoint accuracy, 20000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 20000 | >= 20000 | PASS |
| valid_n | 20000 | == 20000 | PASS |
| precision | 0.34075 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.477489 | <= 0.1 | FAIL |
| center_rms_sigma | 1.22672 | <= 0.2 | FAIL |
| mass_tv | 0.12895 | <= 0.06 | FAIL |
| radial_ks | 0.395368 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.73546 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.43416 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.751221 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.477489 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.648141 | No declared gate | DIAGNOSTIC |
| within_radius_n | 6815 | No declared gate | DIAGNOSTIC |

Independent live holdout accuracy, 100000 saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.34371 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.487619 | <= 0.1 | FAIL |
| center_rms_sigma | 1.20339 | <= 0.2 | FAIL |
| mass_tv | 0.12345 | <= 0.06 | FAIL |
| radial_ks | 0.38768 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.66066 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.40046 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.673016 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.487619 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.645181 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34371 | No declared gate | DIAGNOSTIC |

Independent holdout coverage recomputed from the same saved draws

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| n | 100000 | >= 20000 | PASS |
| valid_n | 100000 | == 100000 | PASS |
| precision | 0.3437 | >= 0.97 | FAIL |
| modes | 17 | == 100 | FAIL |
| min_hq_mode_mass | 0.00039 | >= 0.005 | FAIL |
| mass_tv | 0.12345 | <= 0.1 | FAIL |
| max_mode_mass | 0.02311 | <= 0.02 | FAIL |
| min_cov_eig_ratio | 0.0349185 | >= 0.4 | FAIL |
| max_cov_eig_ratio | 2.38622 | <= 1.7 | FAIL |
| min_radial_median_ratio | 1.48731 | >= 0.65 | PASS |
| max_radial_median_ratio | 2.18468 | <= 1.4 | FAIL |
| passed | false | == true | FAIL |
| hq | 0.3437 | No declared gate | DIAGNOSTIC |
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
| precision | 0.34371 | >= 0.97 | FAIL |
| all_finite | true | == true | PASS |
| abs_cov_trace_bias | 0.487619 | <= 0.1 | FAIL |
| center_rms_sigma | 1.20339 | <= 0.2 | FAIL |
| mass_tv | 0.12345 | <= 0.06 | FAIL |
| radial_ks | 0.38768 | <= 0.04 | FAIL |
| accuracy_pass | false | == true | FAIL |
| frozen_pass | false | == true | FAIL |
| passed | false | == true | FAIL |
| accuracy_score | 5.66066 | No declared gate | DIAGNOSTIC |
| center_max_sigma | 2.40046 | No declared gate | DIAGNOSTIC |
| cov_frob_rms | 0.673016 | No declared gate | DIAGNOSTIC |
| cov_trace_bias | 0.487619 | No declared gate | DIAGNOSTIC |
| precision_gap | 0.645181 | No declared gate | DIAGNOSTIC |
| within_radius_n | 34371 | No declared gate | DIAGNOSTIC |

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
| independently confirmed scheduled states | 1 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.22602 | <= 0.2 | FAIL |
| std_ratio | 0.979462 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.102013 | <= 0.05 | FAIL |
| mean | 2.11301 | No declared gate | DIAGNOSTIC |
| std | 0.489731 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 709 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 709 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 1 | No declared gate | DIAGNOSTIC |

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
| recon_mse | 0.00753126 | <= 0.05 | PASS |
| hold | 0.026135 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0424687 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00753126 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.323865 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.026135 | No declared gate | DIAGNOSTIC |
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
| mass_tv | 0.0681152 | <= 0.15 | PASS |
| hq | 0.912842 | >= 0.85 | PASS |
| component_covariance_error | 5.41065 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.291513 | >= 0.15 | PASS |
| component_core_covariance_error | 0.476769 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 0.561416 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 0.644164 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.504025 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.574529 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.128418 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.338975 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.510833 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 0.480111 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.38525 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 0.295787 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.689774 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 0.648687 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.305234 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.311558 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.679919 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 0.569617 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.824969 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.781995 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.639144 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.62161 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.904545 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.526726 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.291472 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.394317 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.460482 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.685482 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.507719 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.606574 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.595889 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.632497 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.615456 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.850589 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.291472 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 296 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 235 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 220 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 347 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 308 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 266 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 218 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 313 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 270 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 179 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 200 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 271 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 256 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 254 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 217 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 2.3856 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 28.1879 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 0.894901 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 0.71313 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.263 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.336884 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 0.501961 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 0.598016 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.38525 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 0.317288 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 0.765773 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 0.686382 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.35148 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 0.28212 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 0.718654 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 49.1821 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0722656 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.057373 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0537109 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0847168 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0751953 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0649414 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0532227 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.076416 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.065918 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0437012 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0488281 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0661621 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0625 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0620117 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0529785 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.135135 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.260163 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.157447 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.0636364 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0806916 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0422078 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0488722 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.0963303 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0159744 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.0333333 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.0670391 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.05 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0221402 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.0273438 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0629921 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.285714 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0930781 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 256 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 182 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 198 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 206 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 319 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 295 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 253 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 197 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 308 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 261 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 167 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 265 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 249 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 238 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 155 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.285714 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0513819 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.605469 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.699219 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0478301 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.0818848 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0681152 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0628418 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.912842 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -4.56065 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 5.41065 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.141513 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.291513 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.333333 | No declared gate | DIAGNOSTIC |

## {'prior_reg': 0.001}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.998489 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.198242 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 6.48139e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 21 | No declared gate | DIAGNOSTIC |
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
| confirmed_steps[19] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[20] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 1667 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 21 | No declared gate | DIAGNOSTIC |

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

Saved-only reproduction commands require the [covariance feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/18e2aa1018f25de8c447a59287025df6f6aded3e) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

# BCAP prior proximal search

Executed source `a28c9631d59e2977616c0375e2a07c73171a9930`; protocol seed 0; ordinary revision 9.

The finite round is complete. All three positive strengths fail required Tier 1 gates, so all their Tier 2 tasks are **BLOCKED and unmeasured**. See the [failure readout](readout.md), [source/recipe/initializer/stream/budget audit](audit.json), and [49 actual-training GIFs with saved-artifact receipts](media/README.md).

Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.

| Actual strength | Tier 1 passes | Tier 1 failures | Tier 2 measured | Tier 2 outcome |
| ---: | --- | --- | --- | --- |
| 0, fresh leader control | 6/6 | none | 21/21 | 9 PASS, 12 FAIL |
| 0.1 | 4/6 | Gaussian acquisition, Ring16 acquisition | 0/21 | 21 BLOCKED |
| 1 | 5/6 | Ring16 acquisition | 0/21 | 21 BLOCKED |
| 10 | 4/6 | AE/GAN hold, Ring16 acquisition | 0/21 | 21 BLOCKED |

No Tier 2 repair or regression is measured for a positive strength. All nine fresh control Tier 2 passes are unresolved in each arm: five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, vector_two_broad, vector_spiral, img_stripes2.

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

## {'prior_proximal_strength': 0.1}: gaussian1d_smoke — FAIL

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
| mean_error_sigma | 0.0867332 | <= 0.2 | PASS |
| std_ratio | 0.9239 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.078219 | <= 0.05 | FAIL |
| mean | 1.95663 | No declared gate | DIAGNOSTIC |
| std | 0.46195 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| first_confirmed_step | unavailable | No declared gate | UNAVAILABLE |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 0 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 0.1}: two_pole — PASS

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

## {'prior_proximal_strength': 0.1}: unused_token_hold — PASS

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

## {'prior_proximal_strength': 0.1}: ae_gan_hold — PASS

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
| recon_mse | 0.00316379 | <= 0.05 | PASS |
| hold | 0.0114921 | <= 0.35 | PASS |

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
| metrics[0].margin | 0.0468362 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00316379 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.338508 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0114921 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 0.1}: ring16_acquisition — FAIL

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
| mass_tv | 0.064209 | <= 0.15 | PASS |
| hq | 0.862549 | >= 0.85 | PASS |
| component_covariance_error | 3.51422 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.372671 | >= 0.15 | PASS |
| component_core_covariance_error | 0.950248 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 1.33807 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.55454 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 1.26483 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 1.0141 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 0.647343 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.531455 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 0.752426 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.23469 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.325863 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 1.05262 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.852098 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.58561 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 0.25207 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.856112 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 0.872672 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 1.06946 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 1.2385 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.914094 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.773024 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.733715 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.919105 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.248494 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.593355 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.517265 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.584324 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.685916 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.558349 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 1.15654 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.87326 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.899719 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.701736 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 1.04656 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.248494 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 273 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 226 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 213 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 232 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 356 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 236 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 258 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 272 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 301 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 183 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 205 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 285 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 305 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 261 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 244 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 2.46026 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 2.2109 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 2.20387 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 1.43633 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 0.885901 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 0.569066 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 1.12513 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 1.94353 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 0.639379 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 1.53872 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 1.53669 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 2.96422 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 0.361374 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 1.36033 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 1.24946 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 33.7424 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0666504 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0551758 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.052002 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0566406 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0869141 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0576172 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0629883 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0664062 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0734863 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0446777 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0500488 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0695801 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0744629 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0637207 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0595703 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.190476 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.190265 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.159624 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.137931 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.0983146 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.0677966 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.0968992 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.194853 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.0398671 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.158537 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.120219 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.287805 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.0912281 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.104918 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.0766283 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.258197 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0882814 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 221 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 183 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 179 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 200 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 321 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 220 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 233 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 219 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 289 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 207 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 161 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 146 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 259 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 273 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 241 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 181 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.287805 | No declared gate | DIAGNOSTIC |
| mean_error | 0.00617088 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.570312 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.714844 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0326578 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.085791 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.064209 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | 0.0125488 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.862549 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -2.66422 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 3.51422 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.222671 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.372671 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.333333 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 0.1}: five_word_joint_smoke — PASS

any scheduled full joint pass plus independent same-state confirmation

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| independently confirmed scheduled states | 16 | >= 1 | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 1024 | >= 1024 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| modes | 5 | == 5 | PASS |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| minimum_reconstruction_token_probability | 0.999781 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.85763e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 16 | No declared gate | DIAGNOSTIC |
| confirmed_steps[0] | 2501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 3334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 5001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 5834 | No declared gate | DIAGNOSTIC |
| confirmed_steps[4] | 6667 | No declared gate | DIAGNOSTIC |
| confirmed_steps[5] | 7501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[6] | 8334 | No declared gate | DIAGNOSTIC |
| confirmed_steps[7] | 9168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[8] | 10001 | No declared gate | DIAGNOSTIC |
| confirmed_steps[9] | 11668 | No declared gate | DIAGNOSTIC |
| confirmed_steps[10] | 12501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[11] | 14168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[12] | 15835 | No declared gate | DIAGNOSTIC |
| confirmed_steps[13] | 17501 | No declared gate | DIAGNOSTIC |
| confirmed_steps[14] | 18335 | No declared gate | DIAGNOSTIC |
| confirmed_steps[15] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 2501 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 16 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 0.1}: gaussian1d_stability — BLOCKED

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

## {'prior_proximal_strength': 0.1}: five_word_joint_hold — BLOCKED

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

## {'prior_proximal_strength': 0.1}: trajectory — BLOCKED

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

## {'prior_proximal_strength': 0.1}: residual_student — BLOCKED

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

## {'prior_proximal_strength': 0.1}: unipolar — BLOCKED

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

## {'prior_proximal_strength': 0.1}: cover_leftover — BLOCKED

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

## {'prior_proximal_strength': 0.1}: mid_scale_identity — BLOCKED

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

## {'prior_proximal_strength': 0.1}: mode_hold — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_two_broad — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_unequal_mass — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_unequal_width — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_anisotropic — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_overlap — BLOCKED

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

## {'prior_proximal_strength': 0.1}: vector_spiral — BLOCKED

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

## {'prior_proximal_strength': 0.1}: img_stripes2 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: img_bars4 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: img_blobs4 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: img_intensity2 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: grid100 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: rotated100 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: staggered100 — BLOCKED

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

## {'prior_proximal_strength': 0.1}: clockfree_audit_measurement_v1 — PASS

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

## {'prior_proximal_strength': 1}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.0130435 | <= 0.2 | PASS |
| std_ratio | 1.01937 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.0261354 | <= 0.05 | PASS |
| mean | 2.00652 | No declared gate | DIAGNOSTIC |
| std | 0.509685 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 209 | No declared gate | DIAGNOSTIC |
| confirmed_steps[1] | 375 | No declared gate | DIAGNOSTIC |
| confirmed_steps[2] | 917 | No declared gate | DIAGNOSTIC |
| confirmed_steps[3] | 1000 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 5 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 1}: two_pole — PASS

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

## {'prior_proximal_strength': 1}: unused_token_hold — PASS

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

## {'prior_proximal_strength': 1}: ae_gan_hold — PASS

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 9 | >= 5 | PASS |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00562503 | <= 0.05 | PASS |
| hold | 0.00950276 | <= 0.35 | PASS |

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
| confirmation_fraction | 0.836 | No declared gate | DIAGNOSTIC |
| convergence.complete | true | No declared gate | DIAGNOSTIC |
| convergence.confirmed_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.confirmed_step | 209 | No declared gate | DIAGNOSTIC |
| convergence.first_pass_step | 42 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 9 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | 167 | No declared gate | DIAGNOSTIC |
| metrics[0].margin | 0.044375 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00562503 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.340497 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.00950276 | No declared gate | DIAGNOSTIC |
| passed | true | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 1}: ring16_acquisition — FAIL

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
| mass_tv | 0.0454102 | <= 0.15 | PASS |
| hq | 0.729492 | >= 0.85 | FAIL |
| component_covariance_error | 3.36332 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.9776 | >= 0.15 | PASS |
| component_core_covariance_error | 1.46223 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 2.22996 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.61197 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 0.963471 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 2.13001 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 1.4251 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 0.568281 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 1.6752 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.56581 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 1.82792 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 1.74109 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 0.943796 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.49367 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 1.22161 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 0.997506 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.17614 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 1.8241 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.604343 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 0.745977 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 0.635813 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.335072 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.657928 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.748479 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.719927 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.445002 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.616781 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.349431 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.483915 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.479689 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.216103 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.570315 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.691231 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.857583 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.216103 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 265 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 214 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 229 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 246 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 284 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 239 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 257 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 224 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 292 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 275 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 198 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 262 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 283 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 274 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 265 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 289 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 4.90435 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 3.57151 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 2.52713 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 3.78278 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 4.00684 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 2.55055 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 3.65912 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 4.02899 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 3.66952 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 3.05255 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 2.89917 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 2.94121 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 2.42297 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 2.36051 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 2.90176 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 4.53419 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0646973 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0522461 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.0559082 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0600586 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0693359 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0583496 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0627441 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0546875 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0712891 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0671387 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0483398 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0639648 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0690918 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0668945 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0646973 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0705566 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.366038 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.266355 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.240175 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.304878 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.295775 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.213389 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.303502 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.308036 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.297945 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.243636 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.222222 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.244275 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.194346 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.211679 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.237736 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.359862 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.0706078 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 168 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 157 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 174 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 171 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 200 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 188 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 179 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 155 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 205 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 208 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 154 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 198 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 228 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 216 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 202 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 185 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.366038 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0256148 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.601562 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.773438 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0324553 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.10459 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0454102 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.120508 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.729492 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -2.51332 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 3.36332 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 0.8276 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 0.9776 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.356962 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 1}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.999777 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.1875 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.208008 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 1.7557e-05 | No declared gate | DIAGNOSTIC |
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

## {'prior_proximal_strength': 1}: gaussian1d_stability — BLOCKED

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

## {'prior_proximal_strength': 1}: five_word_joint_hold — BLOCKED

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

## {'prior_proximal_strength': 1}: trajectory — BLOCKED

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

## {'prior_proximal_strength': 1}: residual_student — BLOCKED

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

## {'prior_proximal_strength': 1}: unipolar — BLOCKED

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

## {'prior_proximal_strength': 1}: cover_leftover — BLOCKED

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

## {'prior_proximal_strength': 1}: mid_scale_identity — BLOCKED

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

## {'prior_proximal_strength': 1}: mode_hold — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_two_broad — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_unequal_mass — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_unequal_width — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_anisotropic — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_overlap — BLOCKED

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

## {'prior_proximal_strength': 1}: vector_spiral — BLOCKED

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

## {'prior_proximal_strength': 1}: img_stripes2 — BLOCKED

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

## {'prior_proximal_strength': 1}: img_bars4 — BLOCKED

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

## {'prior_proximal_strength': 1}: img_blobs4 — BLOCKED

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

## {'prior_proximal_strength': 1}: img_intensity2 — BLOCKED

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

## {'prior_proximal_strength': 1}: grid100 — BLOCKED

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

## {'prior_proximal_strength': 1}: rotated100 — BLOCKED

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

## {'prior_proximal_strength': 1}: staggered100 — BLOCKED

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

## {'prior_proximal_strength': 1}: clockfree_audit_measurement_v1 — PASS

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

## {'prior_proximal_strength': 10}: gaussian1d_smoke — PASS

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
| mean_error_sigma | 0.239813 | <= 0.2 | FAIL |
| std_ratio | 0.879232 | >= 0.8 and <= 1.2 | PASS |
| cdf_ks | 0.15382 | <= 0.05 | FAIL |
| mean | 1.88009 | No declared gate | DIAGNOSTIC |
| std | 0.439616 | No declared gate | DIAGNOSTIC |
| step | 1000 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_steps[0] | 417 | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 417 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 1 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 10}: two_pole — PASS

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

## {'prior_proximal_strength': 10}: unused_token_hold — PASS

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

## {'prior_proximal_strength': 10}: ae_gan_hold — FAIL

recomputed complete live curve and terminal suffix

Sustained gate

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| terminal passing suffix | 4 | >= 5 | FAIL |
| passing scheduled observations | 19 | No declared gate | DIAGNOSTIC |
| complete declared execution | true | == true | PASS |

Recorded endpoint

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.00820117 | <= 0.05 | PASS |
| hold | 0.0230128 | <= 0.35 | PASS |

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
| convergence.first_pass_step | 32 | No declared gate | DIAGNOSTIC |
| convergence.minimum_stable_checks | 5 | No declared gate | DIAGNOSTIC |
| convergence.observations | 24 | No declared gate | DIAGNOSTIC |
| convergence.passing_observations | 19 | No declared gate | DIAGNOSTIC |
| convergence.passing_suffix | 4 | No declared gate | DIAGNOSTIC |
| convergence.stable_from_seconds | unavailable | No declared gate | UNAVAILABLE |
| convergence.stable_from_step | unavailable | No declared gate | UNAVAILABLE |
| metrics[0].margin | 0.0417988 | No declared gate | DIAGNOSTIC |
| metrics[0].threshold | 0.05 | No declared gate | DIAGNOSTIC |
| metrics[0].value | 0.00820117 | No declared gate | DIAGNOSTIC |
| metrics[1].margin | 0.326987 | No declared gate | DIAGNOSTIC |
| metrics[1].threshold | 0.35 | No declared gate | DIAGNOSTIC |
| metrics[1].value | 0.0230128 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 10}: ring16_acquisition — FAIL

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
| mass_tv | 0.0405273 | <= 0.15 | PASS |
| hq | 0.656494 | >= 0.85 | FAIL |
| component_covariance_error | 4.68711 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 1.66494 | >= 0.15 | PASS |
| component_core_covariance_error | 1.65637 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[0] | 1.84448 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[1] | 1.581 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[2] | 1.1618 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[3] | 0.508865 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[4] | 1.82251 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[5] | 2.02546 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[6] | 1.65075 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[7] | 1.76097 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[8] | 0.902068 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[9] | 1.34231 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[10] | 1.62765 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[11] | 1.32719 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[12] | 2.26622 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[13] | 2.45243 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[14] | 1.5891 | No declared gate | DIAGNOSTIC |
| component_core_covariance_errors[15] | 2.63914 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[0] | 0.83575 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[1] | 1.23408 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[2] | 1.48753 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[3] | 0.965916 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[4] | 0.780989 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[5] | 0.632605 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[6] | 0.590279 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[7] | 0.725074 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[8] | 0.311596 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[9] | 0.509109 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[10] | 0.257497 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[11] | 0.30013 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[12] | 0.3681 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[13] | 0.510012 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[14] | 0.528275 | No declared gate | DIAGNOSTIC |
| component_core_eigen_ratios[15] | 0.659918 | No declared gate | DIAGNOSTIC |
| component_core_min_eigen_ratio | 0.257497 | No declared gate | DIAGNOSTIC |
| component_counts[0] | 244 | No declared gate | DIAGNOSTIC |
| component_counts[1] | 229 | No declared gate | DIAGNOSTIC |
| component_counts[2] | 249 | No declared gate | DIAGNOSTIC |
| component_counts[3] | 273 | No declared gate | DIAGNOSTIC |
| component_counts[4] | 267 | No declared gate | DIAGNOSTIC |
| component_counts[5] | 280 | No declared gate | DIAGNOSTIC |
| component_counts[6] | 297 | No declared gate | DIAGNOSTIC |
| component_counts[7] | 258 | No declared gate | DIAGNOSTIC |
| component_counts[8] | 251 | No declared gate | DIAGNOSTIC |
| component_counts[9] | 276 | No declared gate | DIAGNOSTIC |
| component_counts[10] | 212 | No declared gate | DIAGNOSTIC |
| component_counts[11] | 261 | No declared gate | DIAGNOSTIC |
| component_counts[12] | 205 | No declared gate | DIAGNOSTIC |
| component_counts[13] | 252 | No declared gate | DIAGNOSTIC |
| component_counts[14] | 302 | No declared gate | DIAGNOSTIC |
| component_counts[15] | 240 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[0] | 5.85032 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[1] | 3.8235 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[2] | 3.14784 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[3] | 2.51959 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[4] | 6.73314 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[5] | 5.25346 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[6] | 4.23546 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[7] | 7.1799 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[8] | 3.52194 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[9] | 2.99725 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[10] | 3.9525 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[11] | 3.54431 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[12] | 4.76236 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[13] | 5.85099 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[14] | 3.89059 | No declared gate | DIAGNOSTIC |
| component_covariance_errors[15] | 7.73068 | No declared gate | DIAGNOSTIC |
| component_mass[0] | 0.0595703 | No declared gate | DIAGNOSTIC |
| component_mass[1] | 0.0559082 | No declared gate | DIAGNOSTIC |
| component_mass[2] | 0.060791 | No declared gate | DIAGNOSTIC |
| component_mass[3] | 0.0666504 | No declared gate | DIAGNOSTIC |
| component_mass[4] | 0.0651855 | No declared gate | DIAGNOSTIC |
| component_mass[5] | 0.0683594 | No declared gate | DIAGNOSTIC |
| component_mass[6] | 0.0725098 | No declared gate | DIAGNOSTIC |
| component_mass[7] | 0.0629883 | No declared gate | DIAGNOSTIC |
| component_mass[8] | 0.0612793 | No declared gate | DIAGNOSTIC |
| component_mass[9] | 0.0673828 | No declared gate | DIAGNOSTIC |
| component_mass[10] | 0.0517578 | No declared gate | DIAGNOSTIC |
| component_mass[11] | 0.0637207 | No declared gate | DIAGNOSTIC |
| component_mass[12] | 0.0500488 | No declared gate | DIAGNOSTIC |
| component_mass[13] | 0.0615234 | No declared gate | DIAGNOSTIC |
| component_mass[14] | 0.0737305 | No declared gate | DIAGNOSTIC |
| component_mass[15] | 0.0585938 | No declared gate | DIAGNOSTIC |
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
| component_spill[0] | 0.42623 | No declared gate | DIAGNOSTIC |
| component_spill[1] | 0.340611 | No declared gate | DIAGNOSTIC |
| component_spill[2] | 0.236948 | No declared gate | DIAGNOSTIC |
| component_spill[3] | 0.223443 | No declared gate | DIAGNOSTIC |
| component_spill[4] | 0.411985 | No declared gate | DIAGNOSTIC |
| component_spill[5] | 0.357143 | No declared gate | DIAGNOSTIC |
| component_spill[6] | 0.397306 | No declared gate | DIAGNOSTIC |
| component_spill[7] | 0.360465 | No declared gate | DIAGNOSTIC |
| component_spill[8] | 0.258964 | No declared gate | DIAGNOSTIC |
| component_spill[9] | 0.221014 | No declared gate | DIAGNOSTIC |
| component_spill[10] | 0.316038 | No declared gate | DIAGNOSTIC |
| component_spill[11] | 0.275862 | No declared gate | DIAGNOSTIC |
| component_spill[12] | 0.463415 | No declared gate | DIAGNOSTIC |
| component_spill[13] | 0.464286 | No declared gate | DIAGNOSTIC |
| component_spill[14] | 0.307947 | No declared gate | DIAGNOSTIC |
| component_spill[15] | 0.475 | No declared gate | DIAGNOSTIC |
| covariance_error | 0.084627 | No declared gate | DIAGNOSTIC |
| hq_component_counts[0] | 140 | No declared gate | DIAGNOSTIC |
| hq_component_counts[1] | 151 | No declared gate | DIAGNOSTIC |
| hq_component_counts[2] | 190 | No declared gate | DIAGNOSTIC |
| hq_component_counts[3] | 212 | No declared gate | DIAGNOSTIC |
| hq_component_counts[4] | 157 | No declared gate | DIAGNOSTIC |
| hq_component_counts[5] | 180 | No declared gate | DIAGNOSTIC |
| hq_component_counts[6] | 179 | No declared gate | DIAGNOSTIC |
| hq_component_counts[7] | 165 | No declared gate | DIAGNOSTIC |
| hq_component_counts[8] | 186 | No declared gate | DIAGNOSTIC |
| hq_component_counts[9] | 215 | No declared gate | DIAGNOSTIC |
| hq_component_counts[10] | 145 | No declared gate | DIAGNOSTIC |
| hq_component_counts[11] | 189 | No declared gate | DIAGNOSTIC |
| hq_component_counts[12] | 110 | No declared gate | DIAGNOSTIC |
| hq_component_counts[13] | 135 | No declared gate | DIAGNOSTIC |
| hq_component_counts[14] | 209 | No declared gate | DIAGNOSTIC |
| hq_component_counts[15] | 126 | No declared gate | DIAGNOSTIC |
| max_component_spill | 0.475 | No declared gate | DIAGNOSTIC |
| mean_error | 0.0463251 | No declared gate | DIAGNOSTIC |
| min_hq_mass_ratio | 0.429688 | No declared gate | DIAGNOSTIC |
| min_mass_ratio | 0.800781 | No declared gate | DIAGNOSTIC |
| resolved_core_covariance_error | 0 | No declared gate | DIAGNOSTIC |
| resolved_core_min_eigen_ratio | 1 | No declared gate | DIAGNOSTIC |
| resolved_max_component_spill | 0 | No declared gate | DIAGNOSTIC |
| step | 1600 | No declared gate | DIAGNOSTIC |
| sw1_normalized | 0.0421207 | No declared gate | DIAGNOSTIC |
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
| metrics[2].margin | 0.109473 | No declared gate | DIAGNOSTIC |
| metrics[2].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[2].value | 0.0405273 | No declared gate | DIAGNOSTIC |
| metrics[3].margin | -0.193506 | No declared gate | DIAGNOSTIC |
| metrics[3].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[3].value | 0.656494 | No declared gate | DIAGNOSTIC |
| metrics[4].margin | -3.83711 | No declared gate | DIAGNOSTIC |
| metrics[4].threshold | 0.85 | No declared gate | DIAGNOSTIC |
| metrics[4].value | 4.68711 | No declared gate | DIAGNOSTIC |
| metrics[5].margin | 1.51494 | No declared gate | DIAGNOSTIC |
| metrics[5].threshold | 0.15 | No declared gate | DIAGNOSTIC |
| metrics[5].value | 1.66494 | No declared gate | DIAGNOSTIC |
| passed | false | No declared gate | DIAGNOSTIC |
| shortfall | 0.371276 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 10}: five_word_joint_smoke — PASS

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
| minimum_reconstruction_token_probability | 0.999173 | >= 0.9 | PASS |
| completed_steps | 20001 | No declared gate | DIAGNOSTIC |
| mass_apple | 0.198242 | No declared gate | DIAGNOSTIC |
| mass_berry | 0.208008 | No declared gate | DIAGNOSTIC |
| mass_grape | 0.195312 | No declared gate | DIAGNOSTIC |
| mass_lemon | 0.210938 | No declared gate | DIAGNOSTIC |
| mass_melon | 0.1875 | No declared gate | DIAGNOSTIC |
| output_noise_added | 0 | No declared gate | DIAGNOSTIC |
| policy_latent_perturbation | 0 | No declared gate | DIAGNOSTIC |
| reconstruction_nll | 3.53631e-05 | No declared gate | DIAGNOSTIC |
| served_averaged | 0 | No declared gate | DIAGNOSTIC |
| step | 20001 | No declared gate | DIAGNOSTIC |

Recorded evaluator diagnostics (no additional qualification credit)

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| confirmed_checks | 22 | No declared gate | DIAGNOSTIC |
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
| confirmed_steps[20] | 19168 | No declared gate | DIAGNOSTIC |
| confirmed_steps[21] | 20001 | No declared gate | DIAGNOSTIC |
| endpoint_passed | true | No declared gate | DIAGNOSTIC |
| first_confirmed_step | 1667 | No declared gate | DIAGNOSTIC |
| observations | 24 | No declared gate | DIAGNOSTIC |
| passing_observations | 22 | No declared gate | DIAGNOSTIC |

## {'prior_proximal_strength': 10}: gaussian1d_stability — BLOCKED

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

## {'prior_proximal_strength': 10}: five_word_joint_hold — BLOCKED

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

## {'prior_proximal_strength': 10}: trajectory — BLOCKED

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

## {'prior_proximal_strength': 10}: residual_student — BLOCKED

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

## {'prior_proximal_strength': 10}: unipolar — BLOCKED

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

## {'prior_proximal_strength': 10}: cover_leftover — BLOCKED

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

## {'prior_proximal_strength': 10}: mid_scale_identity — BLOCKED

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

## {'prior_proximal_strength': 10}: mode_hold — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_two_broad — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_unequal_mass — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_unequal_width — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_anisotropic — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_overlap — BLOCKED

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

## {'prior_proximal_strength': 10}: vector_spiral — BLOCKED

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

## {'prior_proximal_strength': 10}: img_stripes2 — BLOCKED

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

## {'prior_proximal_strength': 10}: img_bars4 — BLOCKED

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

## {'prior_proximal_strength': 10}: img_blobs4 — BLOCKED

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

## {'prior_proximal_strength': 10}: img_intensity2 — BLOCKED

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

## {'prior_proximal_strength': 10}: grid100 — BLOCKED

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

## {'prior_proximal_strength': 10}: rotated100 — BLOCKED

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

## {'prior_proximal_strength': 10}: staggered100 — BLOCKED

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

## {'prior_proximal_strength': 10}: clockfree_audit_measurement_v1 — PASS

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

Saved-only reproduction commands require the [proximal feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/a28c9631d59e2977616c0375e2a07c73171a9930) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

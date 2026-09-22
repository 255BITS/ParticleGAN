# Transfer suite: importance-aware selection

Required tests determine eligibility. Ranking tests measure useful generalization without vetoing selection. Diagnostic stress tests have zero influence on selection and remain visible. Tiers never change because a candidate failed.

The nine existing behavioral regressions remain required. The 24 new cases are development data: 16 vector/dynamics cases for fitting and eight image cases for validation. Three reserved families are evaluated only after the challenger is frozen. All training uses seed 0; EMA never determines success.

## Candidate leaderboard

| Candidate | Eligible / ready | Required sustained | Ranking sustained | Balanced pass score | Diagnostic sustained | Ranking shortfall | Phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cosine | Yes / Yes | 9/9 (9 tried) | 4/16 (16 tried) | 25.0% | 2/8 (8 tried) | 0.2814 | full development complete |
| transfer_g00_p02 | Yes / Yes | 9/9 (9 tried) | 3/16 (16 tried) | 16.7% | 2/8 (8 tried) | 0.3116 | full development complete |
| transfer_g00_p04 | Yes / Yes | 9/9 (9 tried) | 3/16 (16 tried) | 16.7% | 2/8 (8 tried) | 0.3888 | full development complete |
| previous_feedback | Yes / No | 9/9 (9 tried) | 3/16 (12 tried) | 16.7% | 2/8 (4 tried) | 0.8521 | fit complete |
| transfer_g01_p04 | Yes / No | 9/9 (9 tried) | 2/16 (12 tried) | 11.1% | 2/8 (4 tried) | 0.8649 | fit complete |
| frozen_bias_only | No / No | 8/9 (9 tried) | 3/16 (16 tried) | 16.7% | 2/8 (8 tried) | 0.3298 | post-freeze ablation |
| transfer_g00_p03 | No / No | 8/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |
| transfer_g01_p00 | No / No | 8/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |
| transfer_g01_p02 | No / No | 8/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |
| transfer_g01_p05 | No / No | 8/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |
| transfer_g00_p05 | No / No | 7/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |
| transfer_g01_p03 | No / No | 7/9 (9 tried) | 0/16 (0 tried) | 0.0% | 0/8 (0 tried) | 2.0000 | screened: required failure |

Eligible means all nine required live tests sustain success. Ready additionally means every required/ranking case was attempted. Missing cases remain in denominators. A recorded training error counts as a failure. Balanced pass score gives data, dynamics and images equal weight, then averages ranking families within each domain equally. Duplicating cases in one family does not give that family more weight. Diagnostic results, timing and EMA cannot break ties.

Eligibility requires every required live test sustained. All required and ranking tests must be attempted before selection. Rank eligible rows by sustained pass fraction averaged equally across data, dynamics and image domains and equally across ranking families within each domain, then worst family fraction, then lower similarly balanced final metric shortfall, then lower confirmation/budget (failure=2). Shortfall is mean positive bound violation divided by abs(bound), with scale 1 for zero bounds and each metric capped at 2; invalid or missing results receive 2. Diagnostics never affect eligibility, score, or tie breaking. Errors are failures; missing attempts cannot improve rank. EMA never affects selection.

## Test importance and reference evidence

Importance describes intended use, not how easy a test is. Reference evidence establishes solvability for an exact budget/setup; it does not establish how well that test predicts real-world transfer. That predictive value is currently unmeasured.

| Test | Use / family | Importance | Why it matters | Reference sustained success | Limitation |
| --- | --- | --- | --- | --- | --- |
| two_pole | fit / existing_behavior | required | Basic adversarial movement remains bounded. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| trajectory | fit / existing_behavior | required | Preserves the shared trajectory's identity constraint. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| residual_student | fit / existing_behavior | required | Moves the intended residual without choosing the wrong target. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| unipolar | fit / existing_behavior | required | Target coverage preserves unrelated content. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| ae_gan_hold | fit / existing_behavior | required | Adversarial training preserves reconstruction and hold behavior. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| cover_leftover | fit / existing_behavior | required | Coverage preserves both content and the unwanted remainder constraint. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| unused_token_hold | fit / existing_behavior | required | Unused controls remain unchanged while active controls move. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| mid_scale_identity | fit / existing_behavior | required | Intermediate control strengths preserve identity and target magnitude. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| mode_hold | fit / existing_behavior | required | All eight modes must remain present with sufficient sample quality. | cosine | Small extracted behavioral host; not evidence of natural-image transfer. |
| vector_two_broad | fit / separated_broad | ranking | Basic learnable multimodal distribution and within-mode spread. | cosine | Finite-particle approximation; one initialization only. |
| vector_unequal_mass | fit / unequal_mass | ranking | Checks target occupancy including the rare 2% component, not uniformity. | not demonstrated | Finite-particle approximation; one initialization only. |
| vector_unequal_width | fit / unequal_width | ranking | Checks component-specific scales without imposing one shared Gaussian width. | not demonstrated | Finite-particle approximation; one initialization only. |
| vector_anisotropic | fit / anisotropic | ranking | Checks covariance shape: a narrow axis cannot be rescued by a wide one. | cosine | Finite-particle approximation; one initialization only. |
| vector_overlap | fit / overlapping | ranking | Scores the observable distribution when latent components are not identifiable. | not demonstrated | Component labels and mode recall are deliberately not scored for overlapping densities. |
| vector_narrow | fit / narrow_resolution | diagnostic | Deliberately narrow components test critic resolution; nonblocking diagnostic. | not demonstrated | Fixed two-band Fourier critic is deliberately mismatched to sigma=.025; failure does not veto a controller. |
| vector_scale_drift | fit / changing_scale | diagnostic | Tests adaptation to changing input units before a stationary final window. | cosine | Nonstationary diagnostic; each checkpoint uses the current target, which stops changing at 60% of budget. |
| vector_spiral | fit / curved_continuous | ranking | Checks continuous curved mass rather than a finite list of target mode centers. | cosine | Finite particles approximate a noisy spiral; sliced distances do not prove identical density. |
| reserved_annulus | reserved / annulus | ranking | Unseen rotationally symmetric continuous support and radial mass law. | reserved_cosine | Reserved family; no samples or evaluations are produced during calibration. |
| stress_fast_critic | fit / learning_rate_imbalance | ranking | A critic learning twice as fast is a realistic optimizer imbalance; maintaining distribution quality matters for routine tuning. | not demonstrated | Changes only the critic learning rate; it does not test additional discriminator updates or minibatch changes. |
| stress_slow_critic | fit / learning_rate_imbalance | ranking | A critic learning half as fast tests whether feedback remains useful when generator transport can outrun density estimation. | not demonstrated | The critic retains adequate architecture; failure within the fixed budget is not proof the target is unsolvable. |
| stress_small_batch | fit / minibatch_noise | ranking | Batch 64 is an ordinary memory-limited setting; feedback should tolerate its noisier gradient observations. | not demonstrated | The update count stays fixed, so this arm sees fewer training examples; report that compute/data difference rather than attributing everything to noise. |
| stress_large_critic | fit / capacity_imbalance | ranking | Doubling critic width is a plausible architecture choice that tests controller transfer across gradient scale and capacity. | not demonstrated | A wider critic costs more per update; use wall time alongside update-normalized convergence. |
| stress_long_horizon | fit / training_horizon | ranking | A doubled training horizon checks persistence of convergence and delayed collapse beyond the usual training budget. | not demonstrated | There are still only 24 fixed observations, now 100 steps apart; stability between observations is not certified. |
| stress_r1_r2 | fit / gradient_penalty_formulation | ranking | A supported R1+R2 penalty tests whether a controller transfers across common discriminator regularization objectives. | not demonstrated | Coefficient .1 is a predeclared reference alternative, not an equal-strength equivalence to the cap penalty or a search over coefficients. |
| stress_weak_critic | fit / architecture_limit | diagnostic | A deliberately narrow, shallow critic without Fourier features diagnoses an architecture bottleneck; this artificial weakness is not a selection requirement. | not demonstrated | Failure can arise because the critic cannot resolve narrow target modes. It supplies diagnostic evidence only, even if another method succeeds. |
| stress_overlapping_data | fit / data_ambiguity | diagnostic | Broad overlapping components diagnose sensitivity to ambiguous mixture labels. Distribution fit matters here; component reconstruction is not an appropriate requirement. | cosine | This intentionally blurred data is not representative of the separated-mode objective. Only global sliced distance is scored; HQ and component-mass/covariance claims are omitted. |
| reserved_alternating_critic_updates | reserved / update_cadence | ranking | A previously unseen discriminator update cadence tests transfer when opponent feedback arrives less often, rather than merely at a different learning rate. | not demonstrated | Never evaluated in development. The target stays static; D updates every second outer step and G every step. Report both actual update counts and wall time because the work per outer step changes. |
| img_stripes2 | validation / image_conv_transpose | ranking | Healthy orientation transfer: two distinct stripe orientations with an adequately sized convolutional GAN. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_bars4 | validation / image_conv_transpose | ranking | Healthy location transfer: four horizontal/vertical bar positions test spatial coverage. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_blobs4 | validation / image_conv_transpose | ranking | Healthy location transfer: four small corner patches test localized quality and coverage. | cosine | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_intensity2 | validation / image_conv_transpose | ranking | Healthy photometric transfer: two patch intensities require intensity fidelity as well as support coverage. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_bars8 | validation / image_conv_transpose | diagnostic | Denser support stress: eight bar positions may exceed the short budget; failure cannot disqualify a controller. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_tiny_generator | validation / image_conv_transpose | diagnostic | Undercapacity stress: width2 and a one-dimensional latent may constrain representation and optimization; non-blocking. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_mean_discriminator | validation / image_conv_transpose | diagnostic | Low-information architecture stress: D sees only image mean; equal-mass patch positions are indistinguishable, so failure is non-blocking. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_uniform_generator | validation / image_conv_transpose | diagnostic | Known representation failure: G can only output spatially uniform images, so stripe quality is impossible; diagnostic only. | not demonstrated | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |
| img_residual_bars4 | reserved / image_conv_residual_upsample | ranking | Reserved architecture transfer: nearest-neighbor upsampling with residual convolutions; never evaluated during development. | reserved_cosine | Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim. |

## Per-test live results

### Required

| Test | cosine | previous_feedback | transfer_g00_p02 | transfer_g00_p03 | transfer_g00_p04 | transfer_g00_p05 | transfer_g01_p00 | transfer_g01_p02 | transfer_g01_p03 | transfer_g01_p04 | transfer_g01_p05 | frozen_bias_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_pole | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| trajectory | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL | PASS | PASS | PASS |
| residual_student | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| unipolar | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| ae_gan_hold | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| cover_leftover | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| unused_token_hold | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mid_scale_identity | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mode_hold | PASS | PASS | PASS | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |

### Ranking

| Test | cosine | previous_feedback | transfer_g00_p02 | transfer_g00_p03 | transfer_g00_p04 | transfer_g00_p05 | transfer_g01_p00 | transfer_g01_p02 | transfer_g01_p03 | transfer_g01_p04 | transfer_g01_p05 | frozen_bias_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_two_broad | PASS | PASS | PASS | MISSING | PASS | MISSING | MISSING | MISSING | MISSING | PASS | MISSING | PASS |
| vector_unequal_mass | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| vector_unequal_width | FAIL | FAIL | PASS | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| vector_anisotropic | PASS | PASS | FAIL | MISSING | PASS | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | PASS |
| vector_overlap | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| vector_spiral | PASS | PASS | PASS | MISSING | PASS | MISSING | MISSING | MISSING | MISSING | PASS | MISSING | PASS |
| stress_fast_critic | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_slow_critic | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_small_batch | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_large_critic | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_long_horizon | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_r1_r2 | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| img_stripes2 | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_bars4 | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_blobs4 | PASS | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_intensity2 | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |

### Diagnostic

| Test | cosine | previous_feedback | transfer_g00_p02 | transfer_g00_p03 | transfer_g00_p04 | transfer_g00_p05 | transfer_g01_p00 | transfer_g01_p02 | transfer_g01_p03 | transfer_g01_p04 | transfer_g01_p05 | frozen_bias_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_narrow | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| vector_scale_drift | PASS | PASS | PASS | MISSING | PASS | MISSING | MISSING | MISSING | MISSING | PASS | MISSING | PASS |
| stress_weak_critic | FAIL | FAIL | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | FAIL | MISSING | FAIL |
| stress_overlapping_data | PASS | PASS | PASS | MISSING | PASS | MISSING | MISSING | MISSING | MISSING | PASS | MISSING | PASS |
| img_bars8 | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_tiny_generator | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_mean_discriminator | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |
| img_uniform_generator | FAIL | MISSING | FAIL | MISSING | FAIL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | FAIL |

## Frozen transfer

Development winner: **cosine**. Frozen nonzero challenger: **transfer_g00_p02**. All reserved results are excluded from this choice.

| Reserved family / test | Arm | Sustained live | Final live metrics | EMA | Seconds |
| --- | --- | --- | --- | --- | ---: |
| annulus / reserved_annulus | reserved_cosine | PASS | sw1_normalized=0.0373477403870264, mean_error=0.03529319167137146, covariance_error=0.07799136638641357 | sw1_normalized=0.026251834559376644, mean_error=0.01689591072499752, covariance_error=0.03455396369099617 | 7.64 |
| update_cadence / reserved_alternating_critic_updates | reserved_cosine | FAIL | sw1_normalized=0.07217872020142986, mass_tv=0.116455078125, hq=0.900390625, component_covariance_error=15.384737007319927, component_min_eigen_ratio=0.3772438168525696 | sw1_normalized=0.07225293517130942, mass_tv=0.116455078125, hq=0.904541015625, component_covariance_error=15.400326207280159, component_min_eigen_ratio=0.3318338990211487 | 8.20 |
| image_conv_residual_upsample / img_residual_bars4 | reserved_cosine | PASS | modes=4, hq=0.9375 | modes=4, hq=0.9375 | 6.60 |
| annulus / reserved_annulus | reserved_feedback | PASS | sw1_normalized=0.03122307419259798, mean_error=0.018438030034303665, covariance_error=0.06446018815040588 | sw1_normalized=0.02691146052083072, mean_error=0.011738132685422897, covariance_error=0.01747356541454792 | 8.68 |
| update_cadence / reserved_alternating_critic_updates | reserved_feedback | FAIL | sw1_normalized=0.08355570050718522, mass_tv=0.098388671875, hq=0.94189453125, component_covariance_error=10.8199004791677, component_min_eigen_ratio=0.4831632971763611 | sw1_normalized=0.08344639163682, mass_tv=0.098388671875, hq=0.94189453125, component_covariance_error=10.692105032503605, component_min_eigen_ratio=0.5222596526145935 | 8.14 |
| image_conv_residual_upsample / img_residual_bars4 | reserved_feedback | FAIL | modes=2, hq=0.71875 | modes=2, hq=0.71875 | 6.11 |
| annulus / reserved_annulus | reserved_bias_only | PASS | sw1_normalized=0.031478994246163586, mean_error=0.004407302942126989, covariance_error=0.07109783589839935 | sw1_normalized=0.025098878146861913, mean_error=0.013594701886177063, covariance_error=0.02550428919494152 | 7.87 |
| update_cadence / reserved_alternating_critic_updates | reserved_bias_only | FAIL | sw1_normalized=0.06003658276268548, mass_tv=0.072265625, hq=0.9150390625, component_covariance_error=10.929692730307579, component_min_eigen_ratio=0.42459532618522644 | sw1_normalized=0.058664791156575136, mass_tv=0.072265625, hq=0.9150390625, component_covariance_error=10.962823927402496, component_min_eigen_ratio=0.30946552753448486 | 9.61 |
| image_conv_residual_upsample / img_residual_bars4 | reserved_bias_only | PASS | modes=4, hq=0.9375 | modes=4, hq=0.9375 | 7.16 |

[Declared protocol](manifest.json) · [Full results and episode hashes](results.json.gz) · [Exact source bundle](source.tar.gz). Every episode retains its action trace and full curve as a compressed JSON artifact.

Times are single CPU observations including measurement and controller work. No seed sweeps were run. The suite does not establish a natural-image or large-network default. Reference calibration attempts are separately retained.

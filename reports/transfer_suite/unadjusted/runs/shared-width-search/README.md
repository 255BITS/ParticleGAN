# Unequal-width discriminator stability search

Every trial uses exact `shared_c6`: Rp logistic, cap6/κ1.25, spread .05, no particle L2, Adam(0,.99), G/D LR .00425 and particle LR .0085, delayed cosine60%/floor5%. Only the discriminator changes; original G, data, initialization rules,256 particles,batch128,1200 updates and all behavioral bounds remain fixed. Seed0, CPU, one Torch thread. EMA is separate.

Sixteen initial raw-input smooth architectures were declared before results and checked against prior normalized discriminator identities. After that negative screen, a separately frozen12-card head/skip refinement was authorized and recorded in scope-v2; the original plan and numerical sources were preserved. A final explicitly authorized six-card scope-v3 tests Softplus sharpness and main-branch gains for the SiLU skip critic. Thus34 unique width architectures are measured; no failed attempt is omitted. Every failed curve is retained. A live PASS requires all24 measurements and at least five final passing checks. Final-only success is a failure. Every width winner is checked on rare mass with exactly the same architecture and recipe; other data tests are unrun for these cards.

**Sustained width winners:** width_last_softplus8_128_l3. 35 episodes are retained: 34 width trials and 1 rare cross-checks.

| Discriminator | D params | Width live | Final streak | Width EMA | Rare live | Final width failing bounds |
| --- | ---: | --- | ---: | --- | --- | --- |
| width_last_softplus8_128_l3 | 33537 | [PASS](last_refinement/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_width.json.gz) | 5/24 | PASS | [FAIL](cross/episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_mass.json.gz) | none |
| width_raw_softplus128_l3 | 33537 | [FAIL](screen/episodes/shared_c6__width_raw_softplus128_l3__vector_unequal_width.json.gz) | 1/24 | FAIL | unrun | none; stability fails |
| width_ref_silu160_l2_linear_skip | 26403 | [FAIL](refinement/episodes/shared_c6__width_ref_silu160_l2_linear_skip__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.147984 |
| width_last_softplus6_128_l3 | 33537 | [FAIL](last_refinement/episodes/shared_c6__width_last_softplus6_128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_covariance_error=0.964443 |
| width_ref_softplus1_128_l3 | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus1_128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | hq=0.816162, component_min_eigen_ratio=0.130901 |
| width_raw_softplus2_128_l3 | 33537 | [FAIL](screen/episodes/shared_c6__width_raw_softplus2_128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.117556 |
| width_raw_silu160_l2 | 26401 | [FAIL](screen/episodes/shared_c6__width_raw_silu160_l2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.103656 |
| width_ref_softplus128_l3_head025 | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus128_l3_head025__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0951421 |
| width_raw_silu160_l4 | 77921 | [FAIL](screen/episodes/shared_c6__width_raw_silu160_l4__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.077398 |
| width_ref_softplus10_128_l3 | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus10_128_l3__vector_unequal_width.json.gz) | 0/24 | PASS | unrun | component_min_eigen_ratio=0.0750955 |
| width_ref_softplus128_l3_head2 | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus128_l3_head2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0742416 |
| width_raw_silu64_l3 | 8577 | [FAIL](screen/episodes/shared_c6__width_raw_silu64_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0673041 |
| width_ref_silu160_l2_head05 | 26401 | [FAIL](refinement/episodes/shared_c6__width_ref_silu160_l2_head05__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0622386 |
| width_ref_softplus128_l3_head05 | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus128_l3_head05__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0601424 |
| width_last_softplus3_128_l3 | 33537 | [FAIL](last_refinement/episodes/shared_c6__width_last_softplus3_128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0580842 |
| width_raw_silu96_l3 | 19009 | [FAIL](screen/episodes/shared_c6__width_raw_silu96_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.057282 |
| width_residual_raw_silu128_l3 | 33537 | [FAIL](screen/episodes/shared_c6__width_residual_raw_silu128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.056998 |
| width_ref_softplus128_l3_residual | 33537 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus128_l3_residual__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0555454 |
| width_raw_silu128_l2 | 17025 | [FAIL](screen/episodes/shared_c6__width_raw_silu128_l2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0493072 |
| width_raw_silu96_l4 | 28321 | [FAIL](screen/episodes/shared_c6__width_raw_silu96_l4__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0424694 |
| width_raw_silu192_l2 | 37825 | [FAIL](screen/episodes/shared_c6__width_raw_silu192_l2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0410727 |
| width_ref_silu160_l2_head2 | 26401 | [FAIL](refinement/episodes/shared_c6__width_ref_silu160_l2_head2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0365164 |
| width_last_softplus4_128_l3 | 33537 | [FAIL](last_refinement/episodes/shared_c6__width_last_softplus4_128_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0351095 |
| width_ref_softplus128_l3_linear_skip | 33539 | [FAIL](refinement/episodes/shared_c6__width_ref_softplus128_l3_linear_skip__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0337566 |
| width_raw_silu160_l3 | 52161 | [FAIL](screen/episodes/shared_c6__width_raw_silu160_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0271187 |
| width_raw_silu128_l3_linear_skip | 33539 | [FAIL](screen/episodes/shared_c6__width_raw_silu128_l3_linear_skip__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0265243 |
| width_ref_silu160_l2_residual | 26401 | [FAIL](refinement/episodes/shared_c6__width_ref_silu160_l2_residual__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | hq=0.744629, component_min_eigen_ratio=0.0413832 |
| width_raw_silu96_l2 | 9697 | [FAIL](screen/episodes/shared_c6__width_raw_silu96_l2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_covariance_error=0.888437, component_min_eigen_ratio=0.0253681 |
| width_last_silu160_l2_linear_skip_head05 | 26403 | [FAIL](last_refinement/episodes/shared_c6__width_last_silu160_l2_linear_skip_head05__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0160709 |
| width_raw_softplus128_l4 | 50049 | [FAIL](screen/episodes/shared_c6__width_raw_softplus128_l4__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.0124518 |
| width_ref_silu160_l2_head025 | 26401 | [FAIL](refinement/episodes/shared_c6__width_ref_silu160_l2_head025__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.012235 |
| width_raw_silu192_l3 | 74881 | [FAIL](screen/episodes/shared_c6__width_raw_silu192_l3__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_min_eigen_ratio=0.00436551 |
| width_last_silu160_l2_linear_skip_head2 | 26403 | [FAIL](last_refinement/episodes/shared_c6__width_last_silu160_l2_linear_skip_head2__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_covariance_error=0.964571, component_min_eigen_ratio=0.0111672 |
| width_raw_silu128_l4 | 50049 | [FAIL](screen/episodes/shared_c6__width_raw_silu128_l4__vector_unequal_width.json.gz) | 0/24 | FAIL | unrun | component_covariance_error=1.16511, component_min_eigen_ratio=0.0248446 |

## Last five measurements of the strongest width results

All24 live and EMA observations for every attempt are in [curves.json](curves.json). The columns below are steps1000,1050,1100,1150,1200.

| D | Metric / bound | 1000 | 1050 | 1100 | 1150 | 1200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| width_last_softplus8_128_l3 | sw1_normalized <= 0.18 | 0.0977202 | 0.0963939 | 0.0864971 | 0.0802527 | 0.0782177 |
| width_last_softplus8_128_l3 | mass_tv <= 0.15 | 0.0354004 | 0.0354004 | 0.0354004 | 0.0354004 | 0.0354004 |
| width_last_softplus8_128_l3 | hq >= 0.85 | 0.965576 | 0.93042 | 0.952393 | 0.963379 | 0.973877 |
| width_last_softplus8_128_l3 | component_covariance_error <= 0.85 | 0.628182 | 0.591036 | 0.600358 | 0.571994 | 0.60358 |
| width_last_softplus8_128_l3 | component_min_eigen_ratio >= 0.15 | 0.256169 | 0.269239 | 0.291357 | 0.284831 | 0.295501 |
| width_last_softplus8_128_l3 | All live bounds | PASS | PASS | PASS | PASS | PASS |
| width_raw_softplus128_l3 | sw1_normalized <= 0.18 | 0.0723577 | 0.063386 | 0.0631525 | 0.0704615 | 0.0749721 |
| width_raw_softplus128_l3 | mass_tv <= 0.15 | 0.0446777 | 0.0446777 | 0.0446777 | 0.0446777 | 0.0446777 |
| width_raw_softplus128_l3 | hq >= 0.85 | 0.98999 | 0.973877 | 0.992188 | 0.979492 | 0.987549 |
| width_raw_softplus128_l3 | component_covariance_error <= 0.85 | 0.476961 | 0.51207 | 0.535639 | 0.502241 | 0.489304 |
| width_raw_softplus128_l3 | component_min_eigen_ratio >= 0.15 | 0.112812 | 0.141473 | 0.114085 | 0.120605 | 0.153344 |
| width_raw_softplus128_l3 | All live bounds | FAIL | FAIL | FAIL | FAIL | PASS |
| width_ref_silu160_l2_linear_skip | sw1_normalized <= 0.18 | 0.0724023 | 0.0780486 | 0.0620966 | 0.0408009 | 0.0492018 |
| width_ref_silu160_l2_linear_skip | mass_tv <= 0.15 | 0.0141602 | 0.0141602 | 0.0141602 | 0.0141602 | 0.0141602 |
| width_ref_silu160_l2_linear_skip | hq >= 0.85 | 0.956299 | 0.742676 | 0.861816 | 0.987305 | 0.950928 |
| width_ref_silu160_l2_linear_skip | component_covariance_error <= 0.85 | 0.539462 | 0.551321 | 0.612181 | 0.595886 | 0.48018 |
| width_ref_silu160_l2_linear_skip | component_min_eigen_ratio >= 0.15 | 0.193762 | 0.159001 | 0.119327 | 0.129445 | 0.147984 |
| width_ref_silu160_l2_linear_skip | All live bounds | PASS | FAIL | FAIL | FAIL | FAIL |

These are inspected development cases. Architecture support within a fixed recipe does not establish one universal discriminator or a production default. All raw-input cards reuse the exact `shared_critic_v1` implementation; source changes only add a separate catalog and a wrapper around the canonical architecture runner. Static checks exercised pointwise scoring and active-cap double backprop; the1000× head scaling belongs only to static derivative checks and is never used in training.

[Predeclared study/full cards](study_plan.json) · [Actual recipes, optimizer receipts and all records](index.json) · [Audit](audit.json) · [Static derivative/dedup checks](static_checks.json) · [Screen plan](screen_plan.json) · [Screen log](screen.log) · [Screen exact source archive](screen/source.tar.gz) · [Separate refinement scope](refinement_scope.json) · [Refinement full cards/plan](refinement_plan.json) · [Refinement checks](refinement_checks.json) · [Refinement log](refinement.log) · [Refinement source](refinement/source.tar.gz) · [Final scope-v3](last_refinement_scope.json) · [Final plan](last_refinement_plan.json) · [Final checks](last_refinement_checks.json) · [Final log](last_refinement.log) · [Final exact source](last_refinement/source.tar.gz) · [Rare cross plan](cross_plan.json) · [Rare cross log](cross.log) · [Rare cross source](cross/source.tar.gz).

Reproduce the initial catalog with `python -m benchmarks.transfer_suite.shared_width_search --plan screen_plan.json --output /tmp/new-width-screen`; use `benchmarks.transfer_suite.shared_width_refinement` and `refinement_plan.json` for the second stage, and `benchmarks.transfer_suite.shared_width_last_refinement` with `last_refinement_plan.json` for the third. Run from the integrated checkout; use new output directories.

No generated aggregate or production files were edited. Parent integration should append these indexes to the existing `shared_c6` row after independent replay of any witness.

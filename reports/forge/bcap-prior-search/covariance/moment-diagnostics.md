# Saved full-location moment diagnostics

Every score below is diagnostic only, with **no qualification target**. Candidate and control use the same checkpointed original reference after initializer tensor receipts and prior-law/capacity checks. All live rows are included; EMA state, serving noise and standardization are excluded.

[Full hashes, reference buffers, own-prefix scope and receipts](moment-diagnostics.json). [Readout](readout.md). No model execution, sampling or optimizer update is added.

| Strength | Task | Rows × dimensions | Mean candidate / control | Covariance candidate / control | Total candidate / control | Total ratio | Target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | gaussian1d_smoke | 256 × 2 | 0.000278718566 / 0.0180403795 | 0.000342455693 / 0.0967484713 | 0.000621174229 / 0.114788853 | 0.00541145081 | No gate |
| 0.1 | ae_gan_hold | 12 × 2 | 0.0897750482 / 0.209451035 | 0.00556125073 / 3.95689368 | 0.0953362957 / 4.16634464 | 0.0228824794 | No gate |
| 0.1 | ring16_acquisition | 256 × 4 | 0.000134363276 / 0.000922442996 | 0.106715113 / 0.367877513 | 0.106849477 / 0.368799955 | 0.289722044 | No gate |
| 0.1 | five_word_joint_smoke | 5 × 2 | 3.41363848e-05 / 0.1925717 | 4.28751082e-06 / 0.394806445 | 3.84238956e-05 / 0.587378144 | 6.54159437e-05 | No gate |
| 0.01 | gaussian1d_smoke | 256 × 2 | 0.0162362382 / 0.0180403795 | 0.0438257605 / 0.0967484713 | 0.0600619987 / 0.114788853 | 0.523238949 | No gate |
| 0.01 | ae_gan_hold | 12 × 2 | 0.291462779 / 0.209451035 | 0.37850517 / 3.95689368 | 0.669967949 / 4.16634464 | 0.160804736 | No gate |
| 0.01 | ring16_acquisition | 256 × 4 | 0.00146448752 / 0.000922442996 | 0.344447643 / 0.367877513 | 0.345912129 / 0.368799955 | 0.937939727 | No gate |
| 0.01 | five_word_joint_smoke | 5 × 2 | 6.58529098e-05 / 0.1925717 | 5.61424542e-08 / 0.394806445 | 6.5909051e-05 / 0.587378144 | 0.000112208892 | No gate |
| 0.01 | gaussian1d_stability | 256 × 2 | 0.00424067397 / 0.0445907339 | 0.00194749399 / 0.0809949711 | 0.00618816819 / 0.125585705 | 0.0492744631 | No gate |
| 0.01 | five_word_joint_hold | 5 × 2 | 7.72542844e-05 / 0.226996571 | 2.78980369e-05 / 0.240500137 | 0.000105152321 / 0.467496693 | 0.000224926343 | No gate |
| 0.01 | trajectory | 12 × 4 | 0.00234210002 / 0.088545382 | 0.00108106853 / 196.350998 | 0.00342316856 / 196.439545 | 1.74260664e-05 | No gate |
| 0.01 | residual_student | 12 × 4 | 0.0127979601 / 1.03874326 | 0.00113569386 / 125.662735 | 0.0139336539 / 126.701477 | 0.000109972309 | No gate |
| 0.01 | cover_leftover | 24 × 4 | 0.033434052 / 0.551309645 | 0.579164922 / 351.343842 | 0.612598956 / 351.895142 | 0.00174085653 | No gate |
| 0.01 | mode_hold | 12 × 4 | 0.00600978918 / 0.00236347248 | 0.113018095 / 0.13231276 | 0.119027883 / 0.134676233 | 0.883807634 | No gate |
| 0.01 | vector_two_broad | 256 × 4 | 0.000136366201 / 0.000770483515 | 0.00357070938 / 0.00828926824 | 0.00370707549 / 0.00905975141 | 0.409180708 | No gate |
| 0.01 | vector_unequal_mass | 256 × 4 | 0.0025696815 / 0.0031048297 | 0.0216037538 / 0.0256907195 | 0.0241734348 / 0.0287955496 | 0.839485098 | No gate |
| 0.01 | vector_unequal_width | 256 × 4 | 0.00381466933 / 0.00313957105 | 0.0112828221 / 0.0174823068 | 0.0150974914 / 0.0206218772 | 0.732110434 | No gate |
| 0.01 | vector_anisotropic | 256 × 4 | 0.000193574204 / 0.00156381726 | 0.0139498003 / 0.0338605568 | 0.0141433747 / 0.035424374 | 0.39925546 | No gate |
| 0.01 | vector_overlap | 256 × 4 | 0.000185174082 / 0.00010297374 | 0.00159937027 / 0.00233406667 | 0.00178454432 / 0.00243704044 | 0.732258805 | No gate |
| 0.01 | vector_spiral | 256 × 4 | 8.95923076e-05 / 0.000318130333 | 0.00624773465 / 0.0121191852 | 0.00633732695 / 0.0124373157 | 0.509541377 | No gate |
| 0.01 | img_stripes2 | 32 × 8 | 0.758948326 / 1.35982358 | 0.366120428 / 3.99312091 | 1.12506878 / 5.35294437 | 0.210177559 | No gate |
| 0.01 | img_bars4 | 32 × 8 | 0.71770823 / 1.23987341 | 0.762703776 / 7.28096628 | 1.48041201 / 8.52083969 | 0.173740155 | No gate |
| 0.01 | img_blobs4 | 32 × 8 | 0.471926212 / 0.732938647 | 0.775900424 / 2.69894147 | 1.24782658 / 3.43188 | 0.363598546 | No gate |
| 0.01 | img_intensity2 | 32 × 8 | 0.904618979 / 1.52141106 | 0.0743034482 / 0.223718256 | 0.978922427 / 1.74512935 | 0.560945484 | No gate |
| 0.01 | grid100 | 20000 × 2 | 0.00012212852 / 0.000146498787 | 0.0148976343 / 0.0183174033 | 0.0150197633 / 0.0184639022 | 0.813466358 | No gate |
| 0.01 | rotated100 | 20000 × 2 | 0.000849355594 / 0.000481085764 | 0.00507090054 / 0.00830374099 | 0.00592025602 / 0.00878482684 | 0.673918351 | No gate |
| 0.01 | staggered100 | 20000 × 2 | 0.00266680587 / 0.000331382995 | 0.0152752958 / 0.0114536388 | 0.0179421008 / 0.011785022 | 1.52244949 | No gate |
| 0.001 | gaussian1d_smoke | 256 × 2 | 0.0179971624 / 0.0180403795 | 0.0815411136 / 0.0967484713 | 0.0995382741 / 0.114788853 | 0.867142339 | No gate |
| 0.001 | ae_gan_hold | 12 × 2 | 0.225087821 / 0.209451035 | 1.71372867 / 3.95689368 | 1.93881655 / 4.16634464 | 0.465351936 | No gate |
| 0.001 | ring16_acquisition | 256 × 4 | 0.000169994702 / 0.000922442996 | 0.407621056 / 0.367877513 | 0.407791048 / 0.368799955 | 1.10572424 | No gate |
| 0.001 | five_word_joint_smoke | 5 × 2 | 0.000249676406 / 0.1925717 | 0.000298311352 / 0.394806445 | 0.000547987758 / 0.587378144 | 0.000932938625 | No gate |

Counts are frozen prior capacities and dimensions. Unequal own continuation prefixes remain explicit in each JSON receipt; these rows do not claim equal trained starting weights or equal consumed prefixes. The full-table diagnostic differs from the native selected-row objective.

## Not applicable or unavailable

| Global strength | Task | Recorded gate | Reason |
| --- | --- | --- | --- |
| 0.1 | two_pole | PASS | No certified completed live latent-prior objective |
| 0.1 | unused_token_hold | PASS | No certified completed live latent-prior objective |
| 0.1 | clockfree_audit_measurement_v1 | PASS | No certified completed live latent-prior objective |
| 0.01 | two_pole | PASS | No certified completed live latent-prior objective |
| 0.01 | unused_token_hold | PASS | No certified completed live latent-prior objective |
| 0.01 | unipolar | PASS | No certified completed live latent-prior objective |
| 0.01 | mid_scale_identity | PASS | No certified completed live latent-prior objective |
| 0.01 | clockfree_audit_measurement_v1 | PASS | No certified completed live latent-prior objective |
| 0.001 | two_pole | PASS | No certified completed live latent-prior objective |
| 0.001 | unused_token_hold | PASS | No certified completed live latent-prior objective |
| 0.001 | clockfree_audit_measurement_v1 | PASS | No certified completed live latent-prior objective |

Saved-only reproduction commands require the [covariance feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/18e2aa1018f25de8c447a59287025df6f6aded3e) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

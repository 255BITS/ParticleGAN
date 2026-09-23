# Smooth discriminator refinement, original recipe

An adaptive six-card follow-up to the previously observed Softplus β5 / D64×2 near-miss. The six candidates and selection rule were frozen before these episodes. All 24 new attempts, including failures, are retained. These are inspected development tasks; this is architecture search, not independent validation.

Every episode uses the original shared recipe: Rp logistic, b_cap coefficient 3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. D alone changes: axis Fourier2, Softplus β, width and depth. Budgets remain 1200 updates except spiral1600. Seed0 and one CPU thread; no seed sweeps.

PASS requires unchanged live bounds and five final passing observations from all24 scheduled checkpoints. `FAIL(n)` displays the final passing suffix length. EMA is recorded separately and never determines success. A missing case is unrun, not a failure or pass.

| Discriminator | D params | Rare mass | Unequal width | Overlap | Two broad | Anisotropic | Spiral | Sustained / attempted | Seconds |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| softplus3_d64_l2_f2 | 4929 | [FAIL(0)](episodes/softplus3_d64_l2_f2__vector_unequal_mass.json.gz) | [FAIL(0)](episodes/softplus3_d64_l2_f2__vector_unequal_width.json.gz) | [FAIL(0)](episodes/softplus3_d64_l2_f2__vector_overlap.json.gz) | unrun | unrun | unrun | 0/3 | 20.50 |
| softplus10_d64_l2_f2 | 4929 | [FAIL(0)](episodes/softplus10_d64_l2_f2__vector_unequal_mass.json.gz) | [PASS(8)](episodes/softplus10_d64_l2_f2__vector_unequal_width.json.gz) | [FAIL(1)](episodes/softplus10_d64_l2_f2__vector_overlap.json.gz) | [PASS(11)](episodes/softplus10_d64_l2_f2__vector_two_broad.json.gz) | [FAIL(0)](episodes/softplus10_d64_l2_f2__vector_anisotropic.json.gz) | [PASS(22)](episodes/softplus10_d64_l2_f2__vector_spiral.json.gz) | 3/6 | 41.88 |
| softplus5_d96_l2_f2 | 10465 | [FAIL(0)](episodes/softplus5_d96_l2_f2__vector_unequal_mass.json.gz) | [FAIL(0)](episodes/softplus5_d96_l2_f2__vector_unequal_width.json.gz) | [FAIL(2)](episodes/softplus5_d96_l2_f2__vector_overlap.json.gz) | unrun | unrun | unrun | 0/3 | 22.44 |
| softplus5_d128_l2_f2 | 18049 | [FAIL(0)](episodes/softplus5_d128_l2_f2__vector_unequal_mass.json.gz) | [FAIL(0)](episodes/softplus5_d128_l2_f2__vector_unequal_width.json.gz) | [FAIL(0)](episodes/softplus5_d128_l2_f2__vector_overlap.json.gz) | unrun | unrun | unrun | 0/3 | 24.42 |
| softplus5_d64_l3_f2 | 9089 | [FAIL(0)](episodes/softplus5_d64_l3_f2__vector_unequal_mass.json.gz) | [FAIL(0)](episodes/softplus5_d64_l3_f2__vector_unequal_width.json.gz) | [FAIL(1)](episodes/softplus5_d64_l3_f2__vector_overlap.json.gz) | unrun | unrun | unrun | 0/3 | 24.76 |
| softplus5_d128_l3_f2 | 34561 | [FAIL(0)](episodes/softplus5_d128_l3_f2__vector_unequal_mass.json.gz) | [FAIL(0)](episodes/softplus5_d128_l3_f2__vector_unequal_width.json.gz) | [FAIL(4)](episodes/softplus5_d128_l3_f2__vector_overlap.json.gz) | [PASS(24)](episodes/softplus5_d128_l3_f2__vector_two_broad.json.gz) | [FAIL(0)](episodes/softplus5_d128_l3_f2__vector_anisotropic.json.gz) | [PASS(24)](episodes/softplus5_d128_l3_f2__vector_spiral.json.gz) | 2/6 | 73.94 |

No new rare-mass sustained witness was found in this bounded refinement.

Different discriminator architectures may support different cases under the same recipe. This table does not claim a single discriminator solves all data tests. Timings are observed serial CPU wall times including metrics, and are not a controlled speed comparison.

**Finalist rule:** Select two by rare-mass sustained PASS first, then total hard sustained passes, then lower mean final normalized bound shortfall, then confirmation fraction, then name. Complete other three data cases for selected cards; never change their settings per case.

**Selected:** softplus10_d64_l2_f2, softplus5_d128_l3_f2.

| Candidate | Task | Final live failing bounds |
| --- | --- | --- |
| softplus3_d64_l2_f2 | unequal_mass | component_covariance_error 7.19761 (bound 0.85), component_min_eigen_ratio 0.0123832 (bound 0.15) |
| softplus3_d64_l2_f2 | unequal_width | component_covariance_error 17.176 (bound 0.85) |
| softplus3_d64_l2_f2 | overlap | sw1_normalized 0.195161 (bound 0.18), mean_error 0.276143 (bound 0.15) |
| softplus10_d64_l2_f2 | unequal_mass | component_covariance_error 3.37722 (bound 0.85), component_min_eigen_ratio 0.00985435 (bound 0.15) |
| softplus10_d64_l2_f2 | unequal_width | none; sustained PASS |
| softplus10_d64_l2_f2 | overlap | none; fewer than five final passing observations |
| softplus5_d96_l2_f2 | unequal_mass | component_min_eigen_ratio 0.141018 (bound 0.15) |
| softplus5_d96_l2_f2 | unequal_width | component_covariance_error 5.32365 (bound 0.85) |
| softplus5_d96_l2_f2 | overlap | none; fewer than five final passing observations |
| softplus5_d128_l2_f2 | unequal_mass | component_min_eigen_ratio 0.00768049 (bound 0.15) |
| softplus5_d128_l2_f2 | unequal_width | component_covariance_error 5.33184 (bound 0.85) |
| softplus5_d128_l2_f2 | overlap | sw1_normalized 0.189691 (bound 0.18), mean_error 0.211615 (bound 0.15), covariance_error 0.545962 (bound 0.45) |
| softplus5_d64_l3_f2 | unequal_mass | component_min_eigen_ratio 0.0122125 (bound 0.15) |
| softplus5_d64_l3_f2 | unequal_width | component_covariance_error 2.20775 (bound 0.85) |
| softplus5_d64_l3_f2 | overlap | none; fewer than five final passing observations |
| softplus5_d128_l3_f2 | unequal_mass | component_min_eigen_ratio 0.0436551 (bound 0.15) |
| softplus5_d128_l3_f2 | unequal_width | component_covariance_error 1.40536 (bound 0.85), component_min_eigen_ratio 0.126579 (bound 0.15) |
| softplus5_d128_l3_f2 | overlap | none; fewer than five final passing observations |
| softplus10_d64_l2_f2 | two_broad | none; sustained PASS |
| softplus10_d64_l2_f2 | anisotropic | sw1_normalized 0.19175 (bound 0.18), mass_tv 0.202799 (bound 0.15) |
| softplus10_d64_l2_f2 | spiral | none; sustained PASS |
| softplus5_d128_l3_f2 | two_broad | none; sustained PASS |
| softplus5_d128_l3_f2 | anisotropic | component_covariance_error 1.25917 (bound 0.85) |
| softplus5_d128_l3_f2 | spiral | none; sustained PASS |

[Frozen plan](plan.json.gz) · [Exact task specs](task_specs.json.gz) · [Resolved episodes, metrics, EMA and hashes](index.json.gz) · [Leader cells](leaderboard.json.gz) · [Finalist selection](selection.json.gz) · [Audit](audit.json.gz) · [Static copy/gradient parity checks](architecture_checks.json.gz) · [Runtime and source hashes](protocol.json.gz) · [Exact numerical source](source.tar.gz) · [Driver](run.py) · [Tailable execution log](run.log).

Source implementation was copied byte-for-byte from the smooth-D study (SHA256 `9958f52f265c6ecf37de1449afc02d25db916f7c05879017c33025757f81bfcc`). All six constructed models matched that source exactly in initial states, outputs, input gradients, cap penalty, and parameter gradients under the same seed0. This is static copy parity, not a rerun of the previous training episode.

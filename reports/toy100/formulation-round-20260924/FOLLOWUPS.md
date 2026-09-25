# Direct measured follow-ups

Every formulation earns its own passes. Full 22 and own-state stability have not
run for these candidates. The Ra hybrid and symmetric b-cap both pass the initial
two blockers, then fail trajectory. The hybrid also fails image intensity.

| Candidate | Gate | Result | Passing suffix |
|---|---|---|---:|
| ra_r1_fake_cap | img_intensity2 | FAIL | 0 |
| ra_r1_fake_cap | trajectory | FAIL | 0 |
| ra_r1_fake_cap | mode_hold | PASS | 6 |
| ra_r1_fake_cap | vector_unequal_mass | PASS | 8 |
| ra_r1r2_exposure_mean | mode_hold | FAIL | 0 |
| ra_r1r2_exposure_mean | vector_unequal_mass | PASS | 5 |
| ra_r1r2_shared_coordinate | mode_hold | FAIL | 1 |
| ra_r1r2_shared_coordinate | vector_unequal_mass | FAIL | 0 |
| bcap_symmetric_half | trajectory | FAIL | 0 |
| bcap_symmetric_half | mode_hold | PASS | 7 |
| bcap_symmetric_half | vector_unequal_mass | PASS | 9 |
| bcap_real_half_fake_one | mode_hold | FAIL | 0 |
| bcap_real_half_fake_one | vector_unequal_mass | PASS | 20 |
| rp_bcap_half | mode_hold | FAIL | 0 |
| rp_bcap_half | vector_unequal_mass | PASS | 6 |
| rp_r1_fake_cap | trajectory | FAIL | 0 |
| rp_r1_fake_cap | mode_hold | PASS | 13 |
| rp_r1_fake_cap | vector_unequal_mass | PASS | 5 |

Original follow-up audit errors are retained. The check initially rejected allowed
recipe fields inside the specification; it now constructs the exact expected spec
from the declared recipe and unchanged archived harness. No scoring, data or budget
was changed and no training rerun was required. [Independent audit](followup-audit.json).

R1-containing candidates remain eligible when supported by measured results. The
hybrid stopped because it failed transfer regressions, not because of its name.
The latest objective ablations restore the original Rp loss to the two candidates
that passed both blockers with Ra but failed trajectory. These are new measured
candidates; their passes cannot be inherited from the Ra counterparts.

All seven direct proposals have completed: 18 training gates, 9 PASS and 9 FAIL.
The Rp hybrid also passes both blockers but fails trajectory (MSE .480884).
No candidate reaches full-22 eligibility. Bars/blobs and later tests are NOT_RUN.

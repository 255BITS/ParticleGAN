# Saved scientific calibration: independent audit

**Audit PASS; profile adoption BLOCKED and exact-profile feasibility INFEASIBLE.** Zero training launched. The saved reduction was reproduced exactly; all seven receipts passed certificate, raw hash, registration, complete-cohort and frozen-evaluator checks.

The unchanged profile has **7/57 measured cells, 50 unknown**, three PASS cells and four scientific FAIL cells, costing **195.525508143 paid seconds** on NVIDIA RTX A6000 CUDA. FLOPs remain unavailable.

| Lineage | Smoke | Independent reference | Classification | Measured paid seconds |
| --- | --- | --- | --- | ---: |
| k3p | FAIL | FAIL | true_reject | 156.283765995 |
| forge-onboarding-anchor-ablation | FAIL | UNKNOWN | unknown | 17.569934434 |
| forge-no-critic-penalty | FAIL | UNKNOWN | unknown | 21.671807714 |

K3P supplies one paired true rejection. The two control reference labels remain UNKNOWN. Empirical false-accept fraction is 0/1; empirical false-reject fraction is **unknown**, because no independent positive exists. Missing measurements and costs remain unknown.

Every declared lineage already fails smoke. With no eventual positive the frozen minimum of one positive fails; with any eventual positive, every positive is falsely rejected and the zero false-reject requirement fails. Therefore completing this matrix cannot approve this profile. This is a feasibility proof for its exact declarations, not a population estimate.

| Exact receipt | Task | Verdict | Paid seconds |
| --- | --- | --- | ---: |
| [49d41e28](../attempts/49d41e284fde41698212677403ca5d67/result.json) | mode_hold | FAIL | 17.930310757 |
| [5693d7b9](../attempts/5693d7b9c35844c78f4440ed15bad4a2/result.json) | img_intensity2_residual16 | PASS | 11.143371592 |
| [680dcd33](../attempts/680dcd337be343f69a1922cd3e89750a/result.json) | img_bars4_residual16 | PASS | 11.170116852 |
| [a1de5d14](../attempts/a1de5d14bd8b4640b95a826209ae3085/result.json) | vector_unequal_mass_published | PASS | 24.002935348 |
| [dca8c7aa](../attempts/dca8c7aa0eca484c9270d07125f7cb0b/result.json) | grid100_affine_square_named_v1 | FAIL | 92.037031446 |
| [eac540cc](../attempts/eac540ccfe564a5ab396abdd545ab013/result.json) | mode_hold | FAIL | 17.569934434 |
| [6d10bbf9](../attempts/6d10bbf90f1b4233a73034391599ecdb/result.json) | mode_hold | FAIL | 21.671807714 |

Source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`; cohort `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`; profile SHA `a6b6d2a1f9d875b9548d94352c642e784427d58490c1959c12b08375f2a913e3`; criteria SHA `6eac2531eed68e392c11a57328999ac5dce509f8fe51d23e7b9031b7a4e3f9d1`.

The complete frozen source snapshot, native 45-file manifest and checkpoint file hash verified. Each reproduced grade exactly matches its frozen grader certificate. The machine-readable audit binds all request/result/evidence hashes, exact revisions, compatibility keys, actual updates, phase timing and measured memory.

Only K3P has complete smoke cost (40.243799201 seconds); reference costs and both controls’ complete smoke costs remain unavailable. The separate A2-off study is excluded. Transfer receipts establish declared initializer/RNG parity and recorded zero unintended deviations, but save no initial tensors or final RNG state for stronger equality claims.

**Recommendation:** stop filling this profile for adoption. Preserve all failures and retain the independent reference purposes and frozen thresholds. Register a justified new screen with a bounded study capable of establishing a compatible positive reference. Accepted calibration and a finished candidate are both absent, so production promotion has no eligible run.

[Machine-readable checks](SCIENTIFIC_CALIBRATION_SAVED_AUDIT.json).

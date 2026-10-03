# Independent A2-off native audit

**A2-off is a negative result on this fixed-seed cell. Both formulations fail; disabling A2 also loses coverage. No integrity discrepancy was found.**

Registration `host-profile-a2-off-native-v1` is byte-identical to the file published at `043ec64e`. Request `336453aafdc3d896335e60e0`; new attempt `d4d633052321469e82caff90210ef309`; baseline `dca8c7aa0eca484c9270d07125f7cb0b`. Both bind source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7` and the same materialized grid task.

| Metric | Baseline A2=.5 | A2=0 |
|---|---:|---:|
| Full verdict | FAIL | FAIL |
| Coverage gate | PASS | FAIL |
| Final modes | 100 | 74 |
| Final HQ | 0.99525 | 0.99725 |
| Final mass TV | 0.04355 | 0.23675 |
| 100k holdout mass TV | 0.032000 | 0.233540 |
| 100k holdout center RMS / sigma | 0.089885 | 0.125088 |
| 100k holdout covariance-trace bias | -0.265594 | -0.426600 |
| 100k holdout radial KS | 0.120006 | 0.210772 |
| Actual A2 applications | 6999 | 0 |
| Charged seconds | 92.037031446 | 85.367950892 |

A2-off never passed either the full coverage or accuracy gate at a recorded checkpoint. Baseline achieved accuracy earlier but failed all five final checks. Both independent holdouts and both EMA holdouts fail; oracle controls pass. Higher A2-off HQ does not compensate for missing modes and tighter components. The existing absolute covariance-bias limit is .10 and radial-KS limit is .04.

| Terminal step | Baseline covariance bias | A2-off covariance bias | Baseline radial KS | A2-off radial KS |
|---|---:|---:|---:|---:|
| 6000 | -0.132836 | -0.284922 | 0.054539 | 0.127703 |
| 6250 | -0.194716 | -0.340132 | 0.083221 | 0.164724 |
| 6500 | -0.227224 | -0.378215 | 0.102181 | 0.186793 |
| 6750 | -0.261674 | -0.417567 | 0.114544 | 0.203730 |
| 7000 | -0.273573 | -0.434388 | 0.126870 | 0.216704 |

Pairing checks passed: identical task, source files, runtime, compute, protocol, prior, host architecture and sampling law. The effective recipe differs only in `latent_damping_max_rate: .5 -> 0`; the candidate removes the required `a2` capability. All complete initialization receipts match, including 11 initial parameter hashes, per-parameter initializer RNG receipts and whole initial G/D state hashes. Saved step-zero live, EMA and target arrays are elementwise identical. All 19 named RNG bindings and final checkpoint stream states match exactly, and the 100k holdout target arrays match. The source provenance `origin_commit` differs because the registrations were made at different repository commits; the entire scientific file manifest and its digest are unchanged.

Verified both exact frozen evaluator grades, immutable registration, raw/durable/scientific hashes, both 45-file artifact manifests, checkpoint file/content hashes, complete 7,000-step optimizer histories, and consistency between trainer and named RNG state. Both receipts report zero unintended RNG deviations. A2-off explicitly reports requested=false, enabled=false and applied=0; baseline reports 6,999 actual applications. Learned MoG sigma remains .025, with learned locations, fixed width/uniform masses and no standardization.

Paid attempt costs are 92.037031446s baseline and 85.367950892s A2-off. Synchronized optimizer-phase measurements are 73.123814s and 69.102893s. Wall time includes setup/evaluation; this single comparison does not establish a general speedup, and FLOPs remain unavailable. The A2-off campaign charge matches its receipt and has zero reserved seconds.

This paired result provides no reason to promote A2-off or extend its failed parent. It does not establish a population effect or identify the complete cause of the baseline failure. No training, source, queue, report or lifecycle mutation was performed. Only `/tmp` audit files were written; the eight watched queue/registration/durable-receipt files remained byte-identical.

# A2-off native diagnostic: reject the ablation

The [registered one-factor diagnostic](studies/A2_OFF_NATIVE_DIAGNOSTIC.md)
completed all 7,000 updates for **85.367950892 paid seconds** and failed. Removing
A2 reduced final mode coverage from 100 to 74 and worsened mass balance and shape
accuracy. Keep A2 in the reference; this run does not support removing it to
repair the reference's late contraction. Both formulations still fail full gates.

| Native grid metric | K3P control | A2 off | Required |
| --- | ---: | ---: | ---: |
| Full verdict | FAIL | FAIL | PASS |
| Coverage verdict | PASS | FAIL | PASS |
| Final modes | 100 | 74 | 100 |
| Passing terminal checks | 0/5 | 0/5 | 5/5 |
| 100k live holdout covariance-trace bias | −0.265594 | −0.426600 | absolute value ≤0.10 |
| 100k live holdout radial KS | 0.120006 | 0.210772 | ≤0.04 |
| 100k live holdout mass TV | 0.032000 | 0.233540 | ≤0.06 |
| 100k live holdout centre RMS / σ | 0.089885 | 0.125088 | ≤0.20 |
| 100k live holdout HQ | 0.99517 | 0.99739 | coverage and accuracy jointly required |
| Original paid seconds | 92.037031 | 85.367951 | measured cost, not FLOPs |

Higher HQ does not offset missing modes or shape failure. EMA holdouts also fail;
both oracle controls pass. The A2-off final live bias is −0.434388 and radial KS
0.216704. All five terminal checks from 6,000 through 7,000 fail. This single
fixed-seed comparison supports rejecting this ablation here, not a population
effect estimate or proof that A2 resolves the remaining baseline failure.

The only resolved recipe change is `latent_damping_max_rate: 0.5 → 0`, with the
required `a2` capability removed intentionally. The original source, task,
named seed 0, initializer, learned-MoG sigma `.025`, live sampling and thresholds
are unchanged. The disabled mechanism is recorded explicitly. The original
control was imported with its certified hashes and cost; it was not rerun.

The [independent audit](A2_OFF_INDEPENDENT_AUDIT.md) reproduced the frozen grade
and verified the native artifacts/checkpoint. All 11 recorded initial parameter
hashes, whole G/D initial states, actual step-zero live/EMA/target arrays and all
19 final named RNG states match the control exactly. Recorded unintended RNG
deviations are zero; actual A2 applications are zero. There are no execution
errors or reservations. The exact one-attempt revision is now concluded.

Evidence:

- [A2-off receipt](attempts/d4d633052321469e82caff90210ef309/result.json),
  request `336453aafdc3d896335e60e0`, candidate revision
  `8493b15ff2d0f677de94a07cf344762a1d9efd58c609987fff7dab311cad8581`.
- [Original control](attempts/dca8c7aa0eca484c9270d07125f7cb0b/result.json).
- [Registration](calibration-lanes/host-profile-a2-off-native-v1/registration.json),
  SHA `29c0f420821f0e38bc1b6fbd216f0cd43cfaa3ae95bd16600b2f518318f9ea9d`,
  published at `043ec64e` before submission and execution.
- Scientific source
  `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`.

The [separate diagnostic matrix](calibration/host-profile-a2-off-v1.md) contains
two measured reference negatives and 36 unknown cells, with no receipt issues.
It imports only the selected native control; the original smoke pairing remains
in its original profile. Missing costs and smoke outcomes stay unknown. No
positive reference, qualification or accepted calibration is established.

Do not extend this failed parent, repeat the control, or run a seed/width sweep.
Preserve this negative in compiled memory. Further calibration requires a
supported new hypothesis or compatible saved positive evidence; finishing the
engine does not imply that a production-quality formulation has been found.

# Scientific calibration: stop the training-noise-removal study

The preregistered one-cell study completed with **scientific FAIL**, for
**93.746687201 new paid seconds** on GPU 0, NVIDIA RTX A6000. Removing output
noise from both training objectives improved clean shape metrics but did not
produce a useful full-native positive. No further training was launched. GPU 1's
unrelated work was preserved. Calibration and production promotion remain blocked.

This executes the handoff's bounded Phase 1 study; its stopping rule permits a
scientifically meaningful failure. The source, all three smoke gates, all 16
independent reference purposes, horizons and acceptance thresholds were retained.
There was no seed, width, amplitude or best-checkpoint search.

## Native comparison and cost-aware leaderboard

All rows use the same named affine grid task, source, learned-MoG sigma .025,
clean-live scoring, seed 0 and full 7k gates. They are distinct exact candidate
revisions. **All full verdicts FAIL; no winner qualifies.** Table order is a
control/ablation comparison, not an undeclared aggregate ranking.

| Native formulation | Full / sustained coverage | Final modes | Joint terminal passes | 100k live centre RMS / σ | Signed covariance bias | Radial KS | Mass TV | Original paid seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K3P control, reused | FAIL / PASS | 100 | 0/5 | .089885 | −.265594 | .120006 | .032000 | 92.037031 |
| Training output noise off, new | FAIL / FAIL | 100 | 0/5 | .188341 | −.163005 | .067130 | .031700 | 93.746687 |
| A2 off, separate saved study | FAIL / FAIL | 74 | 0/5 | .125088 | −.426600 | .210772 | .233540 | 85.367951 |
| Frozen accuracy limit | PASS required | 100 | 5/5 | ≤.20 | absolute value ≤.10 | ≤.04 | ≤.06 | Measured, not FLOPs |

Noise removal reduces the covariance and radial deficits while worsening centre
accuracy and sustained coverage. This supports an effect of the changed training
law on this selected host; it does not establish output noise as the sole cause
of contraction or noise removal as a sufficient repair. Keep A2: its already
completed removal substantially worsened coverage. No historical cloud, noisy
served-model or EMA pass supplies current clean-live credit.

The new terminal live checks at 6,000/6,250/6,500/6,750/7,000 all fail the joint
accuracy gate. At 6,000 and 6,750 the coverage gate additionally fails maximum
covariance eigenvalue ratios **1.877673 and 2.274458**, above 1.7. Only one
consecutive coverage observation passes at the end; five are required. Final
100-mode coverage and HQ .98225 cannot replace sustained quality. Final live
covariance bias is −.164550 and radial KS .069688. The independent live holdout
fails both shape limits; its centre and mass limits pass. EMA holdout also fails
(bias −.167400, KS .070931), while the oracle passes. No serving-policy switch
is justified.

| Compute measurement | K3P control | Output noise off |
| --- | ---: | ---: |
| Completed G / D / prior optimizer updates | 7,000 / 7,000 / 7,000 | 7,000 / 7,000 / 7,000 |
| Synchronized optimizer-phase seconds | 73.123814 | 77.773669 |
| Evaluation seconds / calls | 4.583560 / 35 | 5.022525 / 35 |
| Sampling seconds / calls | .077027 / 70 | .097942 / 70 |
| Peak PyTorch allocated bytes | 125,372,416 | 125,364,736 |
| Peak PyTorch reserved bytes | 192,937,984 | 192,937,984 |
| Process-lifetime peak RSS bytes | 2,163,372,032 | 2,162,356,224 |

Paid wall time includes setup and evaluation. Phase coverage is partial; allocator
peaks exclude other processes and non-PyTorch allocations. These measurements
do not establish a speed advantage or total-device memory. FLOPs are unavailable.
Exactly one new reservation used the 3,600-second ceiling; its unused allowance
is not permission for another run. No error, retry or repair occurred.

## Registration, identity and independent validation

[The protocol](studies/NO_OUTPUT_NOISE_REFERENCE_V1.md) and
[immutable registration](calibration-lanes/no-output-noise-native-v1/registration.json)
were committed at `0959af91` before submission. Registration SHA is
`ea15bebb965f0c12fa15c647072cabe10669d5bd07e18c30d491c15b8c38a1dd`;
request `37ceb438a71a1ac65bdea9b5`,
[attempt `180af0fc`](attempts/180af0fcfee447b697d5b5d05dc2877f/result.json),
exact revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603`.
Only `output_noise_std: .029 → 0` changes. The mechanism is an objective
convolution removal implemented through an existing public scalar field, not a
tuned amplitude or sampling-only patch. A2 applies on 6,999 updates. The critic
guard has zero actual activations; its synthetic component check is not training
activation evidence.

Source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`
and resolver-derived cohort
`eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`
remain identical across all four lineages. Plain Python 3.14.7 supplies the
compatible runtime. `origin/develop` remains the already integrated `a8b9d397`;
no incoming E22 API change required integration.

[Independent audit](studies/NO_OUTPUT_NOISE_INDEPENDENT_AUDIT.md) passes integrity
checks and reproduces the scientific FAIL from the frozen evaluators. It verifies
the source, registrations, 45-file native artifact manifest, complete checkpoint
and optimizer states. All 11 initial parameter hashes, whole G/D states and
step-zero live/EMA/target arrays match the control. Eighteen of 19 final named RNG
states match; only generator-output noise differs intentionally, with its disabled
stream still at its initial state. All recorded unintended RNG deviations are zero.
No images were inspected as decision evidence.

## Calibration result and promotion readiness

The old profile remains **7/57 measured, 50 unknown**, costing 195.525508143
seconds; [its independent replay](studies/SCIENTIFIC_CALIBRATION_SAVED_AUDIT.md)
reproduces the original infeasibility finding. Seven explicit verified import
bindings reuse those exact receipts without another launch or charge.

The [new four-lineage matrix](calibration/no-output-noise-reference-v1.md) contains
**8/76 measured cells, 68 unknown**: three PASS cells and five scientific FAIL
cells, with **289.272195344 seconds of unique recorded cost**, including the
seven reused receipts once. The separate A2-off receipt appears in the native
comparison above but is outside this profile's denominator and cost subtotal.

| Lineage | Smoke | Independent reference | Measured cells |
| --- | --- | --- | ---: |
| K3P | FAIL | FAIL | 5/19 |
| Anchor off | FAIL | UNKNOWN | 1/19 |
| No critic penalty | FAIL | UNKNOWN | 1/19 |
| Training output noise off | UNKNOWN | FAIL | 1/19 |

This exact new roster is **infeasible for acceptance**: every lineage already has
either a smoke failure or a reference failure, ruling out any true acceptance.
Any eventual reference positive would therefore be a false rejection; without a
positive, the required minimum is unmet. Do not fill this matrix for approval.
Two independent reference negatives are now known, but only K3P is paired with
a measured smoke decision. The paired fraction is 1/4. False accepts are 0/1
paired negatives; empirical false rejection remains unknown. No missing cell is
recoded as FAIL. Complete reference costs and smoke/reference ratios remain
unknown. These selected lineages do not estimate population gate accuracy.

[Machine-readable readiness](SCIENTIFIC_CALIBRATION_PROMOTION_READINESS.json)
records the exact unmet promotion prerequisites:

- Accepted receipt-backed calibration: **BLOCKED**, with no complete positive.
- Finished ordinary smoke/quality/endurance qualification: **FAIL/UNKNOWN**.
  The new native diagnostic fails; three smoke and 15 other reference cells are
  unmeasured for this lineage. Diagnostics grant no ordinary qualification.
- Concluded exact-revision readout: **PASS**, preserving the negative result in
  [Forge's immutable record](records/readout-abd06840fa1f42a4f64c13ab.json).
- Frozen formulation and scoring: **PASS**; no new public API or serving change.
- Matching production application evidence: **BLOCKED**, absent.

No robustness stage or seed trials were registered, and no public default is
recommended. Calibration alone would still not establish production readiness.

The completed [saved-geometry diagnosis](studies/NO_OUTPUT_NOISE_SAVED_GEOMETRY.md)
uses both final checkpoints and existing live arrays, with zero training or new
random draws. Analytic transformed kernel variance stays nearly unchanged:
.672576 → .677400 of target variance. Learned-location spread increases unevenly;
76/100 new holdout modes remain below .90 of radius-conditioned target variance.
The lowest-variance 50 modes hold only 9.72% of summed location variance, while
the highest-variance 10 hold 35.28%. Centre displacement also increases. These
descriptive metrics suggest heterogeneous row spread, not global affine/kernel
expansion. Untruncated kernel-plus-location covariance is not a decomposition of
the gate's radius-conditioned covariance; no alternate pass is inferred.

Stop this exact noise-removal candidate and both infeasible profiles. Preserve
all receipts and failed ideas in the compiled memory. The smallest next bounded
measurement is a zero-training diagnostic of deterministic critic radial gradients
or public prior-update residual moments on these exact saved final states: does
the critic/prior response concentrate most rows while allowing a minority to
broaden or drift? Review that evidence before declaring any new mechanism.
A further training stage needs a supported substantive hypothesis and its own
bounded preregistration. No smoke completion, failed-parent 14k extension,
seed sweep, width/affine-scale tuning or noise-amplitude search follows.

[Full comparison, costs and feasibility proof](studies/no-output-noise-v1-comparison.json)
and [compiled experiment memory](EXPERIMENT_MEMORY.md) retain the searchable
decision record. Central logs:
`/home/martyn/dev/ParticleGAN/runs/forge/events.jsonl`; attempt log:
`/home/martyn/dev/ParticleGAN/runs/forge/calibration-no-output-noise-native-v1/180af0fcfee447b697d5b5d05dc2877f/run.log`.

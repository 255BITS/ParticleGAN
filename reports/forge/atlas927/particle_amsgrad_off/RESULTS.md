# Table-only AMSGrad ablation: FAIL

The single new variant 927 case failed the unchanged original two-pole gates.
Its graded mean absolute position rose from 0.002445647 to
0.015023150, below 0.30. This is a distinct optimizer-law variant.
Noise/critic AMSGrad and generator defaults stay enabled. The corrected
completed-Adam LR clock, q, betas/gains, seed 0, data, original 80 updates,
24 reads and final five-check rule are preserved.

| Original scored gate | Completed 921 | Table AMSGrad off 927 | Requirement |
| --- | ---: | ---: | --- |
| mean_abs | 0.002445647 FAIL | 0.015023150 FAIL | >=0.30 |
| grad_med | 0.010896996 PASS | 0.009985171 PASS | <=1 |
| Full result | FAIL | FAIL | All gates and final5 |
| Passing observations | 0/24 | 0/24 | Stable final5 |

## Retained motion and spread

Comparable before/after Adam positions cover updates 1–65.
Both recorder traces are UNKNOWN/incomplete; all 80 training updates and 24
scoring reads completed. The arithmetic below uses only the common prefix.

| Ungated diagnostic | Baseline 921 | Variant 927 |
| --- | ---: | ---: |
| Cumulative mean per-row absolute Adam travel | 0.059611719 | 0.164043810 |
| Net mean per-row absolute Adam displacement | 0.007175425 | 0.011099621 |
| Adam cancellation fraction | 87.963% | 93.234% |
| Common translation share of Adam travel | 96.707% | 88.067% |
| Final step 80 particle range | 0.009765730 | 0.055783669 |
| Final step 80 population standard deviation | 0.002719837 | 0.016676383 |
| Final step 80 negative/positive rows | 5/7 | 4/8 |
| Final step 80 rows within 0.1 of either target pole | 0/12 | 0/12 |

[comparison.json](comparison.json) includes all 24 observations, per-read spread,
sign-centroid separation (null when one side is absent), and recorded motion
deltas. These diagnostics add no gate. In this fixed case, the higher recorded travel supports stale-max damping
as a contributing factor. The remaining failure persists: 93.23% of recorded
Adam travel cancels and the final particle cloud stays near zero. This is an
inference from the single controlled comparison, not a general family result.

## Reproduction and verification

Maintained Forge API candidate `atlas-two-pole-particle-amsgrad-off927-v1`,
view `atlas_two_pole_particle_amsgrad_off927_v1`, READY Study
`atlas-two-pole-particle-amsgrad-off927-study-v1`, CPU backend, through tier 1.
Original Task digest: `2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b`.
Base 79-rule Recipe provenance: `d5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95`.
Distinct effective optimizer variant:
`950edc64094c417d98d43b2eaf59fbaaba0524f7823e2acb001ba7576db1d051`.
The historical 921 control is metadata only: factory_runnable=false, with no
result or grade transfer on this new Source.

Eight actual focused variant checks and four pure declaration checks passed
on one CPU thread and no GPUs. Retained 916 recorder/checkpoint controls and completed 921 controls
were not rerun and keep their original producer provenance. Source-only reviews
are separately labeled. A fresh 19-field proof joins the full Queue-added
request, copied Source, actual controlled files, original Task and variant.

Request `ea7dabbbd6bd9fa1781dcd7a`, attempt
`05d5b90ed0334a7d835582b8a43b8194`, Source
`a515e30f7de19dc84be907522b653a6fb9b1991b7cde48159ef680a5cb446b91`, revision
`9b0d1cb654e208162e8e641feb353bbf97536216df3f7c463869dfff06a17e5b`.
The full 300-second one CPU thread and no GPUs allowance remains conservatively charged.
Measured matched parent span: 20.862499s;
no refund. The existing 916 metadata allowance of 120 seconds continues. Earlier INVALIDs, failures,
Root reporting errors and costs remain preserved once. Exact inclusive costs,
persistence/publication tails and Source-only agent time remain qualified
UNKNOWN_NOT_ZERO or UNMEASURED.

[RESULTS.json](RESULTS.json), [software-proof.json](software-proof.json),
[graded-result.json](graded-result.json), [raw-result.json.gz](raw-result.json.gz)
and [artifact-pins.json](artifact-pins.json) contain the verification chain.
This single fixed-seed result leaves the original FAIL intact. No second case,
baseline rerun, merge/default, family winner or speed claim follows.

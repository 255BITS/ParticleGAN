# Original frozen CUDA screens

State: **FAIL**. Acceptance counts: portability {'FAIL': 4, 'PASS': 9}; native {'FAIL': 3}.

Primary verdicts are copied from original result.json. Validity is recorded separately; source, stream, canonical fixture, runtime or evidence errors cannot become accepted quality results.

| Task | Original verdict | Fixture/evidence | Accepted verdict | Steps | Peak reserved MiB |
|---|---|---|---|---:|---:|
| mode_hold | FAIL | VALID | FAIL | 1200 | 88.0 |
| img_intensity2 | PASS | VALID | PASS | 600 | 86.0 |
| img_blobs4 | FAIL | VALID | FAIL | 600 | 86.0 |
| img_stripes2 | PASS | VALID | PASS | 600 | 86.0 |
| img_bars4 | PASS | VALID | PASS | 600 | 86.0 |
| vector_two_broad | PASS | VALID | PASS | 1200 | 90.0 |
| vector_unequal_mass | FAIL | VALID | FAIL | 1200 | 92.0 |
| vector_unequal_width | PASS | VALID | PASS | 1200 | 92.0 |
| vector_anisotropic | PASS | VALID | PASS | 1200 | 90.0 |
| vector_overlap | PASS | VALID | PASS | 1200 | 90.0 |
| vector_spiral | PASS | VALID | PASS | 1600 | 90.0 |
| ring_shift | FAIL | VALID | FAIL | 4600 | 188.0 |
| stationary | PASS | VALID | PASS | 7500 | 188.0 |
| grid100 | FAIL | VALID | FAIL | 7000 | 262.0 |
| rotated100 | FAIL | VALID | FAIL | 7000 | 262.0 |
| staggered100 | FAIL | VALID | FAIL | 7000 | 262.0 |

The original CUDA screen, its original hosts and scorers, and the frozen candidate are unchanged. Each native run uses seed 1234, N=20000, z=2, batch 2048, all 34 observations, five terminal 20k clouds and an independent 100k holdout. Live noisy scoring decides the verdict; clean and EMA metrics are diagnostics.

Old CPU results remain separate evidence. Archived E22 GPU results may be cited as noncontemporary controls, with their source/task/options identities; no new E22 native jobs or seed sweeps are scheduled.

Screen and context elapsed times include evaluations and diagnostics. GPU memory comes from the guarded execution receipt, with fraction .2 on physical GPU0. These screen timings are not training-only throughput.

## Native metrics

| Task | Coverage | Accuracy | Terminal checks | Holdout precision | Center RMS/σ | Absolute trace bias | Radial KS |
|---|---|---|---|---:|---:|---:|---:|
| grid100 | FAIL | FAIL | [False, False, False, False, False] | 0.94284 | 0.20411962507030204 | 0.3630187048909559 | 0.1490040924868854 |
| rotated100 | FAIL | FAIL | [False, False, False, False, False] | 0.92094 | 0.1817155342385519 | 0.38251804070294515 | 0.15372136909425171 |
| staggered100 | FAIL | FAIL | [False, False, False, False, False] | 0.93624 | 0.24675855056616972 | 0.34347859179901397 | 0.14481215997115882 |

## Mechanism activity

| Task | Ordinary evaluations | Discoveries | Ordinary moves | Isolation evaluations | Isolation moves |
|---|---:|---:|---:|---:|---:|
| mode_hold | 1200 | 34 | 0 | 1200 | 0 |
| img_intensity2 | 600 | 155 | 3 | 600 | 0 |
| img_blobs4 | 600 | 127 | 0 | 600 | 0 |
| img_stripes2 | 600 | 119 | 0 | 600 | 0 |
| img_bars4 | 600 | 81 | 0 | 600 | 0 |
| vector_two_broad | 600 | 55 | 0 | 600 | 0 |
| vector_unequal_mass | 600 | 166 | 57 | 600 | 0 |
| vector_unequal_width | 600 | 162 | 90 | 600 | 0 |
| vector_anisotropic | 600 | 106 | 56 | 600 | 0 |
| vector_overlap | 600 | 32 | 0 | 600 | 0 |
| vector_spiral | 800 | 41 | 0 | 800 | 0 |
| ring_shift | 460 | 14812 | 277106 | 460 | 3388 |
| stationary | 750 | 7608 | 145347 | 750 | 624 |
| grid100 | 700 | 941 | 18033 | 700 | 806 |
| rotated100 | 700 | 928 | 24484 | 700 | 4983 |
| staggered100 | 700 | 1122 | 25825 | 700 | 2393 |

Ordinary move total: 491001. The ordinary feature-cell reaction path acts on the recorded frozen tasks.

Recommendation: retain the passing reference until the reported failures are addressed and independently retested; the candidate does not satisfy this CUDA screen suite.

# GPU leaderboard — cuda_fp32_v1

**Complete: 97 GPU toy runs and eight GPU convergence runs; all saved verdicts audited.**
The remaining 79 matrix cells are unsupported by the published adapters and receive no credit.
CPU scores are historical and do not enter this table. [Protocol and replay](README.md).

## Full 22-toy coverage

| Candidate | Toy PASS | Toy FAIL | Confirmed at | Good hold checks / 1200 | Hold result |
|---|---:|---:|---:|---:|---|
| shared_column_rms | 11/22 | 11 | 1400 | 131 | FAIL at 1532 |
| shared_rms | 11/22 | 11 | 4018 | 35 | FAIL at 4054 |
| H | 11/22 | 11 | — | 0 | Not confirmed by 6000 |
| eps_net_1m | 9/22 | 13 | 1764 | 19 | FAIL at 1784 |

Coverage rank is by toy pass count; equal counts are tied. Hold results are shown separately.

## Limited adapter coverage

| Candidate | Toy PASS / supported | Toy FAIL | Unsupported | Confirmed at | Good hold checks / 1200 | Hold result |
|---|---:|---:|---:|---:|---:|---|
| shared_rms_average2 | 3/3 | 0 | 19 | 1400 | 22 | FAIL at 1423 |
| PR143 | 0/2 | 2 | 20 | 1400 | 11 | FAIL at 1412 |
| PR107 | 0/2 | 2 | 20 | — | 0 | Not confirmed by 6000 |
| PR140 | 0/2 | 2 | 20 | — | 0 | Not confirmed by 6000 |

Limited rows cannot take an overall 22-toy slot. A supported-subset pass is not full qualification.

## All 22 toys

| Toy | eps_net_1m | H | shared_rms | shared_column_rms | shared_rms_average2 | PR107 | PR140 | PR143 |
|---|---|---|---|---|---|---|---|---|
| two_pole | FAIL | FAIL | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| mode_hold | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | FAIL |
| unipolar | FAIL | FAIL | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| mid_scale_identity | FAIL | FAIL | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| cover_leftover | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| trajectory | PASS | PASS | PASS | PASS | UNSUPPORTED | FAIL | FAIL | FAIL |
| residual_student | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| img_stripes2 | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| img_bars4 | PASS | PASS | PASS | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_overlap | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| img_blobs4 | FAIL | PASS | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| img_intensity2 | FAIL | PASS | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_unequal_mass | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_unequal_width | PASS | PASS | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_two_broad | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_anisotropic | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| vector_spiral | PASS | PASS | PASS | PASS | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| ae_gan_hold | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| unused_token_hold | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| grid100 | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| rotated100 | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| staggered100 | FAIL | FAIL | FAIL | FAIL | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

## Native 100-mode results

| Candidate | Toy | Final modes | Final HQ | Coverage | Accuracy |
|---|---|---:|---:|---|---|
| shared_column_rms | grid100 | 100/100 | 0.86570 | FAIL | FAIL |
| shared_column_rms | rotated100 | 100/100 | 0.92545 | FAIL | FAIL |
| shared_column_rms | staggered100 | 100/100 | 0.90655 | FAIL | FAIL |
| shared_rms | grid100 | 99/100 | 0.85805 | FAIL | FAIL |
| shared_rms | rotated100 | 100/100 | 0.91910 | FAIL | FAIL |
| shared_rms | staggered100 | 100/100 | 0.90695 | FAIL | FAIL |
| H | grid100 | 88/100 | 0.75190 | FAIL | FAIL |
| H | rotated100 | 100/100 | 0.90170 | FAIL | FAIL |
| H | staggered100 | 100/100 | 0.90065 | FAIL | FAIL |
| eps_net_1m | grid100 | 100/100 | 0.87345 | FAIL | FAIL |
| eps_net_1m | rotated100 | 100/100 | 0.92085 | FAIL | FAIL |
| eps_net_1m | staggered100 | 100/100 | 0.90385 | FAIL | FAIL |

**No release-qualified winner.**

All toy verdicts use their frozen sustained live-model criteria. The separate hold begins only after
200 consecutive qualifying checks; learning-time dips do not themselves fail that hold.
A hold failure stops the diagnostic at its first miss, so the count does not describe later recovery.

[Audit](audit.json) · [Raw ledger](ledger.jsonl) · [Declarations](candidates.json) · [Device repairs](repairs.json)

Completed 2026-09-24T21:39:17.513569+00:00

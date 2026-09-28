# Full 22-task matrix — prior EMA relaxation

Frozen outcomes on one package SHA-256 `64f82d9edba2a1422206b8474867cfdd35a793e42f24727c4e08fb79d0d532fc`: **7 PASS, 2 FAIL, 13 ERROR**.

| Task | Status | Passing observations | Reason or gate |
|---|---|---:|---|
| [grid100](evidence/grid100/result.json) | PASS | 21 | native terminal and holdout PASS |
| [rotated100](evidence/rotated100/result.json) | PASS | 8 | native terminal and holdout PASS |
| [staggered100](evidence/staggered100/result.json) | PASS | 19 | native terminal and holdout PASS |
| [mode_hold](all22/mode_hold/result.json) | ERROR | — | fresh real-batch callback missing |
| [img_intensity2](all22/img_intensity2/result.json) | ERROR | — | fresh real-batch callback missing |
| [img_blobs4](all22/img_blobs4/result.json) | ERROR | — | fresh real-batch callback missing |
| [img_stripes2](all22/img_stripes2/result.json) | ERROR | — | fresh real-batch callback missing |
| [img_bars4](all22/img_bars4/result.json) | ERROR | — | fresh real-batch callback missing |
| [vector_two_broad](all22/vector_two_broad/result.json) | PASS | 23 | see result JSON |
| [vector_unequal_mass](all22/vector_unequal_mass/result.json) | FAIL | 0 | see result JSON |
| [vector_unequal_width](all22/vector_unequal_width/result.json) | PASS | 20 | see result JSON |
| [vector_anisotropic](all22/vector_anisotropic/result.json) | PASS | 22 | see result JSON |
| [vector_overlap](all22/vector_overlap/result.json) | FAIL | 21 | see result JSON |
| [vector_spiral](all22/vector_spiral/result.json) | PASS | 24 | see result JSON |
| [two_pole](all22/two_pole/result.json) | ERROR | — | custom host BD parity refused |
| [trajectory](all22/trajectory/result.json) | ERROR | — | custom host BD parity refused |
| [residual_student](all22/residual_student/result.json) | ERROR | — | custom host BD parity refused |
| [unipolar](all22/unipolar/result.json) | ERROR | — | custom host BD parity refused |
| [ae_gan_hold](all22/ae_gan_hold/result.json) | ERROR | — | custom host BD parity refused |
| [cover_leftover](all22/cover_leftover/result.json) | ERROR | — | custom host BD parity refused |
| [unused_token_hold](all22/unused_token_hold/result.json) | ERROR | — | custom host BD parity refused |
| [mid_scale_identity](all22/mid_scale_identity/result.json) | ERROR | — | custom host BD parity refused |

ERROR rows stopped before scoring. The two FAIL rows are measured vector failures. See [README.md](README.md) for the native result and limitations.

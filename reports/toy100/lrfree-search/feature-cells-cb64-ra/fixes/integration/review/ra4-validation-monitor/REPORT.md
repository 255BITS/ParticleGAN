# Corrected CUDA validation artifact review

State: PENDING; 5/16 screens collected.

Primary verdicts and mandatory validity come from the frozen original collector. Active tasks remain pending.

| Task | Primary | Canonical validity | Accepted | Steps | Peak reserved MiB |
|---|---|---|---|---:|---:|
| mode_hold | PASS | INVALID | ERROR | 1200 | 88.0 |
| img_intensity2 | PENDING | UNVERIFIED | PENDING | — | — |
| img_blobs4 | PASS | INVALID | ERROR | 600 | 66.0 |
| img_stripes2 | PENDING | UNVERIFIED | PENDING | — | — |
| img_bars4 | PENDING | UNVERIFIED | PENDING | — | — |
| vector_two_broad | PENDING | UNVERIFIED | PENDING | — | — |
| vector_unequal_mass | PASS | INVALID | ERROR | 1200 | 92.0 |
| vector_unequal_width | PENDING | UNVERIFIED | PENDING | — | — |
| vector_anisotropic | PENDING | UNVERIFIED | PENDING | — | — |
| vector_overlap | PENDING | UNVERIFIED | PENDING | — | — |
| vector_spiral | PENDING | UNVERIFIED | PENDING | — | — |
| ring_shift | PASS | INVALID | ERROR | 4600 | 188.0 |
| stationary | PENDING | UNVERIFIED | PENDING | — | — |
| grid100 | FAIL | INVALID | ERROR | 7000 | 264.0 |
| rotated100 | PENDING | UNVERIFIED | PENDING | — | — |
| staggered100 | PENDING | UNVERIFIED | PENDING | — | — |

## Runtime or fixture errors

- mode_hold: ["original resolved options differ: {'evaluation_generate': 'indexed', 'serial_backward_argument': True, 'strict_streams': True, 'initialization': 'batch_feature_zero', 'diagnostics': True, 'save_final_state': True, 'ring_frozen_control': False, 'eval_output_noise': True, 'image_prior_perturb': False}"]; None
- img_blobs4: ["original resolved options differ: {'evaluation_generate': 'indexed', 'serial_backward_argument': True, 'strict_streams': True, 'initialization': 'batch_feature_zero', 'diagnostics': True, 'save_final_state': True, 'ring_frozen_control': False, 'eval_output_noise': True, 'image_prior_perturb': False}"]; None
- vector_unequal_mass: ["original resolved options differ: {'evaluation_generate': 'indexed', 'serial_backward_argument': True, 'strict_streams': True, 'initialization': 'batch_feature_zero', 'diagnostics': True, 'save_final_state': True, 'ring_frozen_control': False, 'eval_output_noise': True, 'image_prior_perturb': False}"]; None
- ring_shift: ["original resolved options differ: {'evaluation_generate': 'indexed', 'serial_backward_argument': True, 'strict_streams': True, 'initialization': 'batch_feature_zero', 'diagnostics': True, 'save_final_state': True, 'ring_frozen_control': False, 'eval_output_noise': True, 'image_prior_perturb': False}"]; None
- grid100: ["original resolved options differ: {'evaluation_generate': 'indexed', 'serial_backward_argument': True, 'strict_streams': True, 'initialization': 'batch_feature_zero', 'diagnostics': True, 'save_final_state': True, 'ring_frozen_control': False, 'eval_output_noise': True, 'image_prior_perturb': False}"]; None

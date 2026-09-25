# Shared cap6 discriminator normalization structure research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass |
| --- | ---: | --- |
| normstruct_center_pre_affine | 19585 | FAIL (0/24) |
| normstruct_center_pre_fixed | 19009 | FAIL (0/24) |
| normstruct_ln_first_last | 19393 | FAIL (0/24) |
| normstruct_ln_input_injection025 | 19969 | FAIL (0/24) |
| normstruct_ln_last | 19201 | FAIL (0/24) |
| normstruct_ln_post_affine | 19585 | FAIL (0/24) |
| normstruct_ln_post_fixed | 19009 | FAIL (0/24) |
| normstruct_ln_post_raw_blend025 | 19585 | FAIL (0/24) |
| normstruct_ln_pre_fixed | 19009 | FAIL (0/24) |
| normstruct_ln_raw_blend025 | 19585 | FAIL (0/24) |
| normstruct_ln_raw_blend050 | 19585 | FAIL (1/24) |
| normstruct_ln_residual025 | 19585 | FAIL (0/24) |
| normstruct_ln_rms_mix050 | 19873 | FAIL (0/24) |
| normstruct_rms_post_affine | 19297 | FAIL (0/24) |
| normstruct_rms_pre_affine | 19297 | FAIL (0/24) |
| normstruct_rms_pre_fixed | 19009 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| normstruct_ln_pre_fixed | vector_unequal_mass | FAIL | FAIL | 10.49 | [JSON](episodes/shared_c6__normstruct_ln_pre_fixed__vector_unequal_mass.json.gz) |
| normstruct_ln_post_affine | vector_unequal_mass | FAIL | FAIL | 10.14 | [JSON](episodes/shared_c6__normstruct_ln_post_affine__vector_unequal_mass.json.gz) |
| normstruct_ln_post_fixed | vector_unequal_mass | FAIL | FAIL | 9.48 | [JSON](episodes/shared_c6__normstruct_ln_post_fixed__vector_unequal_mass.json.gz) |
| normstruct_ln_first_last | vector_unequal_mass | FAIL | FAIL | 8.80 | [JSON](episodes/shared_c6__normstruct_ln_first_last__vector_unequal_mass.json.gz) |
| normstruct_ln_last | vector_unequal_mass | FAIL | FAIL | 8.28 | [JSON](episodes/shared_c6__normstruct_ln_last__vector_unequal_mass.json.gz) |
| normstruct_rms_pre_affine | vector_unequal_mass | FAIL | FAIL | 11.38 | [JSON](episodes/shared_c6__normstruct_rms_pre_affine__vector_unequal_mass.json.gz) |
| normstruct_rms_pre_fixed | vector_unequal_mass | FAIL | FAIL | 10.63 | [JSON](episodes/shared_c6__normstruct_rms_pre_fixed__vector_unequal_mass.json.gz) |
| normstruct_rms_post_affine | vector_unequal_mass | FAIL | FAIL | 11.26 | [JSON](episodes/shared_c6__normstruct_rms_post_affine__vector_unequal_mass.json.gz) |
| normstruct_center_pre_affine | vector_unequal_mass | FAIL | FAIL | 10.33 | [JSON](episodes/shared_c6__normstruct_center_pre_affine__vector_unequal_mass.json.gz) |
| normstruct_center_pre_fixed | vector_unequal_mass | FAIL | FAIL | 9.35 | [JSON](episodes/shared_c6__normstruct_center_pre_fixed__vector_unequal_mass.json.gz) |
| normstruct_ln_raw_blend025 | vector_unequal_mass | FAIL | FAIL | 12.06 | [JSON](episodes/shared_c6__normstruct_ln_raw_blend025__vector_unequal_mass.json.gz) |
| normstruct_ln_raw_blend050 | vector_unequal_mass | FAIL | FAIL | 12.05 | [JSON](episodes/shared_c6__normstruct_ln_raw_blend050__vector_unequal_mass.json.gz) |
| normstruct_ln_post_raw_blend025 | vector_unequal_mass | FAIL | FAIL | 12.81 | [JSON](episodes/shared_c6__normstruct_ln_post_raw_blend025__vector_unequal_mass.json.gz) |
| normstruct_ln_residual025 | vector_unequal_mass | FAIL | FAIL | 10.39 | [JSON](episodes/shared_c6__normstruct_ln_residual025__vector_unequal_mass.json.gz) |
| normstruct_ln_input_injection025 | vector_unequal_mass | FAIL | FAIL | 11.25 | [JSON](episodes/shared_c6__normstruct_ln_input_injection025__vector_unequal_mass.json.gz) |
| normstruct_ln_rms_mix050 | vector_unequal_mass | FAIL | FAIL | 15.11 | [JSON](episodes/shared_c6__normstruct_ln_rms_mix050__vector_unequal_mass.json.gz) |

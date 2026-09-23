# Shared cap6 discriminator normalization structure research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass |
| --- | ---: | --- |
| normstruct_center_fixed_blend025 | 19009 | FAIL (0/24) |
| normstruct_group2_affine | 19585 | FAIL (0/24) |
| normstruct_group2_fixed | 19009 | FAIL (0/24) |
| normstruct_group4_affine | 19585 | FAIL (0/24) |
| normstruct_ln_blend050_first | 19585 | FAIL (0/24) |
| normstruct_ln_blend050_first_last | 19585 | FAIL (0/24) |
| normstruct_ln_blend050_last | 19585 | FAIL (0/24) |
| normstruct_power025_affine | 19585 | FAIL (0/24) |
| normstruct_power025_fixed | 19009 | FAIL (0/24) |
| normstruct_power050_affine | 19585 | FAIL (0/24) |
| normstruct_power050_fixed | 19009 | FAIL (0/24) |
| normstruct_power075_fixed | 19009 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| normstruct_power025_fixed | vector_unequal_mass | FAIL | FAIL | 13.25 | [JSON](episodes/shared_c6__normstruct_power025_fixed__vector_unequal_mass.json.gz) |
| normstruct_power050_fixed | vector_unequal_mass | FAIL | FAIL | 12.27 | [JSON](episodes/shared_c6__normstruct_power050_fixed__vector_unequal_mass.json.gz) |
| normstruct_power075_fixed | vector_unequal_mass | FAIL | FAIL | 12.30 | [JSON](episodes/shared_c6__normstruct_power075_fixed__vector_unequal_mass.json.gz) |
| normstruct_power025_affine | vector_unequal_mass | FAIL | FAIL | 13.80 | [JSON](episodes/shared_c6__normstruct_power025_affine__vector_unequal_mass.json.gz) |
| normstruct_power050_affine | vector_unequal_mass | FAIL | FAIL | 12.63 | [JSON](episodes/shared_c6__normstruct_power050_affine__vector_unequal_mass.json.gz) |
| normstruct_group2_affine | vector_unequal_mass | FAIL | FAIL | 13.48 | [JSON](episodes/shared_c6__normstruct_group2_affine__vector_unequal_mass.json.gz) |
| normstruct_group4_affine | vector_unequal_mass | FAIL | FAIL | 13.86 | [JSON](episodes/shared_c6__normstruct_group4_affine__vector_unequal_mass.json.gz) |
| normstruct_group2_fixed | vector_unequal_mass | FAIL | FAIL | 11.59 | [JSON](episodes/shared_c6__normstruct_group2_fixed__vector_unequal_mass.json.gz) |
| normstruct_ln_blend050_first | vector_unequal_mass | FAIL | FAIL | 10.46 | [JSON](episodes/shared_c6__normstruct_ln_blend050_first__vector_unequal_mass.json.gz) |
| normstruct_ln_blend050_last | vector_unequal_mass | FAIL | FAIL | 10.39 | [JSON](episodes/shared_c6__normstruct_ln_blend050_last__vector_unequal_mass.json.gz) |
| normstruct_ln_blend050_first_last | vector_unequal_mass | FAIL | FAIL | 11.72 | [JSON](episodes/shared_c6__normstruct_ln_blend050_first_last__vector_unequal_mass.json.gz) |
| normstruct_center_fixed_blend025 | vector_unequal_mass | FAIL | FAIL | 11.42 | [JSON](episodes/shared_c6__normstruct_center_fixed_blend025__vector_unequal_mass.json.gz) |

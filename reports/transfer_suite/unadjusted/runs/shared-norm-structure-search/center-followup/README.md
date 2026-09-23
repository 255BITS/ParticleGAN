# Shared cap6 discriminator normalization structure research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass |
| --- | ---: | --- |
| normstruct_center_fixed128_beta4 | 33537 | FAIL (0/24) |
| normstruct_center_fixed128_beta8 | 33537 | FAIL (0/24) |
| normstruct_center_fixed96_beta3 | 19009 | FAIL (0/24) |
| normstruct_center_fixed96_beta5 | 19009 | FAIL (0/24) |
| normstruct_center_fixed96_beta6 | 19009 | FAIL (1/24) |
| normstruct_center_fixed96_beta8 | 19009 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| normstruct_center_fixed96_beta3 | vector_unequal_mass | FAIL | FAIL | 17.30 | [JSON](episodes/shared_c6__normstruct_center_fixed96_beta3__vector_unequal_mass.json.gz) |
| normstruct_center_fixed96_beta5 | vector_unequal_mass | FAIL | FAIL | 8.77 | [JSON](episodes/shared_c6__normstruct_center_fixed96_beta5__vector_unequal_mass.json.gz) |
| normstruct_center_fixed96_beta6 | vector_unequal_mass | FAIL | FAIL | 9.73 | [JSON](episodes/shared_c6__normstruct_center_fixed96_beta6__vector_unequal_mass.json.gz) |
| normstruct_center_fixed96_beta8 | vector_unequal_mass | FAIL | FAIL | 9.73 | [JSON](episodes/shared_c6__normstruct_center_fixed96_beta8__vector_unequal_mass.json.gz) |
| normstruct_center_fixed128_beta4 | vector_unequal_mass | FAIL | PASS | 10.63 | [JSON](episodes/shared_c6__normstruct_center_fixed128_beta4__vector_unequal_mass.json.gz) |
| normstruct_center_fixed128_beta8 | vector_unequal_mass | FAIL | FAIL | 10.65 | [JSON](episodes/shared_c6__normstruct_center_fixed128_beta8__vector_unequal_mass.json.gz) |

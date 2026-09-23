# Shared cap6 pointwise normalized discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass |
| --- | ---: | --- |
| pointnorm_layer_all_softplus128_l3 | 34305 | FAIL (0/24) |
| pointnorm_layer_all_softplus160_l3 | 53121 | FAIL (0/24) |
| pointnorm_layer_all_softplus64_l3 | 8961 | FAIL (0/24) |
| pointnorm_layer_all_softplus96_beta10_l3 | 19585 | FAIL (0/24) |
| pointnorm_layer_all_softplus96_beta2_l3 | 19585 | FAIL (0/24) |
| pointnorm_layer_all_softplus96_skip_l3 | 19587 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| pointnorm_layer_all_softplus64_l3 | vector_unequal_mass | FAIL | FAIL | 10.08 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus64_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus128_l3 | vector_unequal_mass | FAIL | FAIL | 13.48 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus128_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus160_l3 | vector_unequal_mass | FAIL | FAIL | 13.21 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus160_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus96_beta2_l3 | vector_unequal_mass | FAIL | FAIL | 9.64 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_beta2_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus96_beta10_l3 | vector_unequal_mass | FAIL | FAIL | 10.42 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_beta10_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus96_skip_l3 | vector_unequal_mass | FAIL | FAIL | 10.63 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_skip_l3__vector_unequal_mass.json.gz) |

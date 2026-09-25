# Shared cap6 pointwise normalized discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass | vector_unequal_width |
| --- | ---: | --- | --- |
| pointnorm_layer_all_silu128_l3 | 34305 | FAIL (0/24) | FAIL (0/24) |
| pointnorm_layer_all_softplus96_l3 | 19585 | FAIL (0/24) | FAIL (0/24) |
| pointnorm_layer_first_silu128_l3 | 33793 | FAIL (0/24) | FAIL (0/24) |
| pointnorm_layer_first_softplus96_l3 | 19201 | FAIL (0/24) | FAIL (0/24) |
| pointnorm_rms_all_silu128_l3 | 33921 | FAIL (0/24) | FAIL (0/24) |
| pointnorm_weight_all_silu128_l3 | 33922 | FAIL (0/24) | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| pointnorm_layer_first_softplus96_l3 | vector_unequal_mass | FAIL | FAIL | 9.90 | [JSON](episodes/shared_c6__pointnorm_layer_first_softplus96_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_first_softplus96_l3 | vector_unequal_width | FAIL | FAIL | 8.90 | [JSON](episodes/shared_c6__pointnorm_layer_first_softplus96_l3__vector_unequal_width.json.gz) |
| pointnorm_layer_all_softplus96_l3 | vector_unequal_mass | FAIL | FAIL | 9.95 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_softplus96_l3 | vector_unequal_width | FAIL | FAIL | 9.74 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_l3__vector_unequal_width.json.gz) |
| pointnorm_layer_first_silu128_l3 | vector_unequal_mass | FAIL | FAIL | 9.28 | [JSON](episodes/shared_c6__pointnorm_layer_first_silu128_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_first_silu128_l3 | vector_unequal_width | FAIL | FAIL | 9.25 | [JSON](episodes/shared_c6__pointnorm_layer_first_silu128_l3__vector_unequal_width.json.gz) |
| pointnorm_layer_all_silu128_l3 | vector_unequal_mass | FAIL | FAIL | 12.63 | [JSON](episodes/shared_c6__pointnorm_layer_all_silu128_l3__vector_unequal_mass.json.gz) |
| pointnorm_layer_all_silu128_l3 | vector_unequal_width | FAIL | FAIL | 11.20 | [JSON](episodes/shared_c6__pointnorm_layer_all_silu128_l3__vector_unequal_width.json.gz) |
| pointnorm_rms_all_silu128_l3 | vector_unequal_mass | FAIL | FAIL | 13.76 | [JSON](episodes/shared_c6__pointnorm_rms_all_silu128_l3__vector_unequal_mass.json.gz) |
| pointnorm_rms_all_silu128_l3 | vector_unequal_width | FAIL | FAIL | 13.84 | [JSON](episodes/shared_c6__pointnorm_rms_all_silu128_l3__vector_unequal_width.json.gz) |
| pointnorm_weight_all_silu128_l3 | vector_unequal_mass | FAIL | FAIL | 9.86 | [JSON](episodes/shared_c6__pointnorm_weight_all_silu128_l3__vector_unequal_mass.json.gz) |
| pointnorm_weight_all_silu128_l3 | vector_unequal_width | FAIL | FAIL | 10.27 | [JSON](episodes/shared_c6__pointnorm_weight_all_silu128_l3__vector_unequal_width.json.gz) |

# Shared cap6 pointwise normalized discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_width |
| --- | ---: | --- |
| pointnorm_layer_all_softplus96_beta2_l3 | 19585 | FAIL (0/24) |
| pointnorm_layer_all_softplus96_skip_l3 | 19587 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| pointnorm_layer_all_softplus96_beta2_l3 | vector_unequal_width | FAIL | FAIL | 10.60 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_beta2_l3__vector_unequal_width.json.gz) |
| pointnorm_layer_all_softplus96_skip_l3 | vector_unequal_width | FAIL | FAIL | 10.43 | [JSON](episodes/shared_c6__pointnorm_layer_all_softplus96_skip_l3__vector_unequal_width.json.gz) |

# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_width |
| --- | ---: | --- |
| width_last_silu160_l2_linear_skip_head05 | 26403 | FAIL (0/24) |
| width_last_silu160_l2_linear_skip_head2 | 26403 | FAIL (0/24) |
| width_last_softplus3_128_l3 | 33537 | FAIL (0/24) |
| width_last_softplus4_128_l3 | 33537 | FAIL (0/24) |
| width_last_softplus6_128_l3 | 33537 | FAIL (0/24) |
| width_last_softplus8_128_l3 | 33537 | PASS (5/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| width_last_softplus3_128_l3 | vector_unequal_width | FAIL | FAIL | 10.75 | [JSON](episodes/shared_c6__width_last_softplus3_128_l3__vector_unequal_width.json.gz) |
| width_last_softplus4_128_l3 | vector_unequal_width | FAIL | FAIL | 9.34 | [JSON](episodes/shared_c6__width_last_softplus4_128_l3__vector_unequal_width.json.gz) |
| width_last_softplus6_128_l3 | vector_unequal_width | FAIL | FAIL | 9.36 | [JSON](episodes/shared_c6__width_last_softplus6_128_l3__vector_unequal_width.json.gz) |
| width_last_softplus8_128_l3 | vector_unequal_width | PASS | PASS | 9.86 | [JSON](episodes/shared_c6__width_last_softplus8_128_l3__vector_unequal_width.json.gz) |
| width_last_silu160_l2_linear_skip_head05 | vector_unequal_width | FAIL | FAIL | 8.47 | [JSON](episodes/shared_c6__width_last_silu160_l2_linear_skip_head05__vector_unequal_width.json.gz) |
| width_last_silu160_l2_linear_skip_head2 | vector_unequal_width | FAIL | FAIL | 9.95 | [JSON](episodes/shared_c6__width_last_silu160_l2_linear_skip_head2__vector_unequal_width.json.gz) |

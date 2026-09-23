# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_width |
| --- | ---: | --- |
| width_ref_silu160_l2_head025 | 26401 | FAIL (0/24) |
| width_ref_silu160_l2_head05 | 26401 | FAIL (0/24) |
| width_ref_silu160_l2_head2 | 26401 | FAIL (0/24) |
| width_ref_silu160_l2_linear_skip | 26403 | FAIL (0/24) |
| width_ref_silu160_l2_residual | 26401 | FAIL (0/24) |
| width_ref_softplus10_128_l3 | 33537 | FAIL (0/24) |
| width_ref_softplus128_l3_head025 | 33537 | FAIL (0/24) |
| width_ref_softplus128_l3_head05 | 33537 | FAIL (0/24) |
| width_ref_softplus128_l3_head2 | 33537 | FAIL (0/24) |
| width_ref_softplus128_l3_linear_skip | 33539 | FAIL (0/24) |
| width_ref_softplus128_l3_residual | 33537 | FAIL (0/24) |
| width_ref_softplus1_128_l3 | 33537 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| width_ref_softplus128_l3_head025 | vector_unequal_width | FAIL | FAIL | 10.76 | [JSON](episodes/shared_c6__width_ref_softplus128_l3_head025__vector_unequal_width.json.gz) |
| width_ref_softplus128_l3_head05 | vector_unequal_width | FAIL | FAIL | 9.50 | [JSON](episodes/shared_c6__width_ref_softplus128_l3_head05__vector_unequal_width.json.gz) |
| width_ref_softplus128_l3_head2 | vector_unequal_width | FAIL | FAIL | 9.14 | [JSON](episodes/shared_c6__width_ref_softplus128_l3_head2__vector_unequal_width.json.gz) |
| width_ref_softplus1_128_l3 | vector_unequal_width | FAIL | FAIL | 9.07 | [JSON](episodes/shared_c6__width_ref_softplus1_128_l3__vector_unequal_width.json.gz) |
| width_ref_softplus10_128_l3 | vector_unequal_width | FAIL | PASS | 9.02 | [JSON](episodes/shared_c6__width_ref_softplus10_128_l3__vector_unequal_width.json.gz) |
| width_ref_softplus128_l3_linear_skip | vector_unequal_width | FAIL | FAIL | 9.47 | [JSON](episodes/shared_c6__width_ref_softplus128_l3_linear_skip__vector_unequal_width.json.gz) |
| width_ref_softplus128_l3_residual | vector_unequal_width | FAIL | FAIL | 10.00 | [JSON](episodes/shared_c6__width_ref_softplus128_l3_residual__vector_unequal_width.json.gz) |
| width_ref_silu160_l2_head025 | vector_unequal_width | FAIL | FAIL | 7.34 | [JSON](episodes/shared_c6__width_ref_silu160_l2_head025__vector_unequal_width.json.gz) |
| width_ref_silu160_l2_head05 | vector_unequal_width | FAIL | FAIL | 7.36 | [JSON](episodes/shared_c6__width_ref_silu160_l2_head05__vector_unequal_width.json.gz) |
| width_ref_silu160_l2_head2 | vector_unequal_width | FAIL | FAIL | 7.41 | [JSON](episodes/shared_c6__width_ref_silu160_l2_head2__vector_unequal_width.json.gz) |
| width_ref_silu160_l2_linear_skip | vector_unequal_width | FAIL | FAIL | 8.58 | [JSON](episodes/shared_c6__width_ref_silu160_l2_linear_skip__vector_unequal_width.json.gz) |
| width_ref_silu160_l2_residual | vector_unequal_width | FAIL | FAIL | 8.22 | [JSON](episodes/shared_c6__width_ref_silu160_l2_residual__vector_unequal_width.json.gz) |

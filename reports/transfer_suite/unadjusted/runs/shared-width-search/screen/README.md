# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_width |
| --- | ---: | --- |
| width_raw_silu128_l2 | 17025 | FAIL (0/24) |
| width_raw_silu128_l3_linear_skip | 33539 | FAIL (0/24) |
| width_raw_silu128_l4 | 50049 | FAIL (0/24) |
| width_raw_silu160_l2 | 26401 | FAIL (0/24) |
| width_raw_silu160_l3 | 52161 | FAIL (0/24) |
| width_raw_silu160_l4 | 77921 | FAIL (0/24) |
| width_raw_silu192_l2 | 37825 | FAIL (0/24) |
| width_raw_silu192_l3 | 74881 | FAIL (0/24) |
| width_raw_silu64_l3 | 8577 | FAIL (0/24) |
| width_raw_silu96_l2 | 9697 | FAIL (0/24) |
| width_raw_silu96_l3 | 19009 | FAIL (0/24) |
| width_raw_silu96_l4 | 28321 | FAIL (0/24) |
| width_raw_softplus128_l3 | 33537 | FAIL (1/24) |
| width_raw_softplus128_l4 | 50049 | FAIL (0/24) |
| width_raw_softplus2_128_l3 | 33537 | FAIL (0/24) |
| width_residual_raw_silu128_l3 | 33537 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| width_raw_silu64_l3 | vector_unequal_width | FAIL | FAIL | 34.19 | [JSON](episodes/shared_c6__width_raw_silu64_l3__vector_unequal_width.json.gz) |
| width_raw_silu96_l3 | vector_unequal_width | FAIL | FAIL | 12.31 | [JSON](episodes/shared_c6__width_raw_silu96_l3__vector_unequal_width.json.gz) |
| width_raw_silu160_l3 | vector_unequal_width | FAIL | FAIL | 12.46 | [JSON](episodes/shared_c6__width_raw_silu160_l3__vector_unequal_width.json.gz) |
| width_raw_silu192_l3 | vector_unequal_width | FAIL | FAIL | 21.08 | [JSON](episodes/shared_c6__width_raw_silu192_l3__vector_unequal_width.json.gz) |
| width_raw_silu128_l2 | vector_unequal_width | FAIL | FAIL | 11.04 | [JSON](episodes/shared_c6__width_raw_silu128_l2__vector_unequal_width.json.gz) |
| width_raw_silu128_l4 | vector_unequal_width | FAIL | FAIL | 19.53 | [JSON](episodes/shared_c6__width_raw_silu128_l4__vector_unequal_width.json.gz) |
| width_raw_silu96_l2 | vector_unequal_width | FAIL | FAIL | 6.65 | [JSON](episodes/shared_c6__width_raw_silu96_l2__vector_unequal_width.json.gz) |
| width_raw_silu96_l4 | vector_unequal_width | FAIL | FAIL | 9.16 | [JSON](episodes/shared_c6__width_raw_silu96_l4__vector_unequal_width.json.gz) |
| width_raw_silu160_l2 | vector_unequal_width | FAIL | FAIL | 9.12 | [JSON](episodes/shared_c6__width_raw_silu160_l2__vector_unequal_width.json.gz) |
| width_raw_silu160_l4 | vector_unequal_width | FAIL | FAIL | 15.59 | [JSON](episodes/shared_c6__width_raw_silu160_l4__vector_unequal_width.json.gz) |
| width_raw_silu192_l2 | vector_unequal_width | FAIL | FAIL | 9.92 | [JSON](episodes/shared_c6__width_raw_silu192_l2__vector_unequal_width.json.gz) |
| width_raw_silu128_l3_linear_skip | vector_unequal_width | FAIL | FAIL | 10.41 | [JSON](episodes/shared_c6__width_raw_silu128_l3_linear_skip__vector_unequal_width.json.gz) |
| width_residual_raw_silu128_l3 | vector_unequal_width | FAIL | FAIL | 13.49 | [JSON](episodes/shared_c6__width_residual_raw_silu128_l3__vector_unequal_width.json.gz) |
| width_raw_softplus128_l3 | vector_unequal_width | FAIL | FAIL | 9.58 | [JSON](episodes/shared_c6__width_raw_softplus128_l3__vector_unequal_width.json.gz) |
| width_raw_softplus2_128_l3 | vector_unequal_width | FAIL | FAIL | 9.51 | [JSON](episodes/shared_c6__width_raw_softplus2_128_l3__vector_unequal_width.json.gz) |
| width_raw_softplus128_l4 | vector_unequal_width | FAIL | FAIL | 11.76 | [JSON](episodes/shared_c6__width_raw_softplus128_l4__vector_unequal_width.json.gz) |

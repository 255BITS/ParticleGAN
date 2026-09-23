# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass | vector_unequal_width |
| --- | ---: | --- | --- |
| local_cauchy128_direct | 131 | FAIL (0/24) | FAIL (0/24) |
| local_product_silu128_l2 | 33921 | FAIL (0/24) | FAIL (0/24) |
| local_product_silu64_l2 | 8769 | FAIL (0/24) | FAIL (0/24) |
| local_product_silu96_l2 | 19297 | FAIL (0/24) | FAIL (0/24) |
| local_product_softplus96_l2 | 19297 | FAIL (0/24) | FAIL (0/24) |
| local_quad16_width1 | 99 | FAIL (0/24) | FAIL (0/24) |
| local_quad32_adaptive | 291 | FAIL (0/24) | FAIL (0/24) |
| local_quad32_width05 | 195 | FAIL (0/24) | FAIL (0/24) |
| local_quad32_width1 | 195 | FAIL (0/24) | FAIL (0/24) |
| local_rbf128_adaptive | 515 | FAIL (0/24) | FAIL (0/24) |
| local_rbf128_direct | 131 | FAIL (0/24) | FAIL (0/24) |
| local_rbf128_softplus64 | 12609 | FAIL (0/24) | FAIL (0/24) |
| local_rbf256_direct | 259 | FAIL (0/24) | FAIL (0/24) |
| local_rbf64_direct | 67 | FAIL (0/24) | FAIL (0/24) |
| local_squared_silu64_l3 | 8577 | FAIL (0/24) | FAIL (0/24) |
| local_squared_silu96_l3 | 19009 | FAIL (0/24) | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| local_rbf64_direct | vector_unequal_mass | FAIL | FAIL | 17.33 | [JSON](episodes/shared_c6__local_rbf64_direct__vector_unequal_mass.json.gz) |
| local_rbf64_direct | vector_unequal_width | FAIL | FAIL | 5.50 | [JSON](episodes/shared_c6__local_rbf64_direct__vector_unequal_width.json.gz) |
| local_rbf128_direct | vector_unequal_mass | FAIL | FAIL | 14.93 | [JSON](episodes/shared_c6__local_rbf128_direct__vector_unequal_mass.json.gz) |
| local_rbf128_direct | vector_unequal_width | FAIL | FAIL | 17.09 | [JSON](episodes/shared_c6__local_rbf128_direct__vector_unequal_width.json.gz) |
| local_rbf256_direct | vector_unequal_mass | FAIL | FAIL | 12.65 | [JSON](episodes/shared_c6__local_rbf256_direct__vector_unequal_mass.json.gz) |
| local_rbf256_direct | vector_unequal_width | FAIL | FAIL | 11.33 | [JSON](episodes/shared_c6__local_rbf256_direct__vector_unequal_width.json.gz) |
| local_rbf128_adaptive | vector_unequal_mass | FAIL | FAIL | 10.93 | [JSON](episodes/shared_c6__local_rbf128_adaptive__vector_unequal_mass.json.gz) |
| local_rbf128_adaptive | vector_unequal_width | FAIL | FAIL | 10.53 | [JSON](episodes/shared_c6__local_rbf128_adaptive__vector_unequal_width.json.gz) |
| local_cauchy128_direct | vector_unequal_mass | FAIL | FAIL | 8.91 | [JSON](episodes/shared_c6__local_cauchy128_direct__vector_unequal_mass.json.gz) |
| local_cauchy128_direct | vector_unequal_width | FAIL | FAIL | 7.78 | [JSON](episodes/shared_c6__local_cauchy128_direct__vector_unequal_width.json.gz) |
| local_rbf128_softplus64 | vector_unequal_mass | FAIL | FAIL | 10.69 | [JSON](episodes/shared_c6__local_rbf128_softplus64__vector_unequal_mass.json.gz) |
| local_rbf128_softplus64 | vector_unequal_width | FAIL | FAIL | 14.50 | [JSON](episodes/shared_c6__local_rbf128_softplus64__vector_unequal_width.json.gz) |
| local_quad16_width1 | vector_unequal_mass | FAIL | FAIL | 6.77 | [JSON](episodes/shared_c6__local_quad16_width1__vector_unequal_mass.json.gz) |
| local_quad16_width1 | vector_unequal_width | FAIL | FAIL | 6.85 | [JSON](episodes/shared_c6__local_quad16_width1__vector_unequal_width.json.gz) |
| local_quad32_width1 | vector_unequal_mass | FAIL | FAIL | 7.37 | [JSON](episodes/shared_c6__local_quad32_width1__vector_unequal_mass.json.gz) |
| local_quad32_width1 | vector_unequal_width | FAIL | FAIL | 7.34 | [JSON](episodes/shared_c6__local_quad32_width1__vector_unequal_width.json.gz) |
| local_quad32_width05 | vector_unequal_mass | FAIL | FAIL | 9.95 | [JSON](episodes/shared_c6__local_quad32_width05__vector_unequal_mass.json.gz) |
| local_quad32_width05 | vector_unequal_width | FAIL | FAIL | 7.28 | [JSON](episodes/shared_c6__local_quad32_width05__vector_unequal_width.json.gz) |
| local_quad32_adaptive | vector_unequal_mass | FAIL | FAIL | 8.12 | [JSON](episodes/shared_c6__local_quad32_adaptive__vector_unequal_mass.json.gz) |
| local_quad32_adaptive | vector_unequal_width | FAIL | FAIL | 8.30 | [JSON](episodes/shared_c6__local_quad32_adaptive__vector_unequal_width.json.gz) |
| local_product_silu64_l2 | vector_unequal_mass | FAIL | FAIL | 7.24 | [JSON](episodes/shared_c6__local_product_silu64_l2__vector_unequal_mass.json.gz) |
| local_product_silu64_l2 | vector_unequal_width | FAIL | FAIL | 8.23 | [JSON](episodes/shared_c6__local_product_silu64_l2__vector_unequal_width.json.gz) |
| local_product_silu96_l2 | vector_unequal_mass | FAIL | FAIL | 8.17 | [JSON](episodes/shared_c6__local_product_silu96_l2__vector_unequal_mass.json.gz) |
| local_product_silu96_l2 | vector_unequal_width | FAIL | FAIL | 8.72 | [JSON](episodes/shared_c6__local_product_silu96_l2__vector_unequal_width.json.gz) |
| local_product_silu128_l2 | vector_unequal_mass | FAIL | FAIL | 9.97 | [JSON](episodes/shared_c6__local_product_silu128_l2__vector_unequal_mass.json.gz) |
| local_product_silu128_l2 | vector_unequal_width | FAIL | FAIL | 10.82 | [JSON](episodes/shared_c6__local_product_silu128_l2__vector_unequal_width.json.gz) |
| local_squared_silu64_l3 | vector_unequal_mass | FAIL | FAIL | 7.82 | [JSON](episodes/shared_c6__local_squared_silu64_l3__vector_unequal_mass.json.gz) |
| local_squared_silu64_l3 | vector_unequal_width | FAIL | FAIL | 8.62 | [JSON](episodes/shared_c6__local_squared_silu64_l3__vector_unequal_width.json.gz) |
| local_squared_silu96_l3 | vector_unequal_mass | FAIL | FAIL | 8.71 | [JSON](episodes/shared_c6__local_squared_silu96_l3__vector_unequal_mass.json.gz) |
| local_squared_silu96_l3 | vector_unequal_width | FAIL | FAIL | 8.35 | [JSON](episodes/shared_c6__local_squared_silu96_l3__vector_unequal_width.json.gz) |
| local_product_softplus96_l2 | vector_unequal_mass | FAIL | FAIL | 8.82 | [JSON](episodes/shared_c6__local_product_softplus96_l2__vector_unequal_mass.json.gz) |
| local_product_softplus96_l2 | vector_unequal_width | FAIL | FAIL | 10.14 | [JSON](episodes/shared_c6__local_product_softplus96_l2__vector_unequal_width.json.gz) |

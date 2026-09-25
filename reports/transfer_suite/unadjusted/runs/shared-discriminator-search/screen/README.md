# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass | vector_unequal_width |
| --- | ---: | --- | --- |
| additive_raw_fourier64_l2 | 5796 | FAIL (0/24) | FAIL (0/24) |
| halfscore_fourier_skip96_l2 | 10467 | FAIL (0/24) | FAIL (0/24) |
| quadratic_softplus96_l2 | 9985 | FAIL (0/24) | FAIL (0/24) |
| quadratic_tanh96_l3 | 19297 | FAIL (0/24) | FAIL (0/24) |
| raw_silu128_l3 | 33537 | FAIL (0/24) | FAIL (1/24) |
| raw_softplus96_l3 | 19009 | FAIL (0/24) | FAIL (0/24) |
| residual_lowfreq_softplus96_l3 | 19393 | FAIL (0/24) | FAIL (0/24) |
| residual_raw_softplus96_l3 | 19009 | FAIL (0/24) | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| raw_softplus96_l3 | vector_unequal_mass | FAIL | FAIL | 9.68 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_unequal_mass.json.gz) |
| raw_softplus96_l3 | vector_unequal_width | FAIL | FAIL | 8.37 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_unequal_width.json.gz) |
| raw_silu128_l3 | vector_unequal_mass | FAIL | FAIL | 9.26 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_unequal_mass.json.gz) |
| raw_silu128_l3 | vector_unequal_width | FAIL | FAIL | 9.22 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) |
| quadratic_softplus96_l2 | vector_unequal_mass | FAIL | FAIL | 7.67 | [JSON](episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_mass.json.gz) |
| quadratic_softplus96_l2 | vector_unequal_width | FAIL | FAIL | 8.85 | [JSON](episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_width.json.gz) |
| quadratic_tanh96_l3 | vector_unequal_mass | FAIL | FAIL | 9.27 | [JSON](episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_mass.json.gz) |
| quadratic_tanh96_l3 | vector_unequal_width | FAIL | FAIL | 10.29 | [JSON](episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_width.json.gz) |
| residual_raw_softplus96_l3 | vector_unequal_mass | FAIL | FAIL | 9.69 | [JSON](episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_mass.json.gz) |
| residual_raw_softplus96_l3 | vector_unequal_width | FAIL | FAIL | 8.53 | [JSON](episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_width.json.gz) |
| residual_lowfreq_softplus96_l3 | vector_unequal_mass | FAIL | FAIL | 10.17 | [JSON](episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_mass.json.gz) |
| residual_lowfreq_softplus96_l3 | vector_unequal_width | FAIL | FAIL | 11.18 | [JSON](episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_width.json.gz) |
| halfscore_fourier_skip96_l2 | vector_unequal_mass | FAIL | FAIL | 7.55 | [JSON](episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_mass.json.gz) |
| halfscore_fourier_skip96_l2 | vector_unequal_width | FAIL | FAIL | 9.81 | [JSON](episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_width.json.gz) |
| additive_raw_fourier64_l2 | vector_unequal_mass | FAIL | FAIL | 9.80 | [JSON](episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_mass.json.gz) |
| additive_raw_fourier64_l2 | vector_unequal_width | FAIL | FAIL | 8.61 | [JSON](episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_width.json.gz) |

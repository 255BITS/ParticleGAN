# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_anisotropic | vector_overlap |
| --- | ---: | --- | --- |
| additive_raw_fourier64_l2 | 5796 | PASS (8/24) | FAIL (1/24) |
| quadratic_tanh96_l3 | 19297 | FAIL (4/24) | FAIL (0/24) |
| raw_silu128_l3 | 33537 | PASS (6/24) | FAIL (4/24) |
| raw_softplus96_l3 | 19009 | FAIL (2/24) | PASS (10/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| raw_silu128_l3 | vector_anisotropic | PASS | PASS | 15.72 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_anisotropic.json.gz) |
| raw_silu128_l3 | vector_overlap | FAIL | PASS | 9.20 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_overlap.json.gz) |
| quadratic_tanh96_l3 | vector_anisotropic | FAIL | PASS | 8.46 | [JSON](episodes/shared_c6__quadratic_tanh96_l3__vector_anisotropic.json.gz) |
| quadratic_tanh96_l3 | vector_overlap | FAIL | PASS | 8.66 | [JSON](episodes/shared_c6__quadratic_tanh96_l3__vector_overlap.json.gz) |
| raw_softplus96_l3 | vector_anisotropic | FAIL | FAIL | 10.35 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_anisotropic.json.gz) |
| raw_softplus96_l3 | vector_overlap | PASS | PASS | 7.75 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) |
| additive_raw_fourier64_l2 | vector_anisotropic | PASS | PASS | 8.50 | [JSON](episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) |
| additive_raw_fourier64_l2 | vector_overlap | FAIL | PASS | 9.44 | [JSON](episodes/shared_c6__additive_raw_fourier64_l2__vector_overlap.json.gz) |

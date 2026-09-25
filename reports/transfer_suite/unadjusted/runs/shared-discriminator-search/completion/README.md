# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_two_broad | vector_spiral |
| --- | ---: | --- | --- |
| raw_silu128_l3 | 33537 | FAIL (0/24) | FAIL (0/24) |
| raw_softplus96_l3 | 19009 | PASS (15/24) | PASS (9/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| raw_silu128_l3 | vector_two_broad | FAIL | FAIL | 10.28 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_two_broad.json.gz) |
| raw_silu128_l3 | vector_spiral | FAIL | PASS | 11.97 | [JSON](episodes/shared_c6__raw_silu128_l3__vector_spiral.json.gz) |
| raw_softplus96_l3 | vector_two_broad | PASS | PASS | 7.92 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_two_broad.json.gz) |
| raw_softplus96_l3 | vector_spiral | PASS | FAIL | 10.28 | [JSON](episodes/shared_c6__raw_softplus96_l3__vector_spiral.json.gz) |

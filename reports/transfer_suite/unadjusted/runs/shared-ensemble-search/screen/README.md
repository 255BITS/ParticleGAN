# Shared cap6 smooth ensemble discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass | vector_unequal_width |
| --- | ---: | --- | --- |
| ensemble2_equal_softplus64_l2 | 8834 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_highscale_softplus96_l2 | 19394 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_mixed64_l2 | 8834 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_multiscale_fullscore64_l2 | 8834 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_multiscale_halfscore64_l2 | 8834 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_multiscale_silu64_l3 | 17154 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_multiscale_softplus64_l2 | 8834 | FAIL (0/24) | FAIL (0/24) |
| ensemble2_multiscale_spectrum64_l2 | 9858 | FAIL (0/24) | FAIL (0/24) |
| ensemble3_broad_softplus48_l3 | 14691 | FAIL (0/24) | FAIL (0/24) |
| ensemble3_multiscale_silu64_l2 | 13251 | FAIL (0/24) | FAIL (0/24) |
| ensemble3_multiscale_softplus64_l2 | 13251 | FAIL (0/24) | FAIL (0/24) |
| ensemble4_broad_silu48_l2 | 10180 | FAIL (0/24) | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| ensemble2_equal_softplus64_l2 | vector_unequal_mass | FAIL | FAIL | 23.89 | [JSON](episodes/shared_c6__ensemble2_equal_softplus64_l2__vector_unequal_mass.json.gz) |
| ensemble2_equal_softplus64_l2 | vector_unequal_width | FAIL | FAIL | 9.18 | [JSON](episodes/shared_c6__ensemble2_equal_softplus64_l2__vector_unequal_width.json.gz) |
| ensemble2_multiscale_softplus64_l2 | vector_unequal_mass | FAIL | FAIL | 8.62 | [JSON](episodes/shared_c6__ensemble2_multiscale_softplus64_l2__vector_unequal_mass.json.gz) |
| ensemble2_multiscale_softplus64_l2 | vector_unequal_width | FAIL | FAIL | 9.90 | [JSON](episodes/shared_c6__ensemble2_multiscale_softplus64_l2__vector_unequal_width.json.gz) |
| ensemble3_multiscale_softplus64_l2 | vector_unequal_mass | FAIL | FAIL | 12.75 | [JSON](episodes/shared_c6__ensemble3_multiscale_softplus64_l2__vector_unequal_mass.json.gz) |
| ensemble3_multiscale_softplus64_l2 | vector_unequal_width | FAIL | FAIL | 12.90 | [JSON](episodes/shared_c6__ensemble3_multiscale_softplus64_l2__vector_unequal_width.json.gz) |
| ensemble3_multiscale_silu64_l2 | vector_unequal_mass | FAIL | FAIL | 11.57 | [JSON](episodes/shared_c6__ensemble3_multiscale_silu64_l2__vector_unequal_mass.json.gz) |
| ensemble3_multiscale_silu64_l2 | vector_unequal_width | FAIL | FAIL | 15.17 | [JSON](episodes/shared_c6__ensemble3_multiscale_silu64_l2__vector_unequal_width.json.gz) |
| ensemble2_multiscale_silu64_l3 | vector_unequal_mass | FAIL | FAIL | 10.47 | [JSON](episodes/shared_c6__ensemble2_multiscale_silu64_l3__vector_unequal_mass.json.gz) |
| ensemble2_multiscale_silu64_l3 | vector_unequal_width | FAIL | FAIL | 10.32 | [JSON](episodes/shared_c6__ensemble2_multiscale_silu64_l3__vector_unequal_width.json.gz) |
| ensemble3_broad_softplus48_l3 | vector_unequal_mass | FAIL | FAIL | 12.95 | [JSON](episodes/shared_c6__ensemble3_broad_softplus48_l3__vector_unequal_mass.json.gz) |
| ensemble3_broad_softplus48_l3 | vector_unequal_width | FAIL | FAIL | 13.41 | [JSON](episodes/shared_c6__ensemble3_broad_softplus48_l3__vector_unequal_width.json.gz) |
| ensemble4_broad_silu48_l2 | vector_unequal_mass | FAIL | FAIL | 11.17 | [JSON](episodes/shared_c6__ensemble4_broad_silu48_l2__vector_unequal_mass.json.gz) |
| ensemble4_broad_silu48_l2 | vector_unequal_width | FAIL | FAIL | 11.38 | [JSON](episodes/shared_c6__ensemble4_broad_silu48_l2__vector_unequal_width.json.gz) |
| ensemble2_highscale_softplus96_l2 | vector_unequal_mass | FAIL | FAIL | 10.02 | [JSON](episodes/shared_c6__ensemble2_highscale_softplus96_l2__vector_unequal_mass.json.gz) |
| ensemble2_highscale_softplus96_l2 | vector_unequal_width | FAIL | FAIL | 9.05 | [JSON](episodes/shared_c6__ensemble2_highscale_softplus96_l2__vector_unequal_width.json.gz) |
| ensemble2_mixed64_l2 | vector_unequal_mass | FAIL | FAIL | 7.88 | [JSON](episodes/shared_c6__ensemble2_mixed64_l2__vector_unequal_mass.json.gz) |
| ensemble2_mixed64_l2 | vector_unequal_width | FAIL | FAIL | 7.83 | [JSON](episodes/shared_c6__ensemble2_mixed64_l2__vector_unequal_width.json.gz) |
| ensemble2_multiscale_halfscore64_l2 | vector_unequal_mass | FAIL | FAIL | 7.96 | [JSON](episodes/shared_c6__ensemble2_multiscale_halfscore64_l2__vector_unequal_mass.json.gz) |
| ensemble2_multiscale_halfscore64_l2 | vector_unequal_width | FAIL | FAIL | 8.49 | [JSON](episodes/shared_c6__ensemble2_multiscale_halfscore64_l2__vector_unequal_width.json.gz) |
| ensemble2_multiscale_fullscore64_l2 | vector_unequal_mass | FAIL | FAIL | 9.23 | [JSON](episodes/shared_c6__ensemble2_multiscale_fullscore64_l2__vector_unequal_mass.json.gz) |
| ensemble2_multiscale_fullscore64_l2 | vector_unequal_width | FAIL | FAIL | 10.01 | [JSON](episodes/shared_c6__ensemble2_multiscale_fullscore64_l2__vector_unequal_width.json.gz) |
| ensemble2_multiscale_spectrum64_l2 | vector_unequal_mass | FAIL | FAIL | 10.03 | [JSON](episodes/shared_c6__ensemble2_multiscale_spectrum64_l2__vector_unequal_mass.json.gz) |
| ensemble2_multiscale_spectrum64_l2 | vector_unequal_width | FAIL | FAIL | 9.34 | [JSON](episodes/shared_c6__ensemble2_multiscale_spectrum64_l2__vector_unequal_width.json.gz) |

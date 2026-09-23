# Shared cap6 discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass | vector_unequal_width |
| --- | ---: | --- | --- |
| curvature_raw_silu128_l3_q32_w0p5 | 33729 | FAIL (0/24) | FAIL (0/24) |
| curvature_raw_silu128_l3_q32_w1p0 | 33729 | FAIL (0/24) | FAIL (0/24) |
| curvature_raw_silu128_l3_q64_w1p0 | 33921 | FAIL (0/24) | FAIL (0/24) |
| curvature_raw_softplus96_l3_q32_w0p5 | 19201 | FAIL (0/24) | FAIL (0/24) |
| curvature_raw_softplus96_l3_q32_w1p0 | 19201 | FAIL (0/24) | FAIL (0/24) |
| curvature_raw_softplus96_l3_q64_w1p0 | 19393 | FAIL (0/24) | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| curvature_raw_silu128_l3_q32_w1p0 | vector_unequal_mass | FAIL | FAIL | 13.91 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_mass.json.gz) |
| curvature_raw_silu128_l3_q32_w1p0 | vector_unequal_width | FAIL | FAIL | 12.28 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_width.json.gz) |
| curvature_raw_silu128_l3_q64_w1p0 | vector_unequal_mass | FAIL | FAIL | 14.59 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_mass.json.gz) |
| curvature_raw_silu128_l3_q64_w1p0 | vector_unequal_width | FAIL | FAIL | 15.34 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_width.json.gz) |
| curvature_raw_silu128_l3_q32_w0p5 | vector_unequal_mass | FAIL | FAIL | 12.27 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_mass.json.gz) |
| curvature_raw_silu128_l3_q32_w0p5 | vector_unequal_width | FAIL | FAIL | 12.28 | [JSON](episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_width.json.gz) |
| curvature_raw_softplus96_l3_q32_w1p0 | vector_unequal_mass | FAIL | FAIL | 11.51 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_mass.json.gz) |
| curvature_raw_softplus96_l3_q32_w1p0 | vector_unequal_width | FAIL | FAIL | 13.10 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_width.json.gz) |
| curvature_raw_softplus96_l3_q64_w1p0 | vector_unequal_mass | FAIL | FAIL | 13.15 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_mass.json.gz) |
| curvature_raw_softplus96_l3_q64_w1p0 | vector_unequal_width | FAIL | FAIL | 14.17 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_width.json.gz) |
| curvature_raw_softplus96_l3_q32_w0p5 | vector_unequal_mass | FAIL | FAIL | 11.66 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_mass.json.gz) |
| curvature_raw_softplus96_l3_q32_w0p5 | vector_unequal_width | FAIL | FAIL | 13.03 | [JSON](episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_width.json.gz) |

# Shared cap6 minibatch-feature discriminator research

Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.

| Discriminator | Parameters | vector_unequal_mass |
| --- | ---: | --- |
| batchfeat_center6_density_head | 19013 | FAIL (3/24) |
| batchfeat_center6_distance_head | 19013 | PASS (7/24) |
| batchfeat_center6_std_scalar | 19010 | FAIL (0/24) |
| batchfeat_center6_std_vector | 19105 | FAIL (0/24) |
| batchfeat_layer4_std_scalar | 19586 | FAIL (0/24) |

| D | Task | Live | EMA | Seconds | Artifact |
| --- | --- | --- | --- | ---: | --- |
| batchfeat_center6_std_scalar | vector_unequal_mass | FAIL | FAIL | 13.23 | [JSON](episodes/shared_c6__batchfeat_center6_std_scalar__vector_unequal_mass.json.gz) |
| batchfeat_center6_std_vector | vector_unequal_mass | FAIL | FAIL | 10.33 | [JSON](episodes/shared_c6__batchfeat_center6_std_vector__vector_unequal_mass.json.gz) |
| batchfeat_layer4_std_scalar | vector_unequal_mass | FAIL | FAIL | 11.63 | [JSON](episodes/shared_c6__batchfeat_layer4_std_scalar__vector_unequal_mass.json.gz) |
| batchfeat_center6_density_head | vector_unequal_mass | FAIL | FAIL | 17.99 | [JSON](episodes/shared_c6__batchfeat_center6_density_head__vector_unequal_mass.json.gz) |
| batchfeat_center6_distance_head | vector_unequal_mass | PASS | FAIL | 20.26 | [JSON](episodes/shared_c6__batchfeat_center6_distance_head__vector_unequal_mass.json.gz) |

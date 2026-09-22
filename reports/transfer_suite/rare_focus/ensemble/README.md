# Pointwise critic ensemble search

Same Rp logistic/b_cap3/kappa1.25/prior-reg.05, original beta99/LRs/G/256 particles/batch128. Only D architecture varies: arithmetic mean of 2 or 4 pointwise branches, one unchanged loss and optimizer. Extra D capacity is explicit. No data-derived statistics or labels. Seed0; final5 of24 live checks; EMA separate.

| D architecture | Toy | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | --- | ---: | ---: | --- |
| ensemble2_softplus5_d96 | vector_unequal_mass | FAIL | 0 | None | component_min_eigen_ratio=0, min_mass_ratio=0.20752 |
| ensemble2_softplus6_d96 | vector_unequal_mass | FAIL | 0 | None | component_min_eigen_ratio=0.019735 |
| ensemble4_softplus5_d96 | vector_unequal_mass | FAIL | 0 | None | component_covariance_error=0.85813, component_min_eigen_ratio=0.0060741 |
| ensemble4_softplus6_d96 | vector_unequal_mass | FAIL | 0 | None | component_min_eigen_ratio=0.037546 |

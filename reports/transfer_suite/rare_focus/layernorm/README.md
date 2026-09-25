# Pointwise LayerNorm discriminator search

Same Rp logistic/b_cap3/kappa1.25/prior-reg.05, original beta99/LRs/G/256 particles/batch128. Only D architecture varies. No batch-dependent normalization, data-derived statistics or labels. Seed0; final5 of24 live checks; EMA separate.

| D architecture | Toy | Sustained live | Final passing checks | Confirmed step | Final failing metrics |
| --- | --- | --- | ---: | ---: | --- |
| ln_pre_softplus_d64 | vector_unequal_mass | FAIL | 0 | None | sw1_normalized=0.27469, mass_tv=0.20208, component_covariance_error=6.3266, component_min_eigen_ratio=0, min_mass_ratio=0 |
| ln_pre_softplus_d96 | vector_unequal_mass | FAIL | 0 | None | component_covariance_error=0.96455, component_min_eigen_ratio=0, min_mass_ratio=0 |
| ln_post_softplus_d64 | vector_unequal_mass | FAIL | 0 | None | sw1_normalized=0.18912, component_covariance_error=2.2865, component_min_eigen_ratio=0, min_mass_ratio=0 |
| ln_post_softplus_d96 | vector_unequal_mass | FAIL | 0 | None | sw1_normalized=0.22256, mass_tv=0.16289, component_covariance_error=12.597, component_min_eigen_ratio=0, min_mass_ratio=0 |
| ln_pre_silu_d64 | vector_unequal_mass | FAIL | 0 | None | component_covariance_error=0.86231 |
| ln_pre_silu_d96 | vector_unequal_mass | FAIL | 0 | None | sw1_normalized=0.22488, component_min_eigen_ratio=0, min_mass_ratio=0 |

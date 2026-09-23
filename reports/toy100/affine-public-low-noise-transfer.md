# Small isolated output noise on the public-default F5 core

The exact noiseless F5/H1600 configuration passed all 19 older hosts. Three preregistered shared variants added only `output_noise_rng="isolated"`, a positive output-noise amplitude, and a provenance `name`; warmup remained .2, input noise remained zero, and seed offset 1901 was fixed. All three failed a ten-host frozen-gate screen, so none qualified for a fresh full-19 or native grid run.

| Output σ | Config SHA-256 prefix | Independently regraded result | Failed hosts |
| ---: | --- | ---: | --- |
| .005 | `536c534b` | 7/10 | vector_unequal_mass, vector_unequal_width, img_stripes2 |
| .010 | `cba09407` | 5/10 | trajectory, vector_unequal_mass, vector_unequal_width, vector_overlap, img_stripes2 |
| .015 | `7d736efa` | 7/10 | trajectory, vector_unequal_width, img_bars4 |

The decisive final misses were rare-component covariance eigenvalue ratios of .00056 and .12127 for unequal mass/width at σ=.005 (both require ≥.15); trajectory identity MSE .22175 at σ=.010 and .24216 at σ=.015 (requires ≤.02); unequal-width eigenvalue ratio .05624 at σ=.010 and .03560 at σ=.015; stripes2 HQ .84375 at σ=.010 (requires ≥.9); and bars4 at σ=.015 with only 3/4 modes and HQ .71875. The remaining failures had passing final metrics but lacked the required five consecutive checks: stripes2 suffix 3 at σ=.005, unequal_mass suffix 1 and overlap suffix 2 at σ=.010.

The manifest and complete evidence (`artifacts/toy100-accuracy/affine-noiseless/isolated-low-noise-screen-1c1a086/`) contain all exact configs, source archives, logs, ten compressed episodes per variant, and saved plus independent regrades. Regrading after relocation reproduced valid `FAIL 7/10`, `FAIL 5/10`, and `FAIL 7/10` with no provenance or receipt error. Source remained clean at `1c1a0865fe605c9f212d832c06596c12510f937e`; the original F5 config SHA-256 was `51f6bd91a60e523708b51a36bd2f38907a28bc138a1d0d1b366e6a6a200586b0`.

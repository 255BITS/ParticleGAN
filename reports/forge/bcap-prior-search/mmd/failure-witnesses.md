# Original sustained failure witnesses

One original failed check from each required terminal window. Endpoint scores alone do not establish a sustained pass.

Full original observation arrays stay in the artifact archive; their hashes and each selected packet hash are retained in [audit.json](audit.json).

## control: mode_hold, step 1050

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `2a3fb2c8b82e58a4c201be541468c6bd030f435d0f11d7ccc892fd5498546d45`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 8 | >= 8 | PASS |
| hq | 0.869385 | >= 0.9 | FAIL |

## control: vector_unequal_mass, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `8143a565fc37ea98879b2dd2b45ec9e2964e0c06bf1f7cb36943c47443e48cfe`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.102029 | <= 0.18 | PASS |
| mass_tv | 0.0616211 | <= 0.15 | PASS |
| hq | 0.964844 | >= 0.85 | PASS |
| component_covariance_error | 4.72483 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0067822 | >= 0.15 | FAIL |
| min_mass_ratio | 0.183105 | >= 0.25 | FAIL |

## control: vector_unequal_width, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `da35743a685a0a9d524894dd3bbad551657bd50b415cba60123862e2b2b1d250`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.280473 | <= 0.18 | FAIL |
| mass_tv | 0.297607 | <= 0.15 | FAIL |
| hq | 0.979492 | >= 0.85 | PASS |
| component_covariance_error | 5.9576 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.113667 | >= 0.15 | FAIL |

## control: vector_anisotropic, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `a85177b33f27ad550372755922c2fe9ef723ff117468d357e1d8a2c96efdd2c1`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.263079 | <= 0.18 | FAIL |
| mass_tv | 0.220052 | <= 0.15 | FAIL |
| hq | 0.838623 | >= 0.85 | FAIL |
| component_covariance_error | 9.8417 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.454452 | >= 0.15 | PASS |

## control: vector_overlap, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `a04a475f9ff9a75a8991e45b5609a2abd3ac11d82828b763953c1ba63d75ae65`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.400425 | <= 0.18 | FAIL |
| mean_error | 0.558847 | <= 0.15 | FAIL |
| covariance_error | 0.836653 | <= 0.45 | FAIL |

## control: img_bars4, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `f1f1342ff8839033fb648a7d05d223006fb80f88c3fdb23412ebcf1052817e77`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 2 | >= 4 | FAIL |
| hq | 0.90625 | >= 0.9 | PASS |

## control: img_blobs4, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `9ef6301c503c1fa6ceb85668165357f7388db87ddc65f6f163e06133b06b5c9f`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 3 | >= 4 | FAIL |
| hq | 0.84375 | >= 0.9 | FAIL |

## control: img_intensity2, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `f173a9b4f9dcba738bd896eed327106733fc025e2701624e6bd4943824fedda9`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 1 | >= 2 | FAIL |
| hq | 0.5625 | >= 0.9 | FAIL |

## 1.0: ring16_acquisition, step 1534

Required terminal window: [1534, 1550, 1567, 1584, 1600]; 5 passing checks required.

Original observation SHA256: `1128b16e71270fd5038783cbb2e65ea0991ef53e358cdbbe8f21bcb37c91017f`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0996094 | <= 0.15 | PASS |
| hq | 0.901855 | >= 0.85 | PASS |
| component_covariance_error | 1.96383 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.253778 | >= 0.15 | PASS |

## 0.1: mode_hold, step 1150

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `3244c95e5826359ca833dd011e1d383da491476c6f895fa6f173adcf19a8adeb`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 8 | >= 8 | PASS |
| hq | 0.875488 | >= 0.9 | FAIL |

## 0.1: vector_unequal_mass, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `f534c3cbf3c4138f045f5c1fa9fa6f387b6840a0645c72be9afe7f0ce2f410fa`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.12619 | <= 0.18 | PASS |
| mass_tv | 0.0736621 | <= 0.15 | PASS |
| hq | 0.989014 | >= 0.85 | PASS |
| component_covariance_error | 0.992828 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0267888 | >= 0.15 | FAIL |
| min_mass_ratio | 0.671387 | >= 0.25 | PASS |

## 0.1: vector_unequal_width, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `1a741ed9088445c2d51eaab9108e4af6c1ffe841f01f8e9b3b7fe08670db5dcf`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.27316 | <= 0.18 | FAIL |
| mass_tv | 0.260742 | <= 0.15 | FAIL |
| hq | 0.983154 | >= 0.85 | PASS |
| component_covariance_error | 11.3894 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0916025 | >= 0.15 | FAIL |

## 0.1: vector_anisotropic, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `16e1e9378ca895da6f489a3b0b90ec463e25d918d63c47bc4eeea73b2abd5fd7`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.273368 | <= 0.18 | FAIL |
| mass_tv | 0.250895 | <= 0.15 | FAIL |
| hq | 0.959473 | >= 0.85 | PASS |
| component_covariance_error | 2.42206 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.403139 | >= 0.15 | PASS |

## 0.1: vector_overlap, step 1000

Required terminal window: [1000, 1050, 1100, 1150, 1200]; 5 passing checks required.

Original observation SHA256: `eed6d28bd4ab9ceb98ecd046b1995098f8485d8e9e87316915527ca8a80924f9`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sw1_normalized | 0.135254 | <= 0.18 | PASS |
| mean_error | 0.19106 | <= 0.15 | FAIL |
| covariance_error | 0.164776 | <= 0.45 | PASS |

## 0.1: img_bars4, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `3edaed35fd73831960621ae5e5bbda551d2c6231cd0ffc157fe140f73e798b80`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 2 | >= 4 | FAIL |
| hq | 0.90625 | >= 0.9 | PASS |

## 0.1: img_blobs4, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `d60380281794e3cb098befb389e8f85754f943f2017da74fd82ddb35f64f1d7c`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 3 | >= 4 | FAIL |
| hq | 0.90625 | >= 0.9 | PASS |

## 0.1: img_intensity2, step 500

Required terminal window: [500, 525, 550, 575, 600]; 5 passing checks required.

Original observation SHA256: `123522c89e16cf06a8fbdb869efb4b07e9e5720daa01d7389535fc69d2f8b726`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| modes | 1 | >= 2 | FAIL |
| hq | 0.59375 | >= 0.9 | FAIL |

## 10.0: ae_gan_hold, step 219

Required terminal window: [209, 219, 230, 240, 250]; 5 passing checks required.

Original observation SHA256: `8c2dd0cb59bec49f9687d9cedd993754736ab6a7b9f55a0533f166839dc0661e`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| recon_mse | 0.0826803 | <= 0.05 | FAIL |
| hold | 0.00950167 | <= 0.35 | PASS |

## 10.0: ring16_acquisition, step 1534

Required terminal window: [1534, 1550, 1567, 1584, 1600]; 5 passing checks required.

Original observation SHA256: `a1245eef8dc218f6ff1874ea543891daaea6f9b488ba0b604791007b5377693d`.

| Measurement | Score | Target | Status |
| --- | ---: | --- | --- |
| sample_count | 4096 | >= 4096 | PASS |
| modes | 16 | >= 16 | PASS |
| mass_tv | 0.0349121 | <= 0.15 | PASS |
| hq | 0.758057 | >= 0.85 | FAIL |
| component_covariance_error | 5.21724 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.31388 | >= 0.15 | PASS |

Saved-only reproduction commands require the [mmd feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/e85cabc484140cd8afa3c44ce2abe1b93daf0443) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

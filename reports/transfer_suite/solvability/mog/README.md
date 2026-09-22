# Original-budget MoG training controls

Exactly16 actual GAN episodes, fixed seed0 and cosine schedule, original256 particles/1200 updates/architecture, unchanged task thresholds. Eight shared settings cross unequal-mass and unequal-width targets. This is separate from the target/representation-oracle audit. No final full PASS or sustained full PASS was found. All16 episodes completed24 observations with no errors.

The only substitution is the existing MoGParticlePrior constructor under a scoped runtime patch. Standardization and sigma calibration use the implementation in the exact source archive. Noise uses the original explicit training/evaluation RNG streams. Positive sigma therefore changes the realized sampling sequence as well as broadening each latent component. No source or threshold mutation occurred.

| MoG setting | Task | HQ | Covariance error ≤.85 | Min eigen ≥.15 | Final | Final passing suffix |
|---|---|---:|---:|---:|---|---:|
| mog_sigma0_std0 | vector_unequal_mass | 0.9502 | 5.8001 | 0.9366 | FAIL | 0 |
| mog_sigma0_std0 | vector_unequal_width | 0.9709 | 7.8086 | 0.7037 | FAIL | 0 |
| mog_sigma0.025_std0 | vector_unequal_mass | 0.9597 | 4.7586 | 1.3667 | FAIL | 0 |
| mog_sigma0.025_std0 | vector_unequal_width | 0.9875 | 1.0003 | 0.5583 | FAIL | 0 |
| mog_sigma0.1_std0 | vector_unequal_mass | 0.9641 | 1.7067 | 0.2935 | FAIL | 0 |
| mog_sigma0.1_std0 | vector_unequal_width | 0.9880 | 1.2519 | 0.5807 | FAIL | 0 |
| mog_sigma0.3_std0 | vector_unequal_mass | 0.9661 | 5.6805 | 1.0880 | FAIL | 0 |
| mog_sigma0.3_std0 | vector_unequal_width | 0.9651 | 2.0792 | 0.8113 | FAIL | 0 |
| mog_sigma0_std1 | vector_unequal_mass | 0.9802 | 0.6728 | 0.0828 | FAIL | 0 |
| mog_sigma0_std1 | vector_unequal_width | 0.9524 | 10.8880 | 0.7395 | FAIL | 0 |
| mog_sigma0.025_std1 | vector_unequal_mass | 0.9519 | 1.7522 | 0.0089 | FAIL | 0 |
| mog_sigma0.025_std1 | vector_unequal_width | 0.9575 | 8.8434 | 0.6029 | FAIL | 0 |
| mog_sigma0.1_std1 | vector_unequal_mass | 0.9832 | 1.0265 | 0.0802 | FAIL | 0 |
| mog_sigma0.1_std1 | vector_unequal_width | 0.9570 | 6.9666 | 0.8693 | FAIL | 0 |
| mog_sigma0.3_std1 | vector_unequal_mass | 0.9763 | 2.5403 | 0.8079 | FAIL | 0 |
| mog_sigma0.3_std1 | vector_unequal_width | 0.9302 | 8.6367 | 0.9414 | FAIL | 0 |

The closest unequal-mass spread result uses zero noise with standardized component means: covariance error.6728 passes but min-eigen.0828 fails. The lowest unequal-width covariance error is1.0003 using sigma_rel.025 without standardization; HQ.9875 and min-eigen.5583 pass, but the unchanged covariance bound remains failed. These are descriptive comparisons, not selected winners or evidence a larger untested MoG search cannot work.

Zero sigma without standardization is a semantic control: final live, EMA and all24 metric observations match the original atom-prior cosine runs exactly. Their original JSON files are included, with SHA256 recorded in baseline_atom_diagnostics.json.

Directly enumerated baseline particle outputs explain an observed failure: unequal_mass has actual assigned atom counts[131,81,38,6], with HQ counts[124,77,37,4]. The rare mode is present; two of its six atoms are off-mode and its sample covariance error is16.713. Unequal_width has atom counts[46,84,65,61], HQ counts[43,82,62,61]; the narrowest component has three off-mode atoms and covariance error26.273. This is concrete baseline tail evidence, not a claim about every candidate. Full generated-center coordinates are retained for all16 runs. For positive noise these are component-mean outputs, not every possible generated output.

Files: [protocol and exact cards](protocol.json.gz), [full summary](summary.json.gz), [scoped substitution source](run_mog.py), [numerical source archive](source.tar.gz), [baseline parity and atom evidence](baseline_atom_diagnostics.json.gz), [log](progress.log). Compressed per-episode files preserve original JSON bytes; archive_manifest.json records hashes. No tracked repository files changed.

# Overlap is not required for the observed FID regression

Both 80k-to-100k forks completed and source/config certificates plus final and best checkpoint hashes were verified. The missing unchanged100k quality probe also completed/certified, reproducing its historical FID to within0.00002. GPU1 is idle; no new training queued.

| Model | FID50k | Density | Coverage | Latent confusion |
|---|---:|---:|---:|---:|
| 80k parent | 15.7527 | 0.64686 | 64.26% | 0.0092% |
| 100k unchanged | 16.4609 | 0.59324 | 63.36% | 0.0244% |
| 100k freeze_centers | 16.5846 | 0.63840 | 65.09% | 0.0092% |
| 100k reduced_noise | 19.1835 | 0.60198 | 58.90% | 0.0824% |

Freezing live and EMA centers kept geometry and overlap exactly fixed, yet FID rose from15.7527 to16.5846. The unchanged100k control is16.4609, only0.1237 better. Frozen90k15.7493 is practically tied with the original80k score, not a meaningful breakthrough. Thus increasing overlap is not necessary for the regression over80k-to100k; this does not establish what causes all later160k deterioration.

Freezing does preserve better feature-space coverage:65.09% versus63.36% at matched100k, with density0.63840 versus0.59324. That is a useful tradeoff, not an FID solution or proof of semantic coverage. The original80k reference has64.26% coverage. Avoid calling the freeze arm an unconditional failure.

Sigma x0.75 training endsFID19.1835 and coverage58.90%, clearly worse than the unchanged100k run. Its initial sampling-only FID was15.9090, so the later deterioration is much larger than the initial distribution-change penalty. The reduced-noise prior is still highly distinguishable (only0.0824% latent nearest-center errors). Reducing sigma did not preserve FID, and it should not be extended blindly.

The unchanged100k prior has median nearest-center distance2.3309, above80k2.1637, and only0.0244% observed confusion. FID regression is already present before the large overlap seen at160k. Frozen and moving-center endpoints therefore both weaken a direct Gaussian-intersection explanation.

Recommendation: retain the original sigma and prioritize adversarial-update dynamics. A checkpoint fork from the original80k model with all learning rates halved, preserving ratios/state/architecture/objective, is a clean next low-cost test against the existing continuation. Lower-rate tests on older1k formulations failed; that history remains relevant, and this is not a promised fix. Current reconstruction updates E only, so it does not directly move G; with centers frozen the remaining G changes are adversarial. These results do not isolate D as the culprit or rule out architecture/capacity. Keep the frozen-center checkpoint as a useful coverage reference, but defer repulsion/new architectures and automatic long extensions until the next controlled result.

See curves.png, results.json, CHECKPOINTS.json, and the separate particle_overlap_control_100k/results.json. Matched RNG draws are diagnostic controls, not seed-only training experiments. Existing training sources remain unchanged.

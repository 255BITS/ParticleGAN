# Feature information and quality diagnostics

User requested adding the proposed metrics in a subagent. Implemented by `/root/particle_information_metrics`; root reviewed the estimator, cache/provenance handling and tests, and added experiment-pipeline integration. These are read-only checkpoint probes; historical trainers and objectives remain unchanged.

## Measurements

- **Decodable sibling bits:** conditional on an original parent, classify which child generated frozen Inception features. Train-only class centroids and pooled diagonal variance; validation chooses shrinkage and temperature, including a uniform predictor. Fresh test noise yields log2(K) minus test cross-entropy in bits. Negative estimates retained. This is an estimate of a restricted-decoder variational conditional-MI lower bound, not exact entropy or a guaranteed finite-sample lower bound. Parent standard error describes variation across sampled parents, not training-run uncertainty.
- **Controls:** independently shuffled balanced labels across train/validation/test; exact identical-feature clones within each split. The latter is a synthetic feature control, not a claim that real images are clones. It must yield zero additional bits.
- **Feature variance:** balanced nested ANOVA partitions feature sum of squares among original parents, siblings and within-child noise. Separate independent noise draws are used. Finite-sample between-mean terms contain estimation noise; fractions are descriptive, not semantic mode counts.
- **Density/coverage:** standard strict-radius kNN definitions from Naeem2020, k5, exact chunked squared Euclidean distances. Fixed10000 real CIFAR-train images and10000 generated images per checkpoint. Density is average number of real neighborhoods containing a fake divided byk; coverage is fraction of real neighborhoods containing a fake. Density can exceed1. Neither score alone establishes image quality or semantic coverage.
- **Overall feature moments:** generated/real feature-variance trace ratio and squared feature-mean distance from the same cached real reference.

Feature extractor: frozen torch-fidelity Inception2048, independently fixed from the GAN discriminator. Generator/extractor and distance computation use FP32 with TF32disabled; classifier/ANOVA/statistical moments use float64. Historical FID generation often enabledTF32, so diagnostic generation precision is explicit. No new FID is calculated: the tables attach the existing50k score for the exact checkpoint step.

## Comparison set and budgets

1.1024,4096,8192,16384 particles at20k from the same original10k parent.
2.1024 and4096 at40k.
3.4096 at35k (best observed16.5033) versus its40k endpoint17.2350.

Decoder panel uses the same32 original parents,64train/32validation/64test examples per child; variance uses16 additional draws per child. Per-child budgets are fixed while total work grows with the number of siblings. Original-parent and child-specific generation streams keep matching children/noise aligned across counts. Density/coverage always use equal total sample budgets. A locked, validated real-feature cache ensures all full probes use the exact same reference tensor; each run records its SHA256. Saved feature panels allow later CPU analysis without regenerating images.

## Validation and execution

Five CPU tests passed: density/coverage versus direct brute force and strict-boundary geometry; exact nested ANOVA decomposition; separable clusters approaching2bits with clone/shuffle nulls; test-data independence and negative-bit retention; factor1zero sibling information. Source and parent hashes, frozen-state checks, source.zip, config and pipeline certificates accompany each probe.

Queue waits for active8192/16384 scouts, reruns CPU tests, runs a small4096 GPU smoke, then launches seven full probes using one worker perGPU. A failed stage prevents later launches. Smoke uses128 real/fake images,2parents and4draws/split withk3; these numbers are not benchmarks. See LAUNCH.json/QUEUE_STATUS.json for the actual stage.

`tail -F runs/cifar_particle_ae/particle_information/PIPELINE.log`

Full pipeline writes results.json, LEADERBOARD.md and FINDINGS.md. Smoke detail log: `runs/cifar_particle_ae/particle_information_smoke/PIPELINE.log`. Parent checkpoints, original sources and training jobs are never modified.

## Interpretation

The compression hypothesis predicts additional decodable information accompanied by improved real-feature coverage while fidelity is retained. Higher bits alone can describe artifacts, and weak bits can describe an inadequate decoder. Compare quality proxies and the FID trajectory jointly; do not automatically select or train a winner based on information bits. The user continues to prioritize particle-count scaling towardFID50k<13.

References: [InfoGAN](https://arxiv.org/abs/1606.03657), [density/coverage](https://proceedings.mlr.press/v119/naeem20a.html). No seed experiments were added; seeded diagnostic sampling and shuffled-label controls do not repeat model training.

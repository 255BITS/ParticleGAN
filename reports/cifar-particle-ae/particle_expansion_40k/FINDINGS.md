# Particle expansion advantage persists, but the final FID rises

Both20k→40k continuations completed and certified. The4096 arm beats1024 at all four evaluations. It reaches a best observed FID50k16.5033 at35k, then rises to17.2350 at40k. The1024 control deteriorates to26.0216. A monotonic descent or solved plateau is not established.

| Arm | FID25k | FID30k | FID35k | FID40k | Test MSE40k | Train minutes |
|---|---:|---:|---:|---:|---:|---:|
|4096|17.8876|17.4796|16.5033|17.2350|0.15365|14.66|
|1024|21.8973|21.0193|21.3182|26.0216|0.15228|14.43|

The best observed score is3.5033 above the user's target13; the final checkpoint is4.2350 above it. Do not silently select the best point as the endpoint. Both initial model distributions were matched at10k, and both respective20k checkpoints were resumed with full state, identical rates and one D update. Relative training cost remains small (~1.6%).

At40k expanded sibling latent RMS is0.17535 per coordinate, or0.825 sigma. Between-sibling coupled-noise feature variation is0.03090 versus within-child variation0.02776. Output differences persist; these are not semantic-coverage measurements. Both final sample grids were inspected: varied images with visible shape/detail errors; no total-collapse claim. Control feature-variance trace ratio is1.0974 versus real reference, expanded1.0746, illustrating that overall feature variance alone cannot explain the FID gap.

User explicitly prioritizes further particle-count scaling based on earlier smaller experiments.8192/16384 scouts are queued/launching from the same original10k parent, reusing1024/4096 benchmarks. No Gaussian-prior comparison or automatic200k promotion. Additional read-only feature-information, density/coverage and variance diagnostics are being implemented at the user's request to clarify how added particles improve the distribution.

Completed pipeline PID266873; details in LEADERBOARD.md/results.json. Source/config certificates, frozen-feature/sigma checks and original data/noise RNG pairing passed. Full numbered checkpoints at25k/30k/35k/40k are available in each run directory. Examine the larger-count curve before deciding the next training duration.

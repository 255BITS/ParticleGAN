# Hopfield particle-prior study

This is a one-seed-per-arm screening study. The original ≥2-of-3-seeds success criterion cannot be assessed; no seed mean ± sd is reported.

**Kill test: STOP.** Uniform final TV = 0.2121; Hopfield β=16 final TV = 0.4229. Improvement = -0.2109, required ≥0.2000.

Recommendation: stop before the full grid. At this recipe, M=100 and this seed, the proposed retrieval fails its predeclared reweighting gate. This does not rule out other temperatures, larger memories, or eventual baseline convergence.

Measured true-mixture TV noise floor for `imbalanced_m100_uniform_s1234`: 0.00998 ± 0.00097 (20 draws of 100,000 samples). This repeated-sampling uncertainty is distinct from training-seed uncertainty.

`imbalanced_m100_uniform_s1234` has near-delta within-mode spread (σ ratio 0.0580); its better TV does not make it a successful generative model.

Measured true-mixture TV noise floor for `imbalanced_m100_hopfield_b16_s1234`: 0.00998 ± 0.00097 (20 draws of 100,000 samples). This repeated-sampling uncertainty is distinct from training-seed uncertainty.

`imbalanced_m100_hopfield_b16_s1234`: the large core σ ratio does not establish healthy Gaussian spread: it averages nearest-center widths over modes, including diffuse low-mass assignments; interpret it with coverage and HQ; effective retrieval is close to one particle per query; 70.0% of particles never win an argmax in evaluation; query interpolation misses the 0.90 HQ target.

Lowest-TV measured imbalanced Hopfield arm: `imbalanced_m100_hopfield_b16_s1234` (TV 0.4229; quality floor fails). Ranking by TV alone does not establish study success.

`steps_to_tv` is the start of the final uninterrupted sequence of evaluation points with TV <0.03. `>7000` means no sustained convergence was observed within that horizon; it does not imply convergence is impossible.

All read-health and quality metrics refer to the EMA read. Uniform read-health and interpolation values are N/A. HQ-conditional TV can conceal dropped low-quality mass, so always read it together with HQ, coverage and σ ratio.

Artifacts: [leaderboard](TABLE.md), [TV curves](tv_vs_step.png), [read entropy diagnostic](eff_n_vs_step.png), [mode weights](mode_weights.png).

Scatter animations: [imbalanced_m100_uniform_s1234.gif](imbalanced_m100_uniform_s1234.gif), [imbalanced_m100_hopfield_b16_s1234.gif](imbalanced_m100_hopfield_b16_s1234.gif).

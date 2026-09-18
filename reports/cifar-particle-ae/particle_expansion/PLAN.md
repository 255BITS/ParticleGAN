# Planned particle-expansion experiments

Status: planning only; no implementation or training launched. User asked to plan the next experiments before compacting. Target remains sustained CIFAR FID50k improvement toward <13. The particle-support explanation is a hypothesis, not an established cause.

## Question and predictions

Do 1024 learned centers constrain the number of distinct image configurations G learns? Grouped samples retain similar scenes within a particle; local noise is used, but doubling inference noise worsens FID. Test more independently trainable centers without simultaneously changing G/D architecture or noise scale.

Support for this hypothesis requires improved matched FID trajectories together with increased image differences between descendant centers. More centers or larger latent distances alone are insufficient. If descendants remain near-identical, a negative short scout may indicate slow symmetry breaking rather than adequate support. If they separate but FID does not improve, feedback quality or another constraint becomes more plausible. We cannot establish a hard mode-count ceiling from the current grids.

## Shared checkpoint and fixed recipe

Start from the original scratch CNN E-only 10k checkpoint:
`runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`
SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`; FID50k 19.4482.

Use original G LR 0.0003 (not the failed half-G setting), E 0.0003, prior 0.003, D 0.00045; one D step, bcap coefficient1 every8 x8, E-only reconstruction, EMA0.995. Preserve model/Adam/EMA/RNG state. Same training batch size and evaluation protocol; no seed experiments. Historical full-reconstruction CNN need not be retrained.

## Wave 1: control versus additional trainable centers

| GPU | Arm | Intervention |
|---|---|---|
| 0 | control_1024 | Exact unchanged continuation |
| 1 | split_4096 | Four initially coincident descendants per original particle, independently trainable thereafter |

Continue 10k -> 20k; FID50k at15k/20k, with numbered full checkpoints. Compare both evaluations with the initial checkpoint. Keep cost near the existing ~10-minute two-GPU scout if throughput permits; measure actual expansion overhead before quoting a runtime. Each center receives fewer direct selections at fixed batch size, so log selection counts and avoid interpreting a brief failure as proof of sufficient capacity.

Cloning avoids imposing new global image modes at initialization. Independent sample assignments/noise should break symmetry during training, but this must be measured. Preserve original prior Adam step count and copy each parent's per-row first/second moments into its descendants; do not reset G/D/E optimizers or multiply prior LR. Record that expansion changes per-row sampling frequency and optimizer dynamics: this is not a pure parameter-count intervention with every dynamical quantity held constant.

## Mandatory implementation preflight

Use a new standalone trainer, preserving all historical/shared sources and their certificates. No new trainer exists yet.

1. **Unchanged control replay:** exact full-state continuation under deterministic test settings, as in the previous round. Default settings must reproduce historical behavior.
2. **Normalization:** `prior.means()` divides by unbiased std plus1e-6. Repeating raw rows changes that std. Make duplication preserve centers mathematically, and measure floating-point residuals. One candidate is a persisted reference count N=1024: at expanded count M use `std_M * sqrt(N*(M-1)/(M*(N-1)))` before adding epsilon. At M=N call the original path exactly. This preserves the old denominator when each row is repeated equally. Audit the live and EMA priors independently. This is a proposed implementation, not validated code.
3. **Fixed sigma:** retain saved sigma and d0; never recalibrate nearest-neighbor distances after cloning (coincident descendants would produce zero distances). Preserve calibration metadata and loaded noise-enabled state.
4. **Optimizer and regularizer:** verify G/D/E states unchanged, prior/EMA rows and per-row Adam tensors mapped correctly, step counts preserved, and no aliases between trainable descendant rows. Particle regularization also uses sample-count-dependent variance/covariance estimators. Quantify its cloning-time value/gradient differences; account for them or explicitly document them before attributing effects solely to support.
5. **Sampling:** use coupled parent IDs/noise for the identity test; verify uniform weighting of all 4096 descendants. Prefer a separate saved clone-selection RNG stream so additional choices do not shift the original data/noise draws. Preserve evaluation pairing similarly where feasible. Distinguish identical distribution from identical finite RNG realization.
6. **Real-checkpoint smoke:** 8–64 updates per arm through the pipeline; validate frozen backbone, sigma, finite states, correct recipient gradients, resume/certificate checks and independent descendant updates. Audit original checkpoint hashes unchanged.
7. **Initial image/FID audit:** compare expanded EMA output with parent using coupled inputs, then original FID50k protocol before joint training. If identity fails materially, fix or report that initialization intervention explicitly; do not hide it in a later training gain.

No approval gate is implied by these checks: when the user resumes experimental work, implement and validate within that authorization. This turn is planning-only because the user is compacting.

## Diagnostics accompanying FID

Use a fixed panel of original parent IDs with all four descendants and multiple noise draws. Save grids with sibling centers grouped together at initialization and each evaluation. Measure latent sibling distances, between-sibling image/feature variation, and within-descendant noise variation. Compare with a matched four-draw grouping of the unsplit control; record that feature distances are not semantic coverage scores. Retain timing, per-center sampling exposure, G/D/prior gradient/update summaries and test reconstruction as secondary diagnostics. Avoid expensive continuous instrumentation in the FID training loop.

Do not rank by reconstruction, D AUC, a single selected minimum, or latent separation alone. Report a FID leaderboard, trajectories, training/wall cost and whether descendants actually differentiated.

## Conditional wave 2

- **Clear, sustained FID gain:** continue both saved endpoints to40k (evaluate every5k or10k), preserving their states, to test persistence and diminishing returns. Consider 8192 particles only after 4096 demonstrates useful improvement. A move toward200k needs sustained evidence; do not automatically promote a small/reversing gain.
- **No gain and descendants barely separate:** test a 4096 arm with small balanced sibling perturbations against an unperturbed 4096 continuation. Suggested initialization scale is per-coordinate RMS0.25 times saved sigma, zero mean across each sibling group. Keep component noise sigma fixed. Audit/report its initial FID and retain comparable optimizer state. This tests initialization/symmetry breaking, not a repeat with another seed.
- **Descendants separate but no useful FID gain:** prioritize a controlled discriminator feedback/robustness experiment. Dense bcap remains untested in matched joint FID; D-only results do not settle it. Broader training noise is also distinct from the failed inference-noise change, but avoid bundling it into the particle-count test.

## Planned artifacts and operational notes

Suggested tracks: `particle_expansion_smoke`, `particle_expansion_scout`. Use fresh directories, the experiment pipeline and one worker per GPU. Provide a single log suitable for `tail -F runs/cifar_particle_ae/particle_expansion_scout/PIPELINE.log`. Save configuration/source/parent hashes, expansion mappings, initial-distribution audit, full resumable checkpoints, results.json, LEADERBOARD.md and FINDINGS.md. These paths are proposed; no configs or jobs have been created.

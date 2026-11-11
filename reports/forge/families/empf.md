<!-- Generated Forge family report -->

# EMPF

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

Exponential Moving Prototype Flow (EMPF) is a clock-free, constant-learning-rate ParticleGAN formulation inspired by e-means (Bessa et al., arXiv:2611.09501) with no EMA critic network copy. Instead of computing transport chords solely against noisy raw mini-batches, EMPF maintains a checkpointed online sufficient-statistics prototype bank (S, N) updated with clock-free (K, B) hit-rate EMA scaling and deterministic mass-conserving proportional-split dead-prototype revival, couples non-uniform prototype masses to generated samples via scale-free entropic Optimal Transport, bounds critic gradients along endpoints and prototype chords with an Elastic Born-Infeld potential and zero-mean logit gauge, and regularizes trainable particle priors with a degree-of-freedom-aware EMPF tether/repulsion/revival law.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior with $M$ locations in $\mathbb R^{d_z}$, $x$ a real batch of size $B_r$, $z\sim P$ a latent batch and $y=G(z)$ the generated batch of size $B_f$. $(S, N)$ is the online prototype bank of size $K=\min(K_{\max}, B_r)$ with centroids $C_k=S_k/N_k$ and normalized row masses $w_k=N_k/\sum_j N_j$. $\Pi$ is the entropic Optimal Transport coupling matching row marginals $w$ to uniform column marginals $1/B_f$, $x_{\mathrm{chord}}=(1-u)y+u C_{\pi}$ is a stratified prototype transport chord, and $g_r, g_f, g_c$ are critic input gradients.

**Clock-Free Online e-means Sufficient-Statistics Prototype Update**

$$
S_k^{(t)}=(1-\alpha_{\mathrm{eff}})S_k^{(t-1)}+\alpha_{\mathrm{eff}}\sum_{i:a_i=k}x_i,\quad N_k^{(t)}=(1-\alpha_{\mathrm{eff}})N_k^{(t-1)}+\alpha_{\mathrm{eff}}\,|\{i:a_i=k\}|,\quad \alpha_{\mathrm{eff}}=\operatorname{clip}\!\left(\alpha_0(B_r/K)^\eta,\alpha_{\min},\alpha_{\max}\right)
$$

Maintains denoised online centroids $C_k=S_k/N_k$ and normalized mode-mass shares $w_k=N_k/\sum_j N_j$ without dataset-size or epoch-horizon clocks; prototypes with $N_k<\tau_{\mathrm{rev}}\bar N$ split the heaviest donor deterministically.

**Prototype-Chord Elastic Born-Infeld Penalty and Zero-Mean Gauge**

$$
R_{\mathrm{EMPF}}=\frac{\lambda_{\mathrm{floor}}}{6d}\mathbb E\!\left[\lVert g_r\rVert_2^2+\lVert g_f\rVert_2^2+\lVert g_c\rVert_2^2\right]+\frac{\lambda_{\mathrm{cap}}}{2}\mathbb E\!\left[\phi_\beta(\lVert g_r\rVert_2-\kappa)^2+(1-\alpha_c)\phi_\beta(\lVert g_f\rVert_2-\kappa)^2+\alpha_c\phi_\beta(\lVert g_c\rVert_2-\kappa)^2\right]+\frac{\gamma_{\mathrm{gauge}}}{2}\bar D^2
$$

Bounds critic input-gradient norms along real endpoints, fake endpoints and mass-weighted prototype Sinkhorn chords $x_{\mathrm{chord}}=(1-u)y+u C_\pi$ while pinning translational logit drift via $\bar D=\frac{1}{2}(\mathbb E[D(x)]+\mathbb E[D(y)])$.

## Simplified pseudocode

```text
Choose an adversarial loss, optimizer and constant learning rates.
For each task-owned training update:
  Sample real x and latent z from the task prior P; compute y=G(z).
  On unconditioned continuous vector domains, assign x to nearest centroids C_prev = S / N, update (S, N) with hit-rate EMA rate alpha_eff(K, B_r), and revive any starved prototype (N_k < tau_rev * mean(N)) via deterministic half-mass donor splitting.
  Compute mass-weighted scale-free Sinkhorn coupling Pi between (C, w) and stop_gradient(y); on conditioned critics, bypass the marginal bank and align chords per conditioning fiber.
  Sample stratified transport chords x_chord = (1 - u) stop_gradient(y) + u C_pi.
  Evaluate Elastic Born-Infeld potential on input-gradient norms at x, stop_gradient(y) and x_chord plus zero-mean logit gauge, and update instance-bound state-driven thermal scale.
  Backpropagate critic adversarial loss + EMPF penalty and step D.
  With D fixed, recompute differentiable fake samples y=G(z) and evaluate generator adversarial loss + EMPF prior regularizer.
  Step G, trainable prior P and encoder E when present using constant learning rates.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Selectable non-saturating logistic (selected DualNorm configuration), paired relativistic logistic, EMPF Sinkhorn relativistic logistic, hinge, Wasserstein or least squares. Joint encoder/generator hosts train both critic streams. |
| Optimizer | Supports DualNorm (selected configuration: zero momentum, smoothing 0.001, per-offset quintic convolution, batched SVD backend, BF16 channels-last conv training, direction-blend constraint geometry) as well as native Adam and formulation-owned EMPFCriticAdam without an EMA critic network copy. |
| Learning rates and annealing | Strictly constant (`lr_floor=1.0`, `network_lr_floor=1.0`): global LR $0.012$, critic multiplier $1.5$ and DualNorm sampled-row prior multiplier $2.5$ (`prior_reg=0.01`), matching `bispf-throughput-v1` / `presets[bispf_optimized]`. |
| Parameter-gradient clipping | None. Critic spike guard is disabled; gradient bounding acts smoothly through the Elastic Born-Infeld critic potential. |
| Critic penalties and anchors | Prototype-chord Elastic Born-Infeld regularizer evaluated on real endpoints, fake endpoints and mass-weighted prototype Sinkhorn chords ($\kappa=1$, sharpness $\beta=4$, harmonic floor $\lambda_{\mathrm{floor}}=0.01$, chord blend $0.05$) plus a settling zero-mean logit gauge ($\gamma_{\mathrm{gauge}}=0.05$). Conditioned critics preserve fiber alignment automatically; no EMA critic anchor. |
| Damping and update guards | Instance-bound state-driven smoothstep thermal cooling and 2D harmonic basin capture lock damp post-convergence mini-batch fluctuations below DualNorm's smoothing floor using zero global state and zero step counters. |
| Training and sampling noise | No scheduled additive critic-input or generator-output training noise in the selected benchmark configuration ($0.0$). Task-declared MoG intrinsic sigma is preserved. |
| Parameter averaging and serving | Generator EMA decay=$0$ and serve_average=$0$; all evaluations score live unaveraged weights. |

## Configuration differences

- EMPF translates the online sufficient-statistics EMA and proportional-split dead-unit revival of e-means into a clock-free adversarial transport formulation without dataset-size or epoch-horizon dependencies.
- All prototype bank tensors (proto_sum, proto_count), low-discrepancy phases and thermal scalars live exclusively on the critic optimizer's instance-bound step record and travel bit-for-bit through state_dict() and load_state_dict() with strict corruption rejection.
- The EMPF prior regularizer skips global moment/covariance distortion when prior row capacity is at or below the degree-of-freedom threshold (returning exact zero on unanchored small codebooks and applying only a short-range one-sided prototype non-collision hinge on reference-anchored small discrete codebooks) while applying initial-moment tethering, scale-free k-NN log-repulsion, local scatter circularization, and a local median-normalized 1-NN revival tether on continuous particle clouds.
- One global trainer configuration supplies every displayed result across Tier 1, Tier 2 and Tier 3; task-owned architectures, priors, initializations, budgets and gates remain fixed.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/empf.py](../../../particlegan/empf.py)
- [particlegan/_empf/prototypes.py](../../../particlegan/_empf/prototypes.py)
- [particlegan/_empf/transport.py](../../../particlegan/_empf/transport.py)
- [particlegan/_empf/critic.py](../../../particlegan/_empf/critic.py)
- [particlegan/_empf/penalty.py](../../../particlegan/_empf/penalty.py)
- [particlegan/_empf/prior.py](../../../particlegan/_empf/prior.py)
- [particlegan/_empf/loss.py](../../../particlegan/_empf/loss.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [configs/forge/ideas/empf-v1.json](../../../configs/forge/ideas/empf-v1.json)
- [configs/forge/studies/empf-v1-study.json](../../../configs/forge/studies/empf-v1-study.json)
- [reports/forge/empf/README.md](../empf/README.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

## Performance

DCGAN on deterministic mock RGB data measures finite-update speed and memory. These separate performance-source receipts confer no image-quality, scientific qualification or default-promotion credit. Standard BatchNorm is retained in generator and critic; critic input gradients therefore include batch coupling. The shared critic uses a flattened 4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.

Synchronized wall time for one public trainer update, including its critic/generator/prior work; warmup, construction, mock-data generation, evaluation, GIFs and checkpoint writes are excluded.

Every cell is one run; timings are never pooled across devices or contention states. Compare only rows with the same task and matched key (source, runtime/dtype, physical device, initialization and batch streams). Unknown or contended observations are unpaired. Resolution progression changes the task, while each candidate keeps one global recipe. Images/s is B divided by full update time (one D and one G/prior update with separate real batches). Three warmup and ten measured updates give short-window median/p95 measurements; long-horizon policy costs and component phase times remain unknown.

[Compact performance evidence index](../scaling-performance.json) · [Protocol and bounded reproduction](../../../docs/forge-scaling-performance.md)

This registration covers 9 complete recipes. Historical alternatives outside its registered family roster add no missing benchmark cells. A changed selected recipe within a registered family remains unknown until measured.

### NVIDIA RTX A6000 · GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce

Performance source `87863e1818c8` (commit `cb5e484b059d3474f7259c40beb7acb4b5bb8704`); runtime `fddddb1b314d`; torch.float32; PyTorch 2.14.1+cu130; 48592.125 MiB device memory.

Task prior: particle_cloud, sigma 0.0, 256 initial locations. The common finite-table performance exception retains each recipe's prior update policy and regularizer. Exact prior and resource specifications remain part of every task key.

| Family / recipe | Task (RGB size, batch, width) | Execution | Median / p95 update ms | Images/s | GPU peak allocated / reserved MiB | Contention | Matched key / evidence |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| EMPF · `empf-v1` | dcgan_performance_r32: 32², B16, W64 | COMPLETE | 24.89 / 25.93 | 636.83 | 168.71 / 258.00 | clear | `36faa7c217f8` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_r32-ec526610685681bb.json) |
| EMPF · `empf-v1` | dcgan_performance_r64: 64², B16, W64 | COMPLETE | 33.80 / 34.39 | 472.39 | 336.58 / 544.00 | clear | `1c2f26889df3` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_r64-2e546d20e319cb51.json) |
| EMPF · `empf-v1` | dcgan_performance_w128: 64², B16, W128 | COMPLETE | 64.07 / 64.80 | 250.14 | 923.03 / 1,528.00 | clear | `55d777bc34a2` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_w128-b6b1c350f7a0ed6f.json) |
| EMPF · `empf-v1` | dcgan_performance_b64: 64², B64, W64 | COMPLETE | 39.19 / 40.17 | 1,628.56 | 486.10 / 622.00 | clear | `fe63f8d2a846` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_b64-beef11b5a3196138.json) |
| EMPF · `empf-v1` | dcgan_performance_r128: 128², B16, W64 | COMPLETE | 66.64 / 68.13 | 239.09 | 648.91 / 1,074.00 | clear | `a5a292259144` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_r128-e1cd603feeb6647e.json) |
| EMPF · `empf-v1` | dcgan_performance_r256: 256², B16, W64 | COMPLETE | 156.88 / 157.12 | 102.03 | 1,522.16 / 1,890.00 | clear | `ad1b8aee803c` · [receipt](../scaling-receipts/dcgan-performance-cloud-1acd41cb5f4a/empf-dcgan_performance_r256-54ca73c41f953f72.json) |

<a name="cohort-cuda-23a94f16ca88"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [empf-v1](../../../configs/forge/ideas/empf-v1.json).

Comparison policy: discriminator_stability revision 9, **tier 1**. Actual executed-policy qualification: **tier 1**, discriminator_stability revision 16 (FAIL). These are separate policy results; no qualification reuse.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `1acd41cb5f4ab3b459816a561582fb03c5062722eff343d2b1f3f81e9f3c637c`. Candidate revision: `45a5684f2961affd93bfb3cdd35b79928763c60cfe6a3e913492dc1bc77007e4`. Runtime cohort: `e4a77817f82de838383535a53128c330f372a6f35123e3a5877085e5d22cbd6d`.

[Frozen numerical evidence](../technique-evidence/a90a3a12cbce7cda67a62002af58aa0a00e248ed6d6fb30f8dd6798a3719bd55.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Select empf-v1 (Exponential Moving Prototype Flow: online e-means sufficient-statistics prototype bank (S, N) with (K, B) hit-rate EMA scaling and proportional-split revival, mass-weighted prototype-to-fake Sinkhorn OT, Elastic Born-Infeld transport-chord critic penalty with zero-mean logit gauge, and DoF-aware EMPF prior regularizer) following ordinary discriminator_stability CUDA measurement on NVIDIA RTX A6000 (11/11 required + 1/1 clockfree_audit diagnostic Tier 1 PASS, 8/18 required Tier 2 PASS, 20/38 executed tasks PASS overall). Actual executed-policy qualification and comparison-policy projection retain their separate source-bound proofs; no public-default adoption claim.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation) | [3/3](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) | [8/12](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) | [0/1 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-4) | [11/16 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-adaptation) | [0/7](empf.md#cohort-cuda-23a94f16ca88-held) |
| [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous) | [4/4](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) | [8/12](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) | [0/6 (3 blocked, 3 diagnostic attempts)](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-4) | [12/22 (3 blocked, 3 diagnostic attempts)](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous) | [0/7](empf.md#cohort-cuda-23a94f16ca88-held) |
| [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability) | [11/11](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | [8/18 (2 diagnostic attempts)](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | [0/1 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | [19/30 (3 diagnostic attempts)](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability) | [0/25 (1 incompatible, 15 diagnostic attempts)](empf.md#cohort-cuda-23a94f16ca88-held) |
| [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison) | [3/3](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) | [8/12](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) | [0/1 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-4) | [11/16 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison) | [0/7](empf.md#cohort-cuda-23a94f16ca88-held) |
| [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer) | [3/3](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) | [8/12](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) | [0/1 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-4) | [11/16 (1 diagnostic attempt)](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer) | [0/7](empf.md#cohort-cuda-23a94f16ca88-held) |
| [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage) | [3/3](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | [8/12](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | [0/0](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-4) | [11/15](empf.md#cohort-cuda-23a94f16ca88-quality_coverage) | [0/7](empf.md#cohort-cuda-23a94f16ca88-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Tier 4 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| [lazy_cadence_coverage](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage) | [0/0](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) | [0/0](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-2) | [0/0](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-4) | [0/0](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage) |
| [tier1_policy_coverage](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage) | [0/7 (7 not run)](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1) | [0/0](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-2) | [0/0](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-3) | [0/0](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-4) | [0/7 (7 not run)](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

Certified execution repair: [93c315940834406e83f1cee458f7d399](../technique-receipts/93c315940834406e83f1cee458f7d399.json) → [9998a00908e447449e6c439fe01a9243](../technique-receipts/9998a00908e447449e6c439fe01a9243.json). The original outcome and cost remain recorded; displayed task metrics use the certified successor.

Certified execution repair: [a85d374a9cf040d0948fe0cc9ee7b553](../technique-receipts/a85d374a9cf040d0948fe0cc9ee7b553.json) → [c8a3db850fb64fa4a9a5c5177dc82d37](../technique-receipts/c8a3db850fb64fa4a9a5c5177dc82d37.json). The original outcome and cost remain recorded; displayed task metrics use the certified successor.

<a name="cohort-cuda-23a94f16ca88-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_gan_multimodal](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_ae_reconstruction](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_categorical](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [model_vae_hard](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1) | PASS | matches recorded run |
| [zero_tanh_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [precision_gaussian_mxfp8](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [precision_gaussian_nvfp4](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ae_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [model_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [model_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | [adaptation](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-3), [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [zero_tanh_rail_escape](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | BLOCKED | matches recorded run |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [clockfree_continuous](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3), [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3), [formulation_comparison](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3), [host_profile_transfer](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN |

<a name="cohort-cuda-23a94f16ca88-tier-4"></a>

### Tier 4 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [alice_spiral_bigan](empf.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](empf.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](empf.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_mlp16](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](empf.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_block_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan8_grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [discriminator_stability](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [precision_gaussian_mxfp8](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format MXFP8 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [precision_gaussian_nvfp4](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) | FAIL | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | Can one unchanged family recipe learn the scalar Gaussian with Blackwell-format NVFP4 software-reference training arithmetic? | Consider counted status only after native parity, independent calibration and a global recipe that retains every counted pass. |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2), [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2), [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2), [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [zero_tanh_rail_escape](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [discriminator_stability / Tier 3](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Dynamic target shift (+0.90, -0.90) -> (0.0, 0.0) rail escape with 400-update deadline held pending passing recipe. | Count again when one global recipe passes zero_tanh_rail_escape under its unchanged declaration and satisfies all Tier 1-3 requirements. |
| [alice_spiral_bigan](empf.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | ALICE cycle-consistent BiGAN (G, E, D_xz) on a 1D curved spiral manifold in 2D testing cycle reconstruction. | — |
| [cgan_overlapping_classes](empf.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Mirza-Osindero concat vs Miyato-Koyama projection cGAN on 4 overlapping annular rings testing class separability. | — |
| [cramer_energy_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Bellemare CramerGAN 16D critic embedding with surrogate energy potential on a tight 25-mode 5x5 Gaussian grid. | — |
| [cyclegan_unpaired_constellation](empf.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Unpaired CycleGAN/DiscoGAN dual generators and cycle-consistency on rotated and scaled 8-mode constellations. | — |
| [grid100_mlp16](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-hidden-layer 128-wide LeakyReLU MLP generator on grid100 target held pending passing recipe across all 100 modes. | Count again when one global recipe passes grid100_mlp16 under its unchanged declaration and satisfies all counted requirements. |
| [grid100_res16_scaled_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Paired 16-layer ScaledResidualMLPGenerator + TanhRailGuard on the 10x10 100-mode Gaussian grid isolating residual depth transport from unscaled initialization explosion. | — |
| [grid100_res16_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 16-layer residual MLP with S=5.25 bounded tanh head on grid100 target held pending non-saturating passing recipe. | Count again when one global recipe passes grid100_res16_tanh under its unchanged declaration without saturation. |
| [infogan_anisotropic_modes](empf.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | InfoGAN mutual-information auxiliary head Q(c\|x) on 8 anisotropic rotated Gaussian modes testing factor disentanglement. | — |
| [misgan_block_neural_imputer](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 3-pair MisGAN neural imputer on ambiguous single-sensor queries held pending passing recipe without ppost fallback. | Count again when one global recipe passes misgan_block_neural_imputer under its unchanged declaration. |
| [misgan_block_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | 2-pair incomplete-data MisGAN on 8D block problem held pending passing recipe for manifold recovery and ppost imputation. | Count again when one global recipe passes misgan_block_recovery under its unchanged declaration. |
| [misgan_mcar_p50_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Independent coordinatewise MCAR p=0.50 MisGAN on the 8-mode 8D block manifold with data/mask generator pair. | — |
| [misgan_mcar_p80_anchored](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | Extreme MCAR p=0.80 MisGAN on the 8-mode 8D block manifold with a 10% fully observed anchor subset. | — |
| [pacgan8_grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-8 packed critic on the 10x10 100-mode 2D Gaussian grid testing 100/100 mode coverage. | — |
| [pacgan_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | PacGAN-4 packed critic on an unequal-mass 8-mode ring with 20:1 weight ratio between heavy and light modes. | — |
| [ring16_bigan_continuous](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) | BLOCKED | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | Continuous 2D multimodal joint BiGAN (G, E, D) on 16-Gaussian ring held pending passing generation and reconstruction recipe. | Count again when one global recipe passes ring16_bigan_continuous under its unchanged declaration. |

<a name="cohort-cuda-23a94f16ca88-adaptation"></a>

## adaptation

**adaptation — revision 4**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-adaptation-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality"></a>

## bispf_fastpath_quality

**bispf_fastpath_quality — revision 1**. [View declaration](../../../configs/forge/views/bispf_fastpath_quality.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_w64_optimizer_diagnostic](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_w64_optimizer_diagnostic) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 5**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **4 / 12 / 6 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-clockfree_continuous-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1"></a>

## ddgan-conditional-repair-v1

**ddgan-conditional-repair-v1 — revision 2**. [View declaration](../../../configs/forge/views/ddgan-conditional-repair-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-ddgan_repair_gaussian_chain_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [ddgan_repair_bimodal_chain_wall_v2](empf.md#cohort-cuda-23a94f16ca88-experiment-ddgan_repair_bimodal_chain_wall_v2) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 16**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **11 / 18 / 1 / 0**. Held counts (measured and shown, not counted): **2 / 7 / 1 / 15**. Its highest tier, Tier 4, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | required | PASS | matches recorded run |
| [zero_tanh_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | required | PASS | matches recorded run |
| [model_ae_gan_multimodal](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | required | PASS | matches recorded run |
| [model_ae_reconstruction](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | required | PASS | matches recorded run |
| [model_vae_categorical](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | required | PASS | matches recorded run |
| [model_vae_hard](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [precision_gaussian_fp32](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_fp32) | diagnostic | FAIL | matches recorded run |
| [precision_gaussian_mxfp8](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | FAIL | matches recorded run |
| [precision_gaussian_nvfp4](empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) | [stretch: frontier stretch benchmark beyond the required suite](../precision-stretch/README.md#precision-stretch) | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | required | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | required | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [model_ae_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [model_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | required | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-discriminator_stability-tier-4"></a>

### Tier 4

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_mlp16](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p50_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [misgan_mcar_p80_anchored](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [pacgan8_grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [alice_spiral_bigan](empf.md#cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [infogan_anisotropic_modes](empf.md#cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cgan_overlapping_classes](empf.md#cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cramer_energy_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [cyclegan_unpaired_constellation](empf.md#cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |
| [grid100_res16_scaled_tanh](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh) | [stretch: frontier stretch benchmark beyond the required suite](../test-dispositions-20261015/README.md#unresolved) | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 3**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-formulation_comparison-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 6**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 1 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |
| [img_intensity2_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-host_profile_transfer-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage"></a>

## lazy_cadence_coverage

**lazy_cadence_coverage — revision 1**. [View declaration](../../../configs/forge/views/lazy_cadence_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **0 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [clockfree_audit_measurement_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1"></a>

## misgan-bispf-research-v1

**misgan-bispf-research-v1 — revision 2**. [View declaration](../../../configs/forge/views/misgan-bispf-research-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Scoped research diagnostic; no ordinary qualification or default adoption.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_misgan_four_mode](empf.md#cohort-cuda-23a94f16ca88-experiment-model_misgan_four_mode) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1"></a>

## prior-ablation-full-suite-v1

**prior-ablation-full-suite-v1 — revision 1**. [View declaration](../../../configs/forge/views/prior-ablation-full-suite-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [zero_tanh_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke) | diagnostic | PASS | matches recorded run |
| [model_ae_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) | diagnostic | FAIL | matches recorded run |
| [model_ae_gan_multimodal](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) | diagnostic | PASS | matches recorded run |
| [model_ae_reconstruction](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction) | diagnostic | PASS | matches recorded run |
| [model_ddgan_gaussian_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) | diagnostic | FAIL | matches recorded run |
| [model_vae_categorical](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |
| [model_vae_hard](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |
| [clockfree_audit](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |
| [model_ae_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [model_ddgan_bimodal_chain](empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain) | diagnostic | UNKNOWN | matches recorded run |
| [ring_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_hold) | diagnostic | UNKNOWN | matches recorded run |
| [ring_extension](empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) | diagnostic | UNKNOWN | matches recorded run |
| [zero_tanh_rail_escape](empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) | diagnostic | BLOCKED | matches recorded run |
| [grid100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) | diagnostic | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) | diagnostic | UNKNOWN | recorded definition unavailable |
| [misgan_block_recovery](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) | diagnostic | BLOCKED | matches recorded run |
| [misgan_block_neural_imputer](empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) | diagnostic | BLOCKED | matches recorded run |
| [ring16_bigan_continuous](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 4**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **3 / 12 / 0 / 0**. Held counts (measured and shown, not counted): **0 / 7 / 0 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | required | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | required | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | required | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | required | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-quality_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022"></a>

## r1r2-repair-word-retention-20261022

**r1r2-repair-word-retention-20261022 — revision 1**. [View declaration](../../../configs/forge/views/r1r2-repair-word-retention-20261022.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) | diagnostic | FAIL | matches recorded run |
| [trajectory](empf.md#cohort-cuda-23a94f16ca88-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](empf.md#cohort-cuda-23a94f16ca88-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](empf.md#cohort-cuda-23a94f16ca88-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](empf.md#cohort-cuda-23a94f16ca88-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](empf.md#cohort-cuda-23a94f16ca88-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) | diagnostic | FAIL | matches recorded run |
| [vector_two_broad](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](empf.md#cohort-cuda-23a94f16ca88-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3 / 4): **7 / 0 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage"></a>

## vaegan_coverage

**vaegan_coverage — revision 1**. [View declaration](../../../configs/forge/views/vaegan_coverage.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Two toy gates do not establish screen calibration or whole-suite adoption.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [model_vae_hard](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_hard) | diagnostic | PASS | matches recorded run |
| [model_vae_categorical](empf.md#cohort-cuda-23a94f16ca88-experiment-model_vae_categorical) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-vaegan_coverage-tier-4"></a>

### Tier 4

No experiments assigned.

<a name="cohort-cuda-23a94f16ca88-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-23a94f16ca88-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0211506 | <= 0.35 | PASS |
| recon_mse | 0.00594455 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8e7ca80648614cd39c5adf2c84b48375.json)

Recorded conditions: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-alice_spiral_bigan"></a>

### alice_spiral_bigan

**alice_spiral_bigan: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/alice_spiral_bigan.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. alice_spiral_bigan: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.06); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| arc_bins | >= 15 |
| hq | >= 0.85 |
| mass_tv | <= 0.16 |
| recon_rmse | <= 0.14 |
| latent_recon_rmse | <= 0.25 |
| arc_consistency | >= 0.9 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.06); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cgan_overlapping_classes"></a>

### cgan_overlapping_classes

**cgan_overlapping_classes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cgan_overlapping_classes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cgan_overlapping_classes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| classes_covered | == 4 |
| class_mass_tv | <= 0.08 |
| mean_center_err | <= 0.12 |
| min_cov_eig_ratio | >= 0.6 |
| max_cov_eig_ratio | <= 1.5 |
| overlap_density_ratio | >= 0.55 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **PASS** (new_attempt; qualification unchanged). declared state/horizon/cadence/restart comparisons agree; source audit bound [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [selected-family-coverage-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/3df194389aa1458d8a3affe3689663db.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.960282 | >= 0.75 | PASS |
| leak_ratio | 0.0132236 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0542447 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0284802 | <= 0.2 | PASS |
| same_dir | 0.0296834 | <= 0.25 | PASS |
| u_kept | 1.02451 | >= 0.85 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e97288aef290489fa141c5611a06795f.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-cramer_energy_overlap"></a>

### cramer_energy_overlap

**cramer_energy_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cramer_energy_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cramer_energy_overlap: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 9 |
| hq | >= 0.88 |
| mass_tv | <= 0.1 |
| energy_distance | <= 0.06 |
| min_cov_eig_ratio | >= 0.35 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-cyclegan_unpaired_constellation"></a>

### cyclegan_unpaired_constellation

**cyclegan_unpaired_constellation: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cyclegan_unpaired_constellation: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.08); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes_y | >= 7 |
| modes_x | >= 7 |
| hq_y | >= 0.85 |
| hq_x | >= 0.85 |
| mass_tv_y | <= 0.15 |
| cycle_rmse | <= 0.12 |
| bijection_ber | <= 0.15 |


Declared budget: 3500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.08); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](empf.md#cohort-cuda-23a94f16ca88-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-ddgan_repair_bimodal_chain_wall_v2"></a>

### ddgan_repair_bimodal_chain_wall_v2

**ddgan_repair_bimodal_chain_wall_v2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.95 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 0.5 |
| final_local_covariance_min | >= 0.5 |
| final_local_covariance_max | <= 1.5 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 1200 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../ddgan-conditional-repair/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-ddgan_repair_gaussian_chain_v1"></a>

### ddgan_repair_gaussian_chain_v1

**ddgan_repair_gaussian_chain_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [ddgan-conditional-repair-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-ddgan-conditional-repair-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../ddgan-conditional-repair/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. every joint generation/inverse check throughout the fixed continuation window

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 4834 |
| mass_apple | 0.194336 |
| mass_berry | 0.193359 |
| mass_grape | 0.205078 |
| mass_lemon | 0.191406 |
| mass_melon | 0.21582 |
| mass_tv | 0.0208984 |
| minimum_reconstruction_token_probability | 0.999757 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 6.09842e-05 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 4834 |

[Compact metrics and receipt provenance](../technique-receipts/1c832159b0974a88902a317a4226361f.json)

Recorded conditions: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

Declared budget: 4000 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Dependencies: five_word_joint_smoke (checkpoint).

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full joint pass plus independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [r1r2-repair-word-retention-20261022 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-r1r2-repair-word-retention-20261022-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.208008 |
| mass_berry | 0.210938 |
| mass_grape | 0.195312 |
| mass_lemon | 0.198242 |
| mass_melon | 0.1875 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.972237 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 0.00111316 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/9998a00908e447449e6c439fe01a9243.json)

Recorded conditions: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.224376 |
| finite_fraction | 1 |
| mean | 2.29427 |
| mean_error_sigma | 0.588539 |
| sample_count | 4096 |
| std | 0.534101 |
| std_ratio | 1.0682 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/dee71e364300412385db10238d2a129c.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0798735 |
| finite_fraction | 1 |
| mean | 3.14343 |
| mean_error_sigma | 0.286865 |
| sample_count | 4096 |
| std | 0.586575 |
| std_ratio | 1.17315 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/2308a9791cd24fc587a2efca7dd4d544.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 6000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: gaussian1d_smoke (checkpoint).

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.590192 |
| accuracy_pass | False |
| accuracy_score | 5.85024 |
| all_finite | True |
| center_max_sigma | 1.72493 |
| center_rms_sigma | 1.0618 |
| cov_frob_rms | 0.713217 |
| cov_trace_bias | 0.590192 |
| frozen_pass | False |
| mass_tv | 0.15946 |
| n | 100000 |
| passed | False |
| precision | 0.35909 |
| precision_gap | 0.629801 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.381295 |
| valid_n | 100000 |
| within_radius_n | 35909 |

[Compact metrics and receipt provenance](../technique-receipts/96dc940fdacb4dd4aa03a657d4ecb89c.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent grid100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_mlp16"></a>

### grid100_mlp16

**grid100_mlp16: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_mlp16.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_mlp16: native model variants require an explicit profile

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: native model variants require an explicit profile [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: particle_cloud prior (sigma 0); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_res16_scaled_tanh"></a>

### grid100_res16_scaled_tanh

**grid100_res16_scaled_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_scaled_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_scaled_tanh: native model variants require an explicit profile

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: native model variants require an explicit profile [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh"></a>

### grid100_res16_tanh

**grid100_res16_tanh: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100_res16_tanh.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. native model variants require an explicit profile; grid100_res16_tanh: native model variants require an explicit profile

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: native model variants require an explicit profile [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.96875 | >= 0.9 | PASS |
| modes | 2 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/19aeca3133d04173a29e8f5b0cdb8415.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_bars4_w64_optimizer_diagnostic"></a>

### img_bars4_w64_optimizer_diagnostic

**img_bars4_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.90625 | >= 0.9 | PASS |
| modes | 3 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0ab3d3f870eb41c09b89c462457a0957.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.53125 | >= 0.9 | FAIL |
| modes | 1 | >= 2 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9527428119d84a1594fba05c0660bec9.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1c092dd4dd824d5a95e3ee76342f8216.json)

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-img_stripes2_w64_optimizer_diagnostic"></a>

### img_stripes2_w64_optimizer_diagnostic

**img_stripes2_w64_optimizer_diagnostic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [bispf_fastpath_quality / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bispf_fastpath_quality-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-infogan_anisotropic_modes"></a>

### infogan_anisotropic_modes

**infogan_anisotropic_modes: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/infogan_anisotropic_modes.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. infogan_anisotropic_modes: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| discrete_nmi | >= 0.85 |
| q_acc | >= 0.9 |
| continuous_r2 | >= 0.65 |

At least 5 consecutive passing terminal observations.

Declared budget: 3500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 0.999985 | >= 0.85 | PASS |
| concept_cos_plus | 0.999911 | >= 0.85 | PASS |
| concept_mag_minus | 1.00891 | <= 1.25 | PASS |
| concept_mag_plus | 0.996639 | <= 1.25 | PASS |
| identity_at_0 | 0.991335 | >= 0.85 | PASS |
| identity_at_mid | 0.968472 | >= 0.85 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b41fc4db7e3743999a187972995a2929.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer"></a>

### misgan_block_neural_imputer

**misgan_block_neural_imputer: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_neural_imputer.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_neural_imputer: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| observed_max_error | <= 1e-06 |
| acc | >= 0.95 |
| acc_lo | >= 0.65 |
| missing_std_ratio | >= 0.7 |
| missing_std_ratio | <= 1.3 |


Declared budget: 7000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery"></a>

### misgan_block_recovery

**misgan_block_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_block_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_block_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 100 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| acc | >= 0.96 |
| acc_lo | >= 0.7 |


Declared budget: 7000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p50_recovery"></a>

### misgan_mcar_p50_recovery

**misgan_mcar_p50_recovery: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p50_recovery: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 90 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| off_plane | <= 0.05 |
| mask_tv | <= 0.05 |
| acc | >= 0.9 |
| impute_rmse | <= 0.12 |


Declared budget: 4000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-misgan_mcar_p80_anchored"></a>

### misgan_mcar_p80_anchored

**misgan_mcar_p80_anchored: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. misgan_mcar_p80_anchored: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 80 |
| hq | >= 0.8 |
| mass_tv | <= 0.15 |
| off_plane | <= 0.06 |
| imp_acc | >= 0.65 |
| imp_off_plane | <= 0.06 |
| impute_rmse | <= 0.18 |


Declared budget: 4000 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-mode_hold"></a>

### mode_hold

**mode_hold: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.841064 | >= 0.9 | FAIL |
| modes | 5 | >= 8 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f0d60a71ea704dff91a9066dc6fc8aee.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_bimodal_chain"></a>

### model_ae_ddgan_bimodal_chain

**model_ae_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **FAIL** (new_attempt; qualification unchanged). recomputed complete live curve and terminal suffix [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Dependencies: model_ae_ddgan_gaussian_chain (gate).

[Explanation and existing training artifacts](../api-model-toys/ae-ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain"></a>

### model_ae_ddgan_gaussian_chain

**model_ae_ddgan_gaussian_chain: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| final_covariance_eigen_max | 0.899075 | <= 1.4 | PASS |
| final_covariance_eigen_min | 0.604778 | >= 0.65 | FAIL |
| final_normalized_mean | 0.304754 | <= 0.45 | PASS |
| final_projection_cdf | 0.16974 | <= 0.2 | PASS |
| paired_nmse | 0.0511699 | <= 0.1 | PASS |
| paired_reverse_steps | 4 | == 4 | PASS |
| paired_shuffled_gain | 38.5219 | >= 3 | PASS |
| paired_worst_axis_nrmse | 0.227597 | <= 0.25 | PASS |
| panel_count | 8 | == 8 | PASS |
| samples_per_panel | 2048 | >= 2048 | PASS |
| t1_q0_covariance_eigen_max | 1.42808 | <= 1.8 | PASS |
| t1_q0_covariance_eigen_min | 0.39568 | >= 0.4 | FAIL |
| t1_q0_normalized_mean | 0.338416 | <= 0.65 | PASS |
| t1_q0_projection_cdf | 0.260577 | <= 0.26 | FAIL |
| t1_q1_covariance_eigen_max | 0.601292 | <= 1.8 | PASS |
| t1_q1_covariance_eigen_min | 0.558602 | >= 0.4 | PASS |
| t1_q1_normalized_mean | 0.330252 | <= 0.65 | PASS |
| t1_q1_projection_cdf | 0.216415 | <= 0.26 | PASS |
| t2_q0_covariance_eigen_max | 0.943565 | <= 1.4 | PASS |
| t2_q0_covariance_eigen_min | 0.744104 | >= 0.6 | PASS |
| t2_q0_normalized_mean | 0.204671 | <= 0.35 | PASS |
| t2_q0_projection_cdf | 0.0934495 | <= 0.2 | PASS |
| t2_q1_covariance_eigen_max | 0.861235 | <= 1.4 | PASS |
| t2_q1_covariance_eigen_min | 0.745064 | >= 0.6 | PASS |
| t2_q1_normalized_mean | 0.0942796 | <= 0.35 | PASS |
| t2_q1_projection_cdf | 0.0638522 | <= 0.2 | PASS |
| t3_q0_covariance_eigen_max | 0.97506 | <= 1.4 | PASS |
| t3_q0_covariance_eigen_min | 0.906747 | >= 0.65 | PASS |
| t3_q0_normalized_mean | 0.0300168 | <= 0.25 | PASS |
| t3_q0_projection_cdf | 0.0269585 | <= 0.12 | PASS |
| t3_q1_covariance_eigen_max | 0.985023 | <= 1.4 | PASS |
| t3_q1_covariance_eigen_min | 0.928005 | >= 0.65 | PASS |
| t3_q1_normalized_mean | 0.0406756 | <= 0.25 | PASS |
| t3_q1_projection_cdf | 0.0320822 | <= 0.12 | PASS |
| t4_q0_covariance_eigen_max | 1.03638 | <= 1.4 | PASS |
| t4_q0_covariance_eigen_min | 0.990839 | >= 0.65 | PASS |
| t4_q0_normalized_mean | 0.0323973 | <= 0.25 | PASS |
| t4_q0_projection_cdf | 0.0223998 | <= 0.12 | PASS |
| t4_q1_covariance_eigen_max | 1.01591 | <= 1.4 | PASS |
| t4_q1_covariance_eigen_min | 0.953182 | >= 0.65 | PASS |
| t4_q1_normalized_mean | 0.0141046 | <= 0.25 | PASS |
| t4_q1_projection_cdf | 0.0240515 | <= 0.12 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/405280833a804c13a07f2fa7b6f8827d.json)

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.2 |
| final_normalized_mean | <= 0.45 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.65 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.8 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.65 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.8 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 180 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae-ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal"></a>

### model_ae_gan_multimodal

**model_ae_gan_multimodal: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_gan_multimodal.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_center_rms_sigma | 0.870852 | <= 1 | PASS |
| component_max_eigen_ratio | 1.55497 | <= 4 | PASS |
| component_min_eigen_ratio | 0.400577 | >= 0.1 | PASS |
| mass_tv | 0.0480957 | <= 0.15 | PASS |
| modes | 4 | >= 4 | PASS |
| precision | 0.987061 | >= 0.9 | PASS |
| reconstruction_mode_agreement | 0.999512 | >= 0.98 | PASS |
| reconstruction_mse | 0.00736696 | <= 0.025 | PASS |

Recorded terminal passing observations: **11**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7cc6dff1c3a2499f81aa7e47f809b98d.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |
| modes | >= 4 |
| precision | >= 0.9 |
| mass_tv | <= 0.15 |
| component_min_eigen_ratio | >= 0.1 |
| component_max_eigen_ratio | <= 4 |
| component_center_rms_sigma | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ae_reconstruction"></a>

### model_ae_reconstruction

**model_ae_reconstruction: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_ae_reconstruction.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| reconstruction_mode_agreement | 1 | >= 0.98 | PASS |
| reconstruction_mse | 0.00653656 | <= 0.025 | PASS |

Recorded terminal passing observations: **24**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/6e8d46925c7f4b17a2c1de0eaa161208.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior; exact optimizer updates = True; mechanism exercised = False.

Declared budget: 1200 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ddgan_bimodal_chain"></a>

### model_ddgan_bimodal_chain

**model_ddgan_bimodal_chain: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **FAIL** (new_attempt; qualification unchanged). recomputed complete live curve and terminal suffix [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 5000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Dependencies: model_ddgan_gaussian_chain (gate).

[Explanation and existing training artifacts](../api-model-toys/ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain"></a>

### model_ddgan_gaussian_chain

**model_ddgan_gaussian_chain: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| final_covariance_eigen_max | 1.72463 | <= 1.4 | FAIL |
| final_covariance_eigen_min | 0.69845 | >= 0.65 | PASS |
| final_normalized_mean | 0.816607 | <= 0.25 | FAIL |
| final_projection_cdf | 0.343358 | <= 0.12 | FAIL |
| panel_count | 8 | == 8 | PASS |
| samples_per_panel | 2048 | >= 2048 | PASS |
| t1_q0_covariance_eigen_max | 4.15011 | <= 1.4 | FAIL |
| t1_q0_covariance_eigen_min | 0.489234 | >= 0.4 | PASS |
| t1_q0_normalized_mean | 1.88524 | <= 0.45 | FAIL |
| t1_q0_projection_cdf | 0.640723 | <= 0.26 | FAIL |
| t1_q1_covariance_eigen_max | 3.06759 | <= 1.4 | FAIL |
| t1_q1_covariance_eigen_min | 0.664828 | >= 0.4 | PASS |
| t1_q1_normalized_mean | 0.719276 | <= 0.45 | FAIL |
| t1_q1_projection_cdf | 0.312755 | <= 0.26 | FAIL |
| t2_q0_covariance_eigen_max | 2.01495 | <= 1.4 | FAIL |
| t2_q0_covariance_eigen_min | 0.612245 | >= 0.6 | PASS |
| t2_q0_normalized_mean | 0.44528 | <= 0.35 | FAIL |
| t2_q0_projection_cdf | 0.195848 | <= 0.2 | PASS |
| t2_q1_covariance_eigen_max | 1.51685 | <= 1.4 | FAIL |
| t2_q1_covariance_eigen_min | 0.5482 | >= 0.6 | FAIL |
| t2_q1_normalized_mean | 0.206084 | <= 0.35 | PASS |
| t2_q1_projection_cdf | 0.13796 | <= 0.2 | PASS |
| t3_q0_covariance_eigen_max | 1.33589 | <= 1.4 | PASS |
| t3_q0_covariance_eigen_min | 0.937963 | >= 0.65 | PASS |
| t3_q0_normalized_mean | 0.26932 | <= 0.25 | FAIL |
| t3_q0_projection_cdf | 0.118313 | <= 0.12 | PASS |
| t3_q1_covariance_eigen_max | 1.25172 | <= 1.4 | PASS |
| t3_q1_covariance_eigen_min | 0.911792 | >= 0.65 | PASS |
| t3_q1_normalized_mean | 0.126989 | <= 0.25 | PASS |
| t3_q1_projection_cdf | 0.0610171 | <= 0.12 | PASS |
| t4_q0_covariance_eigen_max | 1.03953 | <= 1.4 | PASS |
| t4_q0_covariance_eigen_min | 1.02041 | >= 0.65 | PASS |
| t4_q0_normalized_mean | 0.0758937 | <= 0.25 | PASS |
| t4_q0_projection_cdf | 0.0389671 | <= 0.12 | PASS |
| t4_q1_covariance_eigen_max | 1.01904 | <= 1.4 | PASS |
| t4_q1_covariance_eigen_min | 0.977859 | >= 0.65 | PASS |
| t4_q1_normalized_mean | 0.0498438 | <= 0.25 | PASS |
| t4_q1_projection_cdf | 0.040066 | <= 0.12 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7c36cbfbabd74bbb92aeb1b432c5c5e6.json)

Recorded conditions: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.45 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.45 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 2500 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.25); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/ddgan.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_misgan_four_mode"></a>

### model_misgan_four_mode

**model_misgan_four_mode: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/model_misgan_four_mode.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [misgan-bispf-research-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-misgan-bispf-research-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| imputer_observed_error | <= 1e-06 |
| imputer_accuracy | >= 0.94 |
| ambiguous_hq | >= 0.9 |
| ambiguous_mass_tv | <= 0.12 |
| ambiguous_query_coverage | >= 0.9 |
| ambiguous_std_ratio | >= 0.7 |
| ambiguous_std_ratio | <= 1.3 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../misgan-bispf/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_vae_categorical"></a>

### model_vae_categorical

**model_vae_categorical: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_categorical.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 0.997314 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.0576172 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 2.66454e-15 | <= 0.0001 | PASS |
| kl_identity_error | 5.91395e-07 | <= 1e-05 | PASS |
| likelihood_hq | 0.990723 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.064209 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 0.688676 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0033 | <= 1.2 | PASS |
| reconstruction_mse | 0.00869026 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 265.06 | >= 5 | PASS |

Recorded terminal passing observations: **14**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/109a9e39417045faa20ccd42fb19ba8f.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/vae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-model_vae_hard"></a>

### model_vae_hard

**model_vae_hard: PASS**. [Current experiment declaration](../../../configs/forge/tasks/model_vae_hard.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [vaegan_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-vaegan_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| decoder_mean_hq | 1 | >= 0.85 | PASS |
| decoder_mean_mass_tv | 0.00366211 | <= 0.15 | PASS |
| decoder_mean_modes | 2 | >= 2 | PASS |
| elbo_identity_error | 4.44089e-16 | <= 0.0001 | PASS |
| kl_identity_error | 7.61862e-09 | <= 1e-05 | PASS |
| likelihood_hq | 0.99707 | >= 0.85 | PASS |
| likelihood_mass_tv | 0.00366211 | <= 0.15 | PASS |
| likelihood_modes | 2 | >= 2 | PASS |
| posterior_mutual_information | 2.47805 | >= 0.3 | PASS |
| posterior_noise_variance_ratio | 1.0009 | <= 1.2 | PASS |
| reconstruction_mse | 0.00482352 | <= 0.05 | PASS |
| shuffle_reconstruction_ratio | 457.459 | >= 5 | PASS |

Recorded terminal passing observations: **13**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/58335b838fe44eb0b709b33c50d64fc5.json)

Recorded conditions: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1200 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); task_model_kernel_and_posterior_without_serving_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../api-model-toys/vae.md)

<a name="cohort-cuda-23a94f16ca88-experiment-pacgan8_grid100"></a>

### pacgan8_grid100

**pacgan8_grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan8_grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan8_grid100: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-pacgan_unequal_mass"></a>

### pacgan_unequal_mass

**pacgan_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/pacgan_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. pacgan_unequal_mass: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4).

Recorded conditions: mog prior (sigma 0.05); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 5 |
| min_mode_mass | >= 0.01 |
| mass_tv | <= 0.1 |
| hq | >= 0.88 |
| min_cov_eig_ratio | >= 0.2 |


Declared budget: 3000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.05); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_fp32"></a>

### precision_gaussian_fp32

**precision_gaussian_fp32: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_fp32.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.121201 | <= 0.05 | FAIL |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.231523 | <= 0.2 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 0.895476 | <= 1.2 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0b953988d30048a2bb2e5477fca474ea.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8"></a>

### precision_gaussian_mxfp8

**precision_gaussian_mxfp8: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_mxfp8.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.134316 | <= 0.05 | FAIL |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.155518 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 0.850892 | <= 1.2 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/4fe8a9db5c1c48fb84b649d23479305a.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4"></a>

### precision_gaussian_nvfp4

**precision_gaussian_nvfp4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0435625 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.049942 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.03106 | <= 1.2 | PASS |

Recorded terminal passing observations: **1**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c8a3db850fb64fa4a9a5c5177dc82d37.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

Declared budget: 1000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000190407 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e0125ae3d45342c3a59ec7d4af1caffc.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.515821 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.422528 | >= 0.15 | PASS |
| hq | 0.941406 | >= 0.85 | PASS |
| mass_tv | 0.0651855 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **19**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d568a00cc41342db82c3c7a8a83c9809.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous"></a>

### ring16_bigan_continuous

**ring16_bigan_continuous: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_bigan_continuous.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring16_bigan_continuous: no registered public Forge adapter for this stretch host; declare and implement its complete recipe, sampling and checkpoint contracts before training

Run attempt: **ERROR** (new_attempt; qualification unchanged). ValueError: unsupported vector target declaration [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 4](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-4) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 16 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| recon_rmse | <= 0.12 |
| mode_consistency | >= 0.98 |


Declared budget: 4000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **PASS** (reused_execution; qualification unchanged). verified observations from the same uninterrupted producer; no new execution [Attempt evidence](../grouped-evidence/35a173b8ac2bbf15563fde4330eef7871341f0cef5e75fde39366f4609935f4b.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: ring_hold (checkpoint).

<a name="cohort-cuda-23a94f16ca88-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **PASS** (new_attempt; qualification unchanged). recomputed first-convergence hold [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-3) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.679716 |
| accuracy_pass | False |
| accuracy_score | 6.02059 |
| all_finite | True |
| center_max_sigma | 1.89072 |
| center_rms_sigma | 1.04508 |
| cov_frob_rms | 0.784041 |
| cov_trace_bias | 0.679716 |
| frozen_pass | False |
| mass_tv | 0.10743 |
| n | 100000 |
| passed | False |
| precision | 0.26929 |
| precision_gap | 0.719601 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.410772 |
| valid_n | 100000 |
| within_radius_n | 26929 |

[Compact metrics and receipt provenance](../technique-receipts/79b6cd39653d4b0db7bbf26c2a35186b.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent rotated100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: rotated100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.519504 |
| accuracy_pass | False |
| accuracy_score | 5.82254 |
| all_finite | True |
| center_max_sigma | 2.14775 |
| center_rms_sigma | 1.22916 |
| cov_frob_rms | 0.682851 |
| cov_trace_bias | 0.519504 |
| frozen_pass | False |
| mass_tv | 0.10101 |
| n | 100000 |
| passed | False |
| precision | 0.27582 |
| precision_gap | 0.713071 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.410633 |
| valid_n | 100000 |
| within_radius_n | 27582 |

[Compact metrics and receipt provenance](../technique-receipts/6579ba9962ea4e69a53d03746072ff2a.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent staggered100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: staggered100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **FAIL** (new_attempt; qualification unchanged). recomputed live recovery and matched frozen negative control [Attempt evidence](../attempt-coverage.json).

Used by: [adaptation / Tier 3](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-3) · [clockfree_continuous / Tier 3](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000201905 | <= 0.02 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1bd143997eee49a1bd8606b87f52c633.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.650808 | <= 1 | PASS |
| mean_abs | 0.958166 | >= 0.3 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/49b2d6ddfc2940e5a7c9f1f7d3b04ba3.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](empf.md#cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](empf.md#cohort-cuda-23a94f16ca88-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-23a94f16ca88-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.988448 | >= 0.85 | PASS |
| neu_hold | 0.993209 | >= 0.85 | PASS |
| off_caption | 7.55969e-05 | <= 0.05 | PASS |

Recorded terminal passing observations: **19**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1ffed96c5d61498a933d325af291a751.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-1) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.988805 | >= 0.85 | PASS |
| unused_hold | 0.992345 | >= 0.85 | PASS |

Recorded terminal passing observations: **11**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/95ad8ff9c9f74e2b8a02a2d1f4d4b5e2.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.261289 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.361832 | >= 0.15 | PASS |
| hq | 0.984131 | >= 0.85 | PASS |
| mass_tv | 0.121663 | <= 0.15 | PASS |
| sw1_normalized | 0.136694 | <= 0.18 | PASS |

Recorded terminal passing observations: **4**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/72ec19ee27474fff87705c3077b8b5ab.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.492478 | <= 0.45 | FAIL |
| mean_error | 0.0977112 | <= 0.15 | PASS |
| sw1_normalized | 0.12933 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d7af12c32d804b1caf70113398f0de01.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0972097 | <= 0.45 | PASS |
| mean_error | 0.112903 | <= 0.15 | PASS |
| sw1_normalized | 0.0777357 | <= 0.18 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/08ad677e7d254c56aeb0a30144410cad.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.396641 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.413355 | >= 0.15 | PASS |
| hq | 0.999756 | >= 0.85 | PASS |
| mass_tv | 0.029541 | <= 0.15 | PASS |
| sw1_normalized | 0.0902461 | <= 0.18 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f585a9f7410d450d833200605580cf5e.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.499749 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.0594819 | >= 0.15 | FAIL |
| hq | 0.981201 | >= 0.85 | PASS |
| mass_tv | 0.0932129 | <= 0.15 | PASS |
| min_mass_ratio | 0.68929 | >= 0.25 | PASS |
| sw1_normalized | 0.160964 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1b354055d52544de83b2d4430b337d5e.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](empf.md#cohort-cuda-23a94f16ca88-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](empf.md#cohort-cuda-23a94f16ca88-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2) · [lazy_cadence_coverage / Tier 1](empf.md#cohort-cuda-23a94f16ca88-lazy_cadence_coverage-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1) · [quality_coverage / Tier 2](empf.md#cohort-cuda-23a94f16ca88-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.442734 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.0789106 | >= 0.15 | FAIL |
| hq | 0.930664 | >= 0.85 | PASS |
| mass_tv | 0.223389 | <= 0.15 | FAIL |
| sw1_normalized | 0.208727 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/2c2cdfd6405c413cb1cf63d929367151.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](empf.md#cohort-cuda-23a94f16ca88-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](empf.md#cohort-cuda-23a94f16ca88-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape"></a>

### zero_tanh_rail_escape

**zero_tanh_rail_escape: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_rail_escape.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. zero_tanh_rail_escape: zero_tanh_rail_escape: unsupported tanh host zero_tanh_rail_escape; only zero_tanh_smoke is routed

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). zero_tanh_rail_escape: unsupported tanh host zero_tanh_rail_escape; only zero_tanh_smoke is routed [Attempt evidence](../attempt-coverage.json).

Used by: [discriminator_stability / Tier 3](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-3) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |

Recovery deadline in updates: 400.

Declared budget: 2000 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

Dependencies: zero_tanh_smoke (gate).

<a name="cohort-cuda-23a94f16ca88-experiment-zero_tanh_smoke"></a>

### zero_tanh_smoke

**zero_tanh_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/zero_tanh_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed the declared final zero-tanh bounds

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](empf.md#cohort-cuda-23a94f16ca88-discriminator_stability-tier-1) · [prior-ablation-full-suite-v1 / Tier 1](empf.md#cohort-cuda-23a94f16ca88-prior-ablation-full-suite-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mean_abs_error | 0.00278569 | <= 0.005 | PASS |
| p99_abs_error | 0.0106375 | <= 0.025 | PASS |
| rms_error | 0.00358642 | <= 0.01 | PASS |
| saturated_fraction | 0 | == 0 | PASS |

[Compact metrics and receipt provenance](../technique-receipts/739cfa5d3a2641309a374e88f3699640.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights unavailable; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |


Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights unbound; output noise clean.

## Historical and diagnostic evidence

Separate configurations, API variants and serving laws retain their own scopes and supply no cells above.

- [Other configurations and original evidence bindings](../technique-inventory.json)
- [Compiled experiment memory](../EXPERIMENT_MEMORY.md)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This page is generated alongside the leaderboard. Register new source evidence before refreshing; editing a page cannot change a verdict or earn qualification.

## References

- [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501). Online sufficient-statistics EMA prototype updates, hit-rate step-size scaling, and proportional-split dead-unit revival inspiring the EMPF prototype bank and outlier tether.
- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895). Entropic Optimal Transport Sinkhorn-Knopp coupling generalized here to non-uniform prototype row masses.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base first-order optimizer family supported alongside DualNorm.

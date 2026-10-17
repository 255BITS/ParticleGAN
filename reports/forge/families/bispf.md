<!-- Generated Forge family report -->

# BISPF

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [selectable-loss](../technique-inventory.md#tag-selectable-loss)

## Technique overview

Born-Infeld Sinkhorn Particle Flow (BISPF) is a constant-learning-rate ParticleGAN formulation with no EMA critic copy. It couples real and generated samples via scale-free entropic Optimal Transport (Sinkhorn-Knopp), regularizes critic gradients along real endpoints, fake endpoints, and Sinkhorn transport chords using a two-regime Elastic Born-Infeld potential with a zero-mean logit gauge, regularizes the particle prior with a degree-of-freedom-aware Stratified Blue-Noise Particle (SBNP) tether/repulsion law, and applies scale-free Harmonic Mode Capture thermal cooling once all modes are captured.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior with $N$ locations in $\mathbb R^{d_z}$, $x$ a real batch, $z\sim P$ a latent batch and $y=G(z)$ the generated batch. $\Pi$ is the doubly-stochastic Sinkhorn coupling computed from median-normalized squared Euclidean distances $C_{ij}=\lVert x_i-y_j\rVert_2^2/\bar d^2$. $x_{\mathrm{chord}}=(1-u)y+u x_{\pi}$ is a stratified Sinkhorn transport chord. $g_r=\nabla_x D(x)$, $g_f=\nabla_y D(y)$, and $g_c=\nabla_{x_{\mathrm{chord}}} D(x_{\mathrm{chord}})$ are input gradients; $d$ counts critic-input coordinates, $\kappa$ is the cap threshold, $\lambda_{\mathrm{floor}}$ the harmonic floor weight, $\lambda_{\mathrm{cap}}$ the elastic cap weight, and $\gamma_{\mathrm{gauge}}$ the zero-mean logit gauge weight.

**Elastic Born-Infeld Transport-Chord Regularizer and Zero-Mean Gauge**

$$
R_{\mathrm{BISPF}}=\frac{\lambda_{\mathrm{floor}}}{6d}\mathbb E\!\left[\lVert g_r\rVert_2^2+\lVert g_f\rVert_2^2+\lVert g_c\rVert_2^2\right]+\frac{\lambda_{\mathrm{cap}}}{2}\mathbb E\!\left[\phi_\beta(\lVert g_r\rVert_2-\kappa)^2+(1-\alpha_c)\phi_\beta(\lVert g_f\rVert_2-\kappa)^2+\alpha_c\phi_\beta(\lVert g_c\rVert_2-\kappa)^2\right]+\frac{\gamma_{\mathrm{gauge}}}{2}\bar D^2
$$

Here $\phi_\beta(u)=\beta^{-1}\log(1+e^{\beta u})$ is the smooth Born-Infeld elastic wall, $g_c=\nabla_x D((1-u)y+u x_\pi)$ is the critic gradient along Sinkhorn transport chords, and $\bar D=\frac{1}{2}(\mathbb E[D(x)]+\mathbb E[D(y)])$ pins the translational logit gauge without an EMA critic network copy.

**Stratified Blue-Noise Particle (SBNP) Prior Regularizer**

$$
R_{\mathrm{SBNP}}(P)=\lambda_P\left(\mathcal T_{\mathrm{moment}}(P)+w_{\mathrm{repel}}\,\mathbb I(N>2k_{\mathrm{nn}}+d_z)\,\frac{1}{N k_{\mathrm{nn}}}\sum_{i=1}^N\sum_{m=1}^{k_{\mathrm{nn}}}\log\!\left(1+\frac{\bar r_{\mathrm{knn}}^2}{\lVert z_i-z_{\mathcal N_m(i)}\rVert_2^2+\epsilon}\right)\right)
$$

Preserves the initial latent mean and (when $N>2k_{\mathrm{nn}}+d_z$) initial covariance while preventing local particle clumping via scale-free $k$-NN log-repulsion.

## Simplified pseudocode

```text
Choose an adversarial loss, optimizer and constant learning rates.
For each task-owned training update:
  Sample real x and latent z from the task prior P; compute y=G(z).
  Compute scale-free Sinkhorn coupling Pi between x and stop_gradient(y).
  Sample stratified Sinkhorn transport chords x_chord = (1 - u) stop_gradient(y) + u x_pi.
  Evaluate EBI potential on input-gradient norms at x, stop_gradient(y) and x_chord plus zero-mean logit gauge.
  Apply Harmonic Mode Capture thermal cooling when bidirectional 95th-percentile nearest-neighbor transport distance is inside the capture basin.
  Backpropagate critic adversarial loss + BISPF penalty and step D.
  With D fixed, recompute differentiable fake samples y=G(z) and evaluate generator adversarial loss + SBNP prior regularizer.
  Step G, trainable prior P and encoder E when present using constant learning rates.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Selectable non-saturating logistic (selected DualNorm configuration), paired relativistic logistic, Sinkhorn relativistic logistic (SRpGAN), hinge, Wasserstein or least squares. Joint encoder/generator hosts train both critic streams. |
| Optimizer | Supports DualNorm (selected configuration: zero momentum, smoothing 0.001, per-offset convolution) as well as native Adam and formulation-owned BISPF optimizers without an EMA critic network copy. |
| Learning rates and annealing | Strictly constant: selected global LR $0.012$, critic multiplier $1.5$ and prior multiplier $2.5$, with network and prior LR floors fixed at $1.0$ (zero LR annealing). |
| Parameter-gradient clipping | None. Critic spike guard is disabled; gradient bounding acts smoothly through the Elastic Born-Infeld critic potential. |
| Critic penalties and anchors | Two-regime Elastic Born-Infeld Transport-Chord Regularizer (EBI-TCR) evaluated on real, fake and Sinkhorn transport-chord inputs ($\kappa=1$, sharpness $\beta=30$, harmonic floor $\lambda_{\mathrm{floor}}=0.0005$, chord blend $0.02$) plus a zero-mean logit gauge ($\gamma_{\mathrm{gauge}}=0.0005$). No EMA critic anchor. |
| Damping and update guards | Scale-free Harmonic Mode Capture thermal cooling damps post-convergence Poisson mini-batch fluctuations below DualNorm's smoothing floor when the bidirectional 95th-percentile nearest-neighbor distance enters the harmonic capture basin. |
| Training and sampling noise | No scheduled additive critic-input or generator-output training noise in the selected benchmark configuration ($0.0$). Task-declared MoG intrinsic sigma is preserved. |
| Parameter averaging and serving | Generator EMA decay=$0$ and serve_average=$0$; all evaluations score live unaveraged weights. |

## Configuration differences

- BISPF combines the Elastic Born-Infeld Transport-Chord Regularizer (EBI-TCR), zero-mean logit gauge, Stratified Blue-Noise Particle (SBNP) prior regularizer, and Harmonic Mode Capture thermal cooling under constant learning rates.
- The SBNP prior regularizer automatically adapts to prior table capacity: when $N \le 2 k_{\mathrm{nn}} + d_z$ (such as the 5-row discrete word codebook), covariance tethering and $k$-NN repulsion are disabled so discrete codebooks are unperturbed; for continuous particle clouds ($N > 2 k_{\mathrm{nn}} + d_z$), both the initial-moment tether and scale-free $k$-NN log-repulsion are active.
- One global trainer configuration supplies every displayed result across Tier 1, Tier 2 and Tier 3; task-owned architectures, priors, initializations, budgets and gates remain fixed.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/bispf.py](../../../particlegan/bispf.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/training.py](../../../particlegan/training.py)
- [configs/forge/ideas/bispf-v1.json](../../../configs/forge/ideas/bispf-v1.json)
- [configs/forge/ideas/bispf-v2.json](../../../configs/forge/ideas/bispf-v2.json)
- [configs/forge/studies/bispf-v1-study.json](../../../configs/forge/studies/bispf-v1-study.json)
- [configs/forge/studies/bispf-v2-study.json](../../../configs/forge/studies/bispf-v2-study.json)
- [reports/forge/bispf/README.md](../bispf/README.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

## Performance

DCGAN on deterministic mock RGB data measures finite-update speed and memory. These separate performance-source receipts confer no image-quality, scientific qualification or default-promotion credit. Standard BatchNorm is retained in generator and critic; critic input gradients therefore include batch coupling. The shared critic uses a flattened 4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.

Synchronized wall time for one public trainer update, including its critic/generator/prior work; warmup, construction, mock-data generation, evaluation, GIFs and checkpoint writes are excluded.

Every cell is one run; timings are never pooled across devices or contention states. Compare only rows with the same task and matched key (source, runtime/dtype, physical device, initialization and batch streams). Unknown or contended observations are unpaired. Resolution progression changes the task, while each candidate keeps one global recipe. Images/s is B divided by full update time (one D and one G/prior update with separate real batches). Three warmup and ten measured updates give short-window median/p95 measurements; long-horizon policy costs and component phase times remain unknown.

[Compact performance evidence index](../scaling-performance.json) · [Protocol and bounded reproduction](../../../docs/forge-scaling-performance.md)

This registration covers 8 complete recipes. Historical alternatives outside its registered family roster add no missing benchmark cells. A changed selected recipe within a registered family remains unknown until measured.

### NVIDIA RTX A6000 · GPU-ed080e41-3193-3755-6756-f3d46c433331

Performance source `018b30e837ce` (commit `58ade47791367abd46c68ede8d5e7d9fea3dbafb`); runtime `ef504138f591`; torch.float32; PyTorch 2.14.0+cu130; 48538.75 MiB device memory.

Task prior: particle_cloud, sigma 0.0, 256 initial locations. The common finite-table performance exception retains each recipe's prior update policy and regularizer. Exact prior and resource specifications remain part of every task key.

| Family / recipe | Task (RGB size, batch, width) | Execution | Median / p95 update ms | Images/s | GPU peak allocated / reserved MiB | Contention | Matched key / evidence |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| BISPF · `bispf-v2` | dcgan_performance_r32: 32², B16, W64 | COMPLETE | 282.29 / 290.64 | 56.50 | 90.40 / 124.00 | unknown | `66ecde1f427c` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_r32-98e39beaee3a0b77.json) |
| BISPF · `bispf-v2` | dcgan_performance_r64: 64², B16, W64 | COMPLETE | 609.52 / 670.34 | 25.81 | 221.78 / 316.00 | unknown | `450423595052` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_r64-666c107d181c8e78.json) |
| BISPF · `bispf-v2` | dcgan_performance_w128: 64², B16, W128 | COMPLETE | 1,526.29 / 1,977.79 | 10.10 | 674.18 / 978.00 | unknown | `60c16d28eb11` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_w128-916fe46e41bccbc8.json) |
| BISPF · `bispf-v2` | dcgan_performance_b64: 64², B64, W64 | COMPLETE | 680.49 / 690.94 | 94.11 | 558.22 / 628.00 | unknown | `eedd0a26764a` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_b64-8980e3805d46c8aa.json) |
| BISPF · `bispf-v2` | dcgan_performance_r128: 128², B16, W64 | COMPLETE | 2,080.19 / 2,321.55 | 7.55 | 632.24 / 842.00 | unknown | `603af9e788d9` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_r128-c1e1d9fdb1e6e01b.json) |
| BISPF · `bispf-v2` | dcgan_performance_r256: 256², B16, W64 | COMPLETE | 3,143.55 / 3,373.81 | 5.02 | 2,119.28 / 2,310.00 | unknown | `84f752c9d664` · [receipt](../scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_r256-ad812b07ff959bbc.json) |

<a name="cohort-cuda-3faddec347a0"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [bispf-v2](../../../configs/forge/ideas/bispf-v2.json).

Recorded qualification: **tier 1**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification. Their counts follow the declared view revisions (adaptation 3, clockfree_continuous 4, discriminator_stability 11, formulation_comparison 2, host_profile_transfer 5, quality_coverage 3); the recorded qualification keeps discriminator_stability revision 9 until new ordinary evidence is registered.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `df5ba19b79b329442f5141da24e4b22fa04569b8def42eb849b7af6d1693e141`. Candidate revision: `96f0f8a2aee0663f0fe92e49641deb107b81201ae0efea33d9a54e24aaba8967`. Runtime cohort: `32a72f2c08988465153a840f30f0926f2f6164347cde45c03d69dbb1c8a852a6`.

[Frozen numerical evidence](../technique-evidence/bf46d13ea30ed217735bcdf841704fbf16e018a445e2290821c6e34a58bd9f45.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Complete Tier 1, Tier 2, and Tier 3 ordinary CUDA measurement of Born-Infeld Sinkhorn Particle Flow v2 (BISPF-v2) on the BCAP DualNorm leader recipe.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests | Held (measured, not counted) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation) | [3/3](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1) | [9/9](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) | [0/0](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-3) | [12/12](bispf.md#cohort-cuda-3faddec347a0-adaptation) | [0/11 (1 not run)](bispf.md#cohort-cuda-3faddec347a0-held) |
| [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous) | [4/4](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) | [9/9](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) | [1/1](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | [14/14](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous) | [0/14 (4 not run)](bispf.md#cohort-cuda-3faddec347a0-held) |
| [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability) | [6/6](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) | [10/10](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) | [1/1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3) | [17/17](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability) | [0/11](bispf.md#cohort-cuda-3faddec347a0-held) |
| [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison) | [3/3](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1) | [9/9](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) | [1/1](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3) | [13/13](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison) | [0/10](bispf.md#cohort-cuda-3faddec347a0-held) |
| [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer) | [3/3](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1) | [9/9](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) | [1/1](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3) | [13/13](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer) | [0/10](bispf.md#cohort-cuda-3faddec347a0-held) |
| [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage) | [3/3](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1) | [9/9](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-3) | [12/12](bispf.md#cohort-cuda-3faddec347a0-quality_coverage) | [0/10](bispf.md#cohort-cuda-3faddec347a0-held) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage) | [0/7 (7 not run)](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1) | [0/0](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-3) | [0/7 (7 not run)](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-3faddec347a0-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](bispf.md#cohort-cuda-3faddec347a0-experiment-ring16_acquisition) | [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_extension](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3) | PASS | matches recorded run |

Held at this tier (measured and shown, not counted):

| Experiment | Held in | Recorded result | Test definition |
| --- | --- | --- | --- |
| [grid100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-3faddec347a0-experiment-target_shift_recovery) | [adaptation](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-3), [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

Removed from the count at this tier (still measured):

| Experiment | Removed in | Reason | Recorded result |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | [removed: duplicate of clockfree_audit_measurement_v1 (counted there)](../test-dispositions-20261015/README.md#clockfree_audit) | UNKNOWN |
| [ring_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_hold) | [clockfree_continuous](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS |

<a name="cohort-cuda-3faddec347a0-held"></a>

### Held tests (still measured)

These tests keep running and keep this recipe's recorded result, but they do not count toward its totals or qualification. Each counts again once its re-entry rule is met.

| Experiment | Held in | Recorded result | Reason | Summary | Re-entry rule |
| --- | --- | --- | --- | --- | --- |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#gaussian1d_stability) | BCAP and BISPF neither keep a learned bell curve (2/72 and 5/72 checks) nor re-learn it after the target moves (0/24); K3P and KA2 keep 71/72 but miss re-learning. Its one candidate that also passes mode_hold fails ring16_acquisition (draft PR #427). | Count again when one global recipe, in one same-source, same-runtime cohort, passes this whole test (72/72, re-acquisition, 24/24) and every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold, plus the counted ring endurance test. |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | 100 tiny blobs on a grid: BCAP puts 24% and BISPF 37% of samples on target; 97% is required (a perfect sampler scores 98.9%). The best 7,000-update result under this clean-sampling declaration is 54%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | Four bar pictures: BCAP and BISPF draw clean bars but only 2 of the 4. R1/R2 and GAN v3 pass it but fail Tier 1 tests and img_stripes2; under the current declarations no recipe passes both bars4 and stripes2. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | Four corner-blob pictures: BCAP covers 2 corners (24/32 clean), BISPF 3 (28/32); the test needs all 4 corners and 29/32 clean. Every recipe passing its current declaration fails or cannot run Tier 1 tests. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | Two brightness levels: BCAP draws both but 10 of 32 pictures are off in brightness; BISPF collapses to one level. The three recipes passing its current declaration fail Tier 1 tests that BCAP and BISPF pass. | Count again when one global recipe has an ordinary PASS on this unchanged declaration and, with the same configuration, still passes img_stripes2 and every other counted Tier 1 and Tier 2 test. |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | The same 100-blob test rotated by 25 degrees: BCAP 26% and BISPF 27% of samples on target against 97%. The best 7,000-update result under this clean-sampling declaration is 51%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. In practice it returns with grid100. |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | The same 100-blob test with alternate rows shifted: BCAP 30% and BISPF 34% of samples on target against 97%. The best 7,000-update result under this clean-sampling declaration is 56%, and none passes. | Count again when one global recipe has an ordinary PASS on this unchanged declaration (clean sampling, live weights, five final checks and the 100,000-sample holdout) and still passes every counted Tier 1 and Tier 2 test. A diagnostic PASS only triggers a review. In practice it returns with grid100. |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | Three stretched, tilted blobs: BCAP and BISPF draw the shapes but the wrong amounts (mass TV 0.196 and 0.191; limit 0.15). Of the 16 recipes passing its current declaration, 12 fail Tier 1 ring16_acquisition, 1 timed out on it and 3 fail mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | One broad cloud made of two overlapping blobs: BCAP and BISPF reach it early, then drift (centre error 0.39 and 0.28; limit 0.15). All 15 recipes passing its current declaration fail Tier 1 ring16_acquisition or mode_hold. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | The rare 2% blob loses its share or collapses (BCAP eigen ratio 0.009, BISPF 0.004; need 0.15). Perfect samples pass 200/200 runs. Its passers fail counted tests, except some round-5 research recipes that fail only held tests and never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_width, so the two will likely return together. |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2), [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2), [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2), [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2), [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2), [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) | FAIL | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | Equal shares, very different widths: BCAP and BISPF get the shares wrong (mass TV 0.29 and 0.25; limit 0.15). Perfect samples pass 200/200 runs. Cell-moment passers fail counted tests; round-5 balanced assignment fails only held tests but never ran most of the counted suite. | Count again when one global recipe (one recipe hash and source, seed 0, the public initializer and one runtime) passes this test under its unchanged declaration and also passes every counted Tier 1 and Tier 2 test, including ring16_acquisition and mode_hold. Round 5 shows it trades off against vector_unequal_mass, so the two will likely return together. |
| [grid100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_14k) | [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | [held: prerequisite grid100 is held](../test-dispositions-20261015/README.md#grid100_14k) | Continues a passing 7,000-update grid100 model to 14,000 updates. No recipe has a passing parent, so every start is refused; a BISPF rescue run from its failed parent also fails (precision 0.68). | Returns in the revision that restores grid100, once the global recipe has an ordinary grid100 PASS with its own saved state. That revision also restores clockfree_audit to the count as this test's gate. |
| [rotated100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_14k) | [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | [held: prerequisite rotated100 is held](../test-dispositions-20261015/README.md#rotated100_14k) | Continues a passing 7,000-update rotated100 model to 14,000 updates. No recipe has a passing parent; a BISPF rescue run from its failed parent also fails (precision 0.65). | Returns in the revision that restores rotated100, once the global recipe has an ordinary rotated100 PASS with its own saved state. That revision also restores clockfree_audit to the count as this test's gate. |
| [staggered100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_14k) | [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | [held: prerequisite staggered100 is held](../test-dispositions-20261015/README.md#staggered100_14k) | Continues a passing 7,000-update staggered100 model to 14,000 updates. No recipe has a passing parent; a BISPF rescue run from its failed parent also fails (precision 0.70). | Returns in the revision that restores staggered100, once the global recipe has an ordinary staggered100 PASS with its own saved state. That revision also restores clockfree_audit to the count as its gate. |
| [target_shift_recovery](bispf.md#cohort-cuda-3faddec347a0-experiment-target_shift_recovery) | [adaptation / Tier 3](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-3), [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) | UNKNOWN | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#target_shift_recovery) | Learn an 8-blob ring, hold it, then recover within 400 updates after it shifts. BISPF recovers but 220 updates late; BCAP flickers before and after the shift. No recipe has passed its current declaration. | Count again in adaptation and clockfree_continuous when one global recipe passes all four parts in one cohort (5/5, 120/120, 81/81 after the deadline, frozen copy 0/81) and every counted Tier 1 and Tier 2 test plus the counted ring endurance test. |

<a name="cohort-cuda-3faddec347a0-adaptation"></a>

## adaptation

**adaptation — revision 3**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 9 / 0**. Held counts (measured and shown, not counted): **0 / 10 / 1**. Its highest tier, Tier 3, has only held tests, so while they are held the view never shows PASS: a recipe that passes every counted tier gets HELD.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-3faddec347a0-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bispf.md#cohort-cuda-3faddec347a0-experiment-target_shift_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#target_shift_recovery) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | diagnostic | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | diagnostic | PASS | matches recorded run |
| [ring16_acquisition](bispf.md#cohort-cuda-3faddec347a0-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-3faddec347a0-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 4**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 9 / 1**. Held counts (measured and shown, not counted): **0 / 10 / 4**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-3faddec347a0-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit) | [removed: duplicate of clockfree_audit_measurement_v1 (counted there)](../test-dispositions-20261015/README.md#clockfree_audit) | UNKNOWN | recorded definition unavailable |
| [ring_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) | required | PASS | matches recorded run |
| [grid100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_14k) | [held: prerequisite grid100 is held](../test-dispositions-20261015/README.md#grid100_14k) | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_14k) | [held: prerequisite rotated100 is held](../test-dispositions-20261015/README.md#rotated100_14k) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_14k) | [held: prerequisite staggered100 is held](../test-dispositions-20261015/README.md#staggered100_14k) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-3faddec347a0-experiment-target_shift_recovery) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#target_shift_recovery) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-dcgan_performance"></a>

## dcgan_performance

**dcgan_performance — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-3faddec347a0-dcgan_performance-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_r32](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_r32) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r64](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_r64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r128](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_r128) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_r256](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_r256) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_b64](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_b64) | diagnostic | UNKNOWN | recorded definition unavailable |
| [dcgan_performance_w128](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_w128) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-dcgan_performance-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-dcgan_performance-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-dcgan_performance_cpu_smoke"></a>

## dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke — revision 1**. [View declaration](../../../configs/forge/views/dcgan_performance_cpu_smoke.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-3faddec347a0-dcgan_performance_cpu_smoke-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-dcgan_performance_cpu_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-dcgan_performance_cpu_smoke-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-dcgan_performance_cpu_smoke-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 11**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 10 / 1**. Held counts (measured and shown, not counted): **0 / 11 / 0**.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](bispf.md#cohort-cuda-3faddec347a0-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#gaussian1d_stability) | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | required | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 2**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 9 / 1**. Held counts (measured and shown, not counted): **0 / 10 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-3faddec347a0-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 5**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 9 / 1**. Held counts (measured and shown, not counted): **0 / 10 / 0**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-3faddec347a0-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_hold) | [removed: phase of ring_extension (counted there)](../test-dispositions-20261015/README.md#ring_hold) | PASS | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-3faddec347a0-experiment-ring_extension) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-3faddec347a0-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 3**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 9 / 0**. Held counts (measured and shown, not counted): **0 / 10 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-3faddec347a0-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-3faddec347a0-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | required | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | required | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_mass) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_unequal_width) | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_anisotropic) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#vector_overlap) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_bars4) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_blobs4) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | [held: no one recipe passes it together with the counted tests](../test-dispositions-20261015/README.md#img_intensity2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#grid100) | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#rotated100) | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | [held: no recipe has passed its current declaration](../test-dispositions-20261015/README.md#staggered100) | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-3faddec347a0-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-3faddec347a0-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-3faddec347a0-experiment-unipolar) | diagnostic | PASS | matches recorded run |
| [cover_leftover](bispf.md#cohort-cuda-3faddec347a0-experiment-cover_leftover) | diagnostic | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-3faddec347a0-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-3faddec347a0-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-3faddec347a0-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-3faddec347a0-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-3faddec347a0-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-3faddec347a0-experiment-grid100) | diagnostic | FAIL | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-3faddec347a0-experiment-rotated100) | diagnostic | FAIL | matches recorded run |
| [staggered100](bispf.md#cohort-cuda-3faddec347a0-experiment-staggered100) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-3faddec347a0-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-3faddec347a0-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-3faddec347a0-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-3faddec347a0-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.018445 | <= 0.35 | PASS |
| recon_mse | 0.00366872 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9b8b2cf4650242e991dd0ba8f4f40bb3.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) · [selected-family-coverage-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/3bc34531e5084d37b99744b7bcbfc612.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.962408 | >= 0.75 | PASS |
| leak_ratio | 0.0178634 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0258542 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.025216 | <= 0.2 | PASS |
| same_dir | 0.0130099 | <= 0.25 | PASS |
| u_kept | 0.999095 | >= 0.85 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/777abc459a7c4b898a0cb28e26bce070.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_b64"></a>

### dcgan_performance_b64

**dcgan_performance_b64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_b64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_cpu_smoke"></a>

### dcgan_performance_cpu_smoke

**dcgan_performance_cpu_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance_cpu_smoke / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance_cpu_smoke-tier-1).

Current pass criteria:


Declared budget: 3 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_r128"></a>

### dcgan_performance_r128

**dcgan_performance_r128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_r256"></a>

### dcgan_performance_r256

**dcgan_performance_r256: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r256.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_r32"></a>

### dcgan_performance_r32

**dcgan_performance_r32: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r32.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_r64"></a>

### dcgan_performance_r64

**dcgan_performance_r64: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_r64.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-dcgan_performance_w128"></a>

### dcgan_performance_w128

**dcgan_performance_w128: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/dcgan_performance_w128.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [dcgan_performance / Tier 1](bispf.md#cohort-cuda-3faddec347a0-dcgan_performance-tier-1).

Current pass criteria:


Declared budget: 13 updates; timeout 180 seconds.

Current measurement: particle_cloud prior (sigma 0); training_update_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-3faddec347a0-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. every joint generation/inverse check throughout the fixed continuation window

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 4834 |
| mass_apple | 0.21582 |
| mass_berry | 0.191406 |
| mass_grape | 0.205078 |
| mass_lemon | 0.193359 |
| mass_melon | 0.194336 |
| mass_tv | 0.0208984 |
| minimum_reconstruction_token_probability | 0.951635 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 0.00218637 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 4834 |

[Compact metrics and receipt provenance](../technique-receipts/9b9f896d59d0440383d8fef55c18098c.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full joint pass plus independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.195312 |
| mass_berry | 0.1875 |
| mass_grape | 0.210938 |
| mass_lemon | 0.198242 |
| mass_melon | 0.208008 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.999617 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 3.4861e-05 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/638005f1457b41e0a0f9daf790d49c01.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.135031 |
| finite_fraction | 1 |
| mean | 1.82751 |
| mean_error_sigma | 0.344978 |
| sample_count | 4096 |
| std | 0.491298 |
| std_ratio | 0.982596 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/70c0315cf2d6447bb4c4d6871ef63844.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0703186 |
| finite_fraction | 1 |
| mean | 3.02737 |
| mean_error_sigma | 0.0547359 |
| sample_count | 4096 |
| std | 0.438255 |
| std_ratio | 0.87651 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/baec8010a206497db6d595b59a9c412b.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.325427 |
| accuracy_pass | False |
| accuracy_score | 6.04411 |
| all_finite | True |
| center_max_sigma | 2.87639 |
| center_rms_sigma | 1.49713 |
| cov_frob_rms | 0.61016 |
| cov_trace_bias | 0.325427 |
| frozen_pass | False |
| mass_tv | 0.17028 |
| n | 100000 |
| passed | False |
| precision | 0.30497 |
| precision_gap | 0.683921 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.42394 |
| valid_n | 100000 |
| within_radius_n | 30497 |

[Compact metrics and receipt provenance](../technique-receipts/946fa7621c40477a8d7e89febd7f7cf7.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.96875 | >= 0.9 | PASS |
| modes | 2 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a8ed2f55cbf441eca23e3652ad626aab.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.8125 | >= 0.9 | FAIL |
| modes | 3 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8a8c883ed54049b39c3aca762a6f0e4b.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.59375 | >= 0.9 | FAIL |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/45d95e83dc7945d4bc461150312a32d4.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c04ee10ecc3e4ccf8e0dea67645acd11.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-3faddec347a0-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 0.999965 | >= 0.85 | PASS |
| concept_cos_plus | 0.999837 | >= 0.85 | PASS |
| concept_mag_minus | 0.996567 | <= 1.25 | PASS |
| concept_mag_plus | 1.00835 | <= 1.25 | PASS |
| identity_at_0 | 0.993095 | >= 0.85 | PASS |
| identity_at_mid | 0.969 | >= 0.85 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/465e27337bf04ae7a5f5453399733b8f.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-mode_hold"></a>

### mode_hold

**mode_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.969238 | >= 0.9 | PASS |
| modes | 8 | >= 8 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/ffb429b1b3314120a7cc791c68ef59b7.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.00023213 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/73104832003f4f7ca6ec21fc0a3a795b.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.466252 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.380435 | >= 0.15 | PASS |
| hq | 0.951904 | >= 0.85 | PASS |
| mass_tv | 0.092041 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/1565061ce93b4f3b96e0ba5199636b8e.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-ring_extension"></a>

### ring_extension

**ring_extension: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed every extension observation without selecting another window

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| confirmation_checks | 200 |
| converged_step | 1782 |
| convergence | first consecutive qualifying window; operational quality criterion |
| extension_checks | 300 |
| extension_passing | 300 |
| first_hold_failure | unavailable |
| frozen_toy_verdicts_unchanged | True |
| hold | the next updates, disjoint from confirmation; no restart after failure |
| hold_budget | 1200 |
| hold_budget_complete | True |
| hold_checks | 1200 |
| longest_settling_streak | 200 |
| min_hq | 0.9 |
| modes | 8 |
| name | first_convergence_then_hold_v1 |
| settling_budget | 4800 |
| settling_checks | 582 |
| settling_failures | 5 |
| start_step | 1200 |
| status | PASS |
| training_feedback | False |

[Compact metrics and receipt provenance](../technique-receipts/30323b33e3c943fbbd46f35f51e780fc.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-ring_hold"></a>

### ring_hold

**ring_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed first-convergence hold

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-3).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| confirmation_checks | 200 |
| converged_step | 1782 |
| convergence | first consecutive qualifying window; operational quality criterion |
| first_hold_failure | unavailable |
| frozen_toy_verdicts_unchanged | True |
| hold | the next updates, disjoint from confirmation; no restart after failure |
| hold_budget | 1200 |
| hold_budget_complete | True |
| hold_checks | 1200 |
| longest_settling_streak | 200 |
| min_hq | 0.9 |
| modes | 8 |
| name | first_convergence_then_hold_v1 |
| settling_budget | 4800 |
| settling_checks | 582 |
| settling_failures | 5 |
| start_step | 1200 |
| status | PASS |
| training_feedback | False |

[Compact metrics and receipt provenance](../technique-receipts/30323b33e3c943fbbd46f35f51e780fc.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | unavailable |
| accuracy_pass | False |
| accuracy_score | unavailable |
| all_finite | True |
| center_max_sigma | unavailable |
| center_rms_sigma | unavailable |
| cov_frob_rms | unavailable |
| cov_trace_bias | unavailable |
| frozen_pass | False |
| mass_tv | 0.11132 |
| n | 100000 |
| passed | False |
| precision | 0.22301 |
| precision_gap | 0.765881 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | unavailable |
| valid_n | 100000 |
| within_radius_n | 22301 |

[Compact metrics and receipt provenance](../technique-receipts/cff676fdda004c3696529ed7a082ddaf.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. coverage, sustained live accuracy, or independent holdout failed

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.54169 |
| accuracy_pass | False |
| accuracy_score | 5.59569 |
| all_finite | True |
| center_max_sigma | 2.5165 |
| center_rms_sigma | 1.14288 |
| cov_frob_rms | 0.708947 |
| cov_trace_bias | 0.54169 |
| frozen_pass | False |
| mass_tv | 0.12142 |
| n | 100000 |
| passed | False |
| precision | 0.33617 |
| precision_gap | 0.652721 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.369111 |
| valid_n | 100000 |
| within_radius_n | 33617 |

[Compact metrics and receipt provenance](../technique-receipts/31fcb73f64774c86973dcd3db11a13a5.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
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

<a name="cohort-cuda-3faddec347a0-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [adaptation / Tier 3](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-3) · [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-3faddec347a0-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000219656 | <= 0.02 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0e978476e02f43da81debbbe3c1b1784.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-3faddec347a0-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.876375 | <= 1 | PASS |
| mean_abs | 0.95826 | >= 0.3 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9377962203d1437cba35db1054915287.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-3faddec347a0-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-3faddec347a0-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.989121 | >= 0.85 | PASS |
| neu_hold | 0.995471 | >= 0.85 | PASS |
| off_caption | 0.000135526 | <= 0.05 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/12ebab6743264aaca9854e1d98bd2420.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.972497 | >= 0.85 | PASS |
| unused_hold | 0.989819 | >= 0.85 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c918480989174ef4af1532b57cfa750a.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-3faddec347a0-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.313814 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.32515 | >= 0.15 | PASS |
| hq | 0.995605 | >= 0.85 | PASS |
| mass_tv | 0.175212 | <= 0.15 | FAIL |
| sw1_normalized | 0.213624 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/554bda1905ef4a3eb7b825679bd9efe4.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.522296 | <= 0.45 | FAIL |
| mean_error | 0.101512 | <= 0.15 | PASS |
| sw1_normalized | 0.149086 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b0c6f67410034a4fa548079525dd41e3.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0804742 | <= 0.45 | PASS |
| mean_error | 0.0714902 | <= 0.15 | PASS |
| sw1_normalized | 0.0555712 | <= 0.18 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/af97fa02492a4f4c8ab8eaf18a094bbc.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.306438 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.483331 | >= 0.15 | PASS |
| hq | 0.990967 | >= 0.85 | PASS |
| mass_tv | 0.0588379 | <= 0.15 | PASS |
| sw1_normalized | 0.126826 | <= 0.18 | PASS |

Recorded terminal passing observations: **6**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/ee6f7c11f0a640ae95e684b119a92905.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 1.76764 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0 | >= 0.15 | FAIL |
| hq | 0.977295 | >= 0.85 | PASS |
| mass_tv | 0.0574219 | <= 0.15 | PASS |
| min_mass_ratio | 0 | >= 0.25 | FAIL |
| sw1_normalized | 0.101744 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c707b0c32be1463a84a9ffb92e87472e.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-3faddec347a0-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-3faddec347a0-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-3faddec347a0-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-3faddec347a0-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-3faddec347a0-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 9.6869 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.0934703 | >= 0.15 | FAIL |
| hq | 0.981934 | >= 0.85 | PASS |
| mass_tv | 0.277588 | <= 0.15 | FAIL |
| sw1_normalized | 0.287383 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/40c162a797fa4b989db47d03cf6f3020.json)

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

<a name="cohort-cuda-3faddec347a0-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-3faddec347a0-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-3faddec347a0-host_profile_transfer-tier-2).

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

- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895). Entropic Optimal Transport Sinkhorn-Knopp coupling used to construct transport chords and scale-free transport residuals.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base first-order optimizer family supported alongside DualNorm.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Relativistic pairing foundation extended by Sinkhorn Relativistic Pairing (SRpGAN).

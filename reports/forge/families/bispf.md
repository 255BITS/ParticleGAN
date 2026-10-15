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
- [configs/forge/studies/bispf-v1-study.json](../../../configs/forge/studies/bispf-v1-study.json)
- [reports/forge/bispf/README.md](../bispf/README.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

<a name="cohort-cuda-6421ba55ca82"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [bispf-v1](../../../configs/forge/ideas/bispf-v1.json).

Recorded qualification: **tier 1**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `fafd661541083840ae55fe9b4b282aa575f68f5f6652ecbe7b6e8e234ae36d8f`. Candidate revision: `45e5304bd6d19af271e0d2072cf1c5d36a3cf8cbcbf45e9eada0218c8ad0782b`. Runtime cohort: `c05372e7eb5c236de0a296c2b77ba9f572ee36ca06da7d7afcafa62653bbebcd`.

[Frozen numerical evidence](../technique-evidence/7f6745f3a0d18f563c25908c905055815aadcf36b077832dd6bf6ebb3f92bf2d.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: recorded_source_measurement. Recorded source-bound original measurement retained; stale against current task contracts. No current measurement, qualification reuse or comparable live ranking. Original selection: Complete Tier 1 and Tier 2 ordinary CUDA measurement of Born-Infeld Sinkhorn Particle Flow (BISPF) on the BCAP DualNorm leader recipe.

</details>

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation) | [3/3](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1) | [9/19](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-3) | [12/23 (1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-adaptation) |
| [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous) | [4/4](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) | [9/19](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | [13/30 (4 blocked, 3 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous) |
| [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability) | [6/6](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) | [10/21](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3) | [16/29 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability) |
| [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison) | [3/3](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1) | [9/19](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3) | [12/24 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison) |
| [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer) | [3/3](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1) | [9/19](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3) | [12/24 (1 blocked, 1 diagnostic attempts)](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer) |
| [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage) | [3/3](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1) | [9/19](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-3) | [12/22](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| View | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage) | [0/7 (7 not run)](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1) | [0/0](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-2) | [0/0](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-3) | [0/7 (7 not run)](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage) |

With run-attempt coverage, not run means no attempt; incompatible means a constructor refusal; blocked means a missing required artifact; diagnostic attempts retain separate-runtime outcomes. Other cohorts retain their recorded missing/preflight counts. All remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-6421ba55ca82-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring16_acquisition) | [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1) | PASS | changed since run |

<a name="cohort-cuda-6421ba55ca82-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | changed since run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | changed since run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | changed since run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | changed since run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2), [quality_coverage](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | FAIL | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_extension) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_hold) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3), [discriminator_stability](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3), [formulation_comparison](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3), [host_profile_transfer](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100_14k) | [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-6421ba55ca82-experiment-target_shift_recovery) | [adaptation](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-3), [clockfree_continuous](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-6421ba55ca82-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-6421ba55ca82-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bispf.md#cohort-cuda-6421ba55ca82-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | diagnostic | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | diagnostic | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | diagnostic | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring16_acquisition) | diagnostic | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 3**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 7**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bispf.md#cohort-cuda-6421ba55ca82-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 9**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring16_acquisition) | required | PASS | matches recorded run |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | required | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | required | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-6421ba55ca82-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-6421ba55ca82-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |
| [img_intensity2_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-6421ba55ca82-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-6421ba55ca82-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 2**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-6421ba55ca82-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-6421ba55ca82-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | required | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | required | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | required | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | required | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | required | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | required | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | required | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | required | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | required | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | required | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | required | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | required | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | required | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | required | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | required | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | required | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | required | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | required | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | required | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_smoke](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke) | diagnostic | PASS | matches recorded run |
| [gaussian1d_stability](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability) | diagnostic | FAIL | matches recorded run |
| [five_word_joint_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold) | diagnostic | PASS | matches recorded run |
| [trajectory](bispf.md#cohort-cuda-6421ba55ca82-experiment-trajectory) | diagnostic | PASS | matches recorded run |
| [residual_student](bispf.md#cohort-cuda-6421ba55ca82-experiment-residual_student) | diagnostic | PASS | matches recorded run |
| [unipolar](bispf.md#cohort-cuda-6421ba55ca82-experiment-unipolar) | diagnostic | PASS | changed since run |
| [cover_leftover](bispf.md#cohort-cuda-6421ba55ca82-experiment-cover_leftover) | diagnostic | PASS | changed since run |
| [mid_scale_identity](bispf.md#cohort-cuda-6421ba55ca82-experiment-mid_scale_identity) | diagnostic | PASS | matches recorded run |
| [mode_hold](bispf.md#cohort-cuda-6421ba55ca82-experiment-mode_hold) | diagnostic | PASS | matches recorded run |
| [vector_two_broad](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_two_broad) | diagnostic | PASS | matches recorded run |
| [vector_unequal_mass](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass) | diagnostic | FAIL | matches recorded run |
| [vector_unequal_width](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_unequal_width) | diagnostic | FAIL | matches recorded run |
| [vector_anisotropic](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_anisotropic) | diagnostic | FAIL | matches recorded run |
| [vector_overlap](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_overlap) | diagnostic | FAIL | matches recorded run |
| [vector_spiral](bispf.md#cohort-cuda-6421ba55ca82-experiment-vector_spiral) | diagnostic | PASS | matches recorded run |
| [img_stripes2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_stripes2) | diagnostic | PASS | matches recorded run |
| [img_bars4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_bars4) | diagnostic | FAIL | matches recorded run |
| [img_blobs4](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_blobs4) | diagnostic | FAIL | matches recorded run |
| [img_intensity2](bispf.md#cohort-cuda-6421ba55ca82-experiment-img_intensity2) | diagnostic | FAIL | matches recorded run |
| [grid100](bispf.md#cohort-cuda-6421ba55ca82-experiment-grid100) | diagnostic | FAIL | changed since run |
| [rotated100](bispf.md#cohort-cuda-6421ba55ca82-experiment-rotated100) | diagnostic | FAIL | changed since run |
| [staggered100](bispf.md#cohort-cuda-6421ba55ca82-experiment-staggered100) | diagnostic | FAIL | changed since run |

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-6421ba55ca82-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-selected-family-coverage-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bispf.md#cohort-cuda-6421ba55ca82-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-6421ba55ca82-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-6421ba55ca82-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0349811 | <= 0.35 | PASS |
| recon_mse | 0.00880684 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/10d0f94101a443c0b507cb1e23c44890.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **PASS** (new_attempt; qualification unchanged). declared state/horizon/cadence/restart comparisons agree; source audit bound [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) · [selected-family-coverage-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/9fb83c3425f8403c8e9fac0b13feed1e.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: PASS**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| content_kept | 0.984276 | >= 0.75 | PASS |
| leak_ratio | 0.00460331 | <= 0.2 | PASS |
| pole_rel_err_minus | 0.0216652 | <= 0.2 | PASS |
| pole_rel_err_plus | 0.0110928 | <= 0.2 | PASS |
| same_dir | 0.0150194 | <= 0.25 | PASS |
| u_kept | 1.01344 | >= 0.85 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/beec3429faef4670aa60b4e45cdaab13.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. every joint generation/inverse check throughout the fixed continuation window

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 6501 |
| mass_apple | 0.176758 |
| mass_berry | 0.207031 |
| mass_grape | 0.216797 |
| mass_lemon | 0.206055 |
| mass_melon | 0.193359 |
| mass_tv | 0.0298828 |
| minimum_reconstruction_token_probability | 0.999934 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 1.25331e-05 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 6501 |

[Compact metrics and receipt provenance](../technique-receipts/f4028fa01160452dbe941554aae6982d.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full joint pass plus independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| completed_steps | 20001 |
| mass_apple | 0.195312 |
| mass_berry | 0.198242 |
| mass_grape | 0.210938 |
| mass_lemon | 0.208008 |
| mass_melon | 0.1875 |
| mass_tv | 0.0189453 |
| minimum_reconstruction_token_probability | 0.999227 |
| modes | 5 |
| output_noise_added | 0 |
| policy_latent_perturbation | 0 |
| quality_fraction | 1 |
| reconstruction_exact | 1 |
| reconstruction_nll | 0.000138832 |
| sample_count | 1024 |
| served_averaged | 0 |
| step | 20001 |

[Compact metrics and receipt provenance](../technique-receipts/064cecd705564a789f1b90bbafb217d8.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.111459 |
| finite_fraction | 1 |
| mean | 2.1328 |
| mean_error_sigma | 0.265609 |
| sample_count | 4096 |
| std | 0.504105 |
| std_ratio | 1.00821 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/ebca93189f0e4b3fbfd1d999ee5ae9ea.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. stationary hold, deadline reacquisition and shifted hold

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.22428 |
| finite_fraction | 1 |
| mean | 3.16409 |
| mean_error_sigma | 0.328175 |
| sample_count | 4096 |
| std | 0.317076 |
| std_ratio | 0.634153 |
| step | 6000 |

[Compact metrics and receipt provenance](../technique-receipts/0031ba37faa14f999038fcfaddee07d5.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-grid100"></a>

### grid100

**grid100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.522187 |
| accuracy_pass | False |
| accuracy_score | 6.08734 |
| all_finite | True |
| center_max_sigma | 2.08598 |
| center_rms_sigma | 1.18048 |
| cov_frob_rms | 0.692701 |
| cov_trace_bias | 0.522187 |
| frozen_pass | False |
| mass_tv | 0.16603 |
| n | 100000 |
| passed | False |
| precision | 0.37015 |
| precision_gap | 0.618741 |
| problem | grid100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.418316 |
| valid_n | 100000 |
| within_radius_n | 37015 |

[Compact metrics and receipt provenance](../technique-receipts/4439b248968f4a0f8df5465ec19f67d8.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent grid100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-6421ba55ca82-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-img_bars4"></a>

### img_bars4

**img_bars4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/cbfc9f9cd70f40c99ba751daf7ac2b3d.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.875 | >= 0.9 | FAIL |
| modes | 3 | >= 4 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0bb6442eb34d4334b5b2422b7d3bbb42.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.5 | >= 0.9 | FAIL |
| modes | 1 | >= 2 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8a6f138f30cb4d86865639c572c61a3d.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: PASS**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 1 | >= 0.9 | PASS |
| modes | 2 | >= 2 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/036390fe010f4f00b5abc5e409af056a.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-6421ba55ca82-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_cos_minus | 0.999979 | >= 0.85 | PASS |
| concept_cos_plus | 0.999924 | >= 0.85 | PASS |
| concept_mag_minus | 0.997219 | <= 1.25 | PASS |
| concept_mag_plus | 1.00543 | <= 1.25 | PASS |
| identity_at_0 | 0.99768 | >= 0.85 | PASS |
| identity_at_mid | 0.973145 | >= 0.85 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f1783daa04b7416f9819e8bcc4d565e6.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-mode_hold"></a>

### mode_hold

**mode_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hq | 0.999512 | >= 0.9 | PASS |
| modes | 8 | >= 8 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/78f9cd763da04fc7a3a7d235eca25ded.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-residual_student"></a>

### residual_student

**residual_student: PASS**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000221774 | <= 0.02 | PASS |
| success_rate | 1 | >= 1 | PASS |
| wrong_pad_rate | 0 | <= 0 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a72beeb326ac40aeb4529359a5c72bb1.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.615957 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.276546 | >= 0.15 | PASS |
| hq | 0.936035 | >= 0.85 | PASS |
| mass_tv | 0.0917969 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/df208f4d5f9d4feab99e14fdac5e3d5c.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent ring_hold: certified artifact binding unavailable to this attempt [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3).

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

<a name="cohort-cuda-6421ba55ca82-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Run attempt: **PASS** (new_attempt; qualification unchanged). recomputed first-convergence hold [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3).

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

<a name="cohort-cuda-6421ba55ca82-experiment-rotated100"></a>

### rotated100

**rotated100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.469412 |
| accuracy_pass | False |
| accuracy_score | 5.93563 |
| all_finite | True |
| center_max_sigma | 2.64035 |
| center_rms_sigma | 1.36376 |
| cov_frob_rms | 0.756556 |
| cov_trace_bias | 0.469412 |
| frozen_pass | False |
| mass_tv | 0.14231 |
| n | 100000 |
| passed | False |
| precision | 0.26601 |
| precision_gap | 0.722881 |
| problem | rotated100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.394311 |
| valid_n | 100000 |
| within_radius_n | 26601 |

[Compact metrics and receipt provenance](../technique-receipts/a59db4ab92d44045be17de2d9f9eed72.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent rotated100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-6421ba55ca82-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-staggered100"></a>

### staggered100

**staggered100: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. coverage, sustained live accuracy, or independent holdout failed

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| abs_cov_trace_bias | 0.571332 |
| accuracy_pass | False |
| accuracy_score | 5.61197 |
| all_finite | True |
| center_max_sigma | 1.54329 |
| center_rms_sigma | 1.06759 |
| cov_frob_rms | 0.699556 |
| cov_trace_bias | 0.571332 |
| frozen_pass | False |
| mass_tv | 0.12023 |
| n | 100000 |
| passed | False |
| precision | 0.34184 |
| precision_gap | 0.647051 |
| problem | staggered100 |
| protocol | toy100-accuracy-v1 |
| radial_ks | 0.375711 |
| valid_n | 100000 |
| within_radius_n | 34184 |

[Compact metrics and receipt provenance](../technique-receipts/ec7f719ad37843c0b536f24169724eef.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **BLOCKED** (new_attempt; qualification unchanged). checkpoint parent staggered100: FAIL [Attempt evidence](../attempt-coverage.json).

Used by: [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-6421ba55ca82-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Run attempt: **FAIL** (new_attempt; qualification unchanged). recomputed live recovery and matched frozen negative control [Attempt evidence](../attempt-coverage.json).

Used by: [adaptation / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-3) · [clockfree_continuous / Tier 3](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-6421ba55ca82-experiment-trajectory"></a>

### trajectory

**trajectory: PASS**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| identity_mse | 0.000261136 | <= 0.02 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/78d7fdecada54a61ba05874e77ac56fe.json)

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-6421ba55ca82-experiment-two_pole"></a>

### two_pole

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.913537 | <= 1 | PASS |
| mean_abs | 0.958492 | >= 0.3 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/4fe88f917ed74a95a6c7170ec9f95d3f.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-unipolar"></a>

### unipolar

**unipolar: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cover | 0.993618 | >= 0.85 | PASS |
| neu_hold | 0.993527 | >= 0.85 | PASS |
| off_caption | 0.00266274 | <= 0.05 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/800c8f10664541c682c1ef6b8b32046a.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.927792 | >= 0.85 | PASS |
| unused_hold | 0.985734 | >= 0.85 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b51b096228f54d0baf40e02b0b67749c.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 2.00314 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.331639 | >= 0.15 | PASS |
| hq | 0.97168 | >= 0.85 | PASS |
| mass_tv | 0.190837 | <= 0.15 | FAIL |
| sw1_normalized | 0.214546 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/774433c2171548d094407f9246af1206.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.531405 | <= 0.45 | FAIL |
| mean_error | 0.278091 | <= 0.15 | FAIL |
| sw1_normalized | 0.194343 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7954040f61484c67aec06acd47612604.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| covariance_error | 0.0664334 | <= 0.45 | PASS |
| mean_error | 0.0218672 | <= 0.15 | PASS |
| sw1_normalized | 0.0351434 | <= 0.18 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/709883c2451748199927eff7067d46b8.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: PASS**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **PASS** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.576835 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.313988 | >= 0.15 | PASS |
| hq | 0.985352 | >= 0.85 | PASS |
| mass_tv | 0.0600586 | <= 0.15 | PASS |
| sw1_normalized | 0.10615 | <= 0.18 | PASS |

Recorded terminal passing observations: **14**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/26a191cb73ea4bfab90d07a9bc2f805a.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `0` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.586879 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.00433455 | >= 0.15 | FAIL |
| hq | 0.950928 | >= 0.85 | PASS |
| mass_tv | 0.0233691 | <= 0.15 | PASS |
| min_mass_ratio | 0.45166 | >= 0.25 | PASS |
| sw1_normalized | 0.0805109 | <= 0.18 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/3e4468c447344a368c81296108c53468.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Run attempt: **FAIL** (retained_selected_row; qualification unchanged). Matched existing execution receipt. [Attempt evidence](../technique-inventory.json).

Actual task device: `1` (recorded execution receipt).

Used by: [adaptation / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](bispf.md#cohort-cuda-6421ba55ca82-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.492137 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.339114 | >= 0.15 | PASS |
| hq | 0.914062 | >= 0.85 | PASS |
| mass_tv | 0.250244 | <= 0.15 | FAIL |
| sw1_normalized | 0.26655 | <= 0.18 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8b2eac3ed12549269a68483719552e24.json)

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

<a name="cohort-cuda-6421ba55ca82-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2).

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

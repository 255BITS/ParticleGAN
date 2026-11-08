# Baseline Formulaic Tier Evaluation Report

## 1. Executive Summary

This report establishes the baseline evaluation status for the ParticleGAN research program following the integration of Pull Request #499 (`fix(forge): execute all 400 selected-recipe x suite tasks to binary PASS/FAIL`, commit `83f7da2358e5a11746d3c10e64e62bed3e8b2f50`) into `develop`.

PR #499 systematically executed all 400 evaluation cells—comprising 8 candidate recipes evaluated across 50 distinct tasks—under a strict binary PASS or FAIL rule. Prior to PR #499, evaluation coverage contained unattempted, missing, or diagnostic-only gaps across several candidate-task pairs. By executing the full matrix, the repository established a definitive, empirical baseline across all qualification tiers.

Key findings of this baseline evaluation include:
- **Total Evaluations**: 400 recipe-task pairs (8 candidate recipes $\times$ 50 tasks).
- **Global Outcome Distribution**: 155 PASS (38.75%) and 245 FAIL (61.25%).
- **Tier 1 Overview**: 14 distinct tasks evaluated. Twelve tasks have at least two passing candidates; exactly two tasks fail across all 8 candidates (`model_ae_ddgan_gaussian_chain` and `model_ddgan_gaussian_chain` at 0/8).
- **Focus Family Coverage**: Among the core focus families (BISPF, Atlas, and BCAP), complementary strengths exist across non-DDGAN Tier 1 tasks. BISPF passes 8/14, Atlas passes 8/14, and BCAP passes 7/14 distinct Tier 1 tasks.
- **Purpose**: This report serves as the formal baseline for Milestone 2 (reclassifying or recalibrating the universal DDGAN Tier 1 failures) and Milestone 3 (recipe improvements and evaluations across BISPF, Atlas, and BCAP to achieve $\ge 1$ passing candidate per Tier 1 task).

---

## 2. Candidate Recipe Denominator

The baseline evaluation cohort fixes exactly one selected recipe for each of the eight primary candidate families. Each candidate is bound to an exact candidate ID and immutable revision hash:

| # | Family | Selected Recipe ID | Candidate Revision Hash | Description & Lineage |
|---|---|---|---|---|
| 1 | **BCAP** | `recipe-prior-refactor-bcap-pure-v1` | `f1c9188b6987d8ba26d7c9ed484d1db0db34065afab7ce4a05ecdf3711584540` | Bounded Critic Alignment Penalty with DualNorm optimizer; selected recipe declared directly without selecting grid search. |
| 2 | **BISPF** | `bispf-optimized-quintic-v1` | `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def` | Born-Infeld Sinkhorn Particle Flow with quintic kernel, EBI transport-chord penalty, and SBNP prior; selected recipe declared directly. |
| 3 | **Atlas** | `atlas-particle-normal-v4` | `efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd` | Particle-Normal Prior with DV12 stationarity policy and birth/death feature cell guard; selected recipe declared directly. |
| 4 | **E22** | `e22` | `26bc81696c14502e279a40afcf91ebfeed4c97777bc29579b1c1c9020876407c` | Baseline unregularized ParticleGAN formulation; untuned single-card reference baseline. |
| 5 | **GAN v3 (0.7)** | `recipe-prior-refactor-release07-gan-v3-v1` | `e3c2cae91adf0ccbedfcbc529cd1442eba3cd4dfc8399c37f63aa6d097b50770` | Historical release 0.7 GAN v3 formulation; tuned on Tier $\le 3$ from `release07-gan-v3-mog-family-defaults-round1-v1`. |
| 6 | **K3P** | `recipe-prior-refactor-k3p-v1` | `e36ba4acdb7f9424db9981ff248e93dfb6f0b256b8a7cf71af713b3a8d7f8e56` | Kinetic 3-Point particle formulation; tuned on Tier $\le 3$ from `k3p-family-defaults-round1-v1`. |
| 7 | **KA2** | `recipe-prior-refactor-ka2-v1` | `cc3bec6763cfcdc5236ac8eb61f6dd2d952e84790322fad0bc05622f4f0a5e98` | Kinetic Anchor 2 de-clocked particle flow; tuned on Tier $\le 3$ from `ka2-family-defaults-round1-v1`. |
| 8 | **R1/R2** | `recipe-prior-refactor-r1r2-v1` | `594012d6f37dcf9bb8d8c2e9c84085cfcd896d5d44843496901fb5d0019c617a` | Gradient penalty baseline (R1 real gradient penalty + R2 fake gradient penalty); tuned on Tier $\le 3$ from `r1r2-modern-family-round1-v1`. |

---

## 3. Tier 1 Evaluation Matrix

Tier 1 represents the foundational smoke tier gating all subsequent qualification. In the primary qualification view (`discriminator_stability`, revision 14), Tier 1 encompasses 14 distinct tasks (13 required tasks and 1 diagnostic measurement task).

### 3.1 Per-Task Binary Outcomes

The table below presents the exact binary PASS/FAIL outcome for all 14 distinct Tier 1 tasks across each evaluated candidate recipe:

| Task ID | Importance | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate | Status / Diagnostic Note |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `ae_gan_hold` | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Passes all 8 candidates |
| `clockfree_audit_measurement_v1` | Diagnostic | PASS | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 3/8 | Passes BCAP, BISPF, Atlas |
| `five_word_joint_smoke` | Required | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | 6/8 | Atlas & E22 fail |
| `gaussian1d_smoke` | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Passes all 8 candidates |
| `model_ae_ddgan_gaussian_chain` | Required | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **0/8** | **UNIVERSAL FAILURE (0/8)** |
| `model_ae_gan_multimodal` | Required | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | 2/8 | Passes GAN v3, R1/R2 |
| `model_ae_reconstruction` | Required | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 7/8 | BCAP fails |
| `model_ddgan_gaussian_chain` | Required | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **0/8** | **UNIVERSAL FAILURE (0/8)** |
| `model_vae_categorical` | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 | BCAP & BISPF fail |
| `model_vae_hard` | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 | BCAP & BISPF fail |
| `ring16_acquisition` | Required | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 2/8 | BCAP & BISPF pass |
| `two_pole` | Required | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | 6/8 | Atlas & E22 fail |
| `unused_token_hold` | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Passes all 8 candidates |
| `zero_tanh_smoke` | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 | BCAP & BISPF fail |
| **Total Tier 1 Distinct Passes** | — | **7/14** | **8/14** | **8/14** | **7/14** | **10/14** | **9/14** | **9/14** | **10/14** | — | — |
| **Tier 1 View Reqs (out of 29)** | — | **22/29** | **23/29** | **18/29** | **17/29** | **25/29** | **24/29** | **24/29** | **25/29** | — | Repeated tests across 6 views |

### 3.2 Analysis of Universal Failures (0/8)

Only two Tier 1 tasks fail universally across all eight candidate recipes:
1. **`model_ae_ddgan_gaussian_chain`** (0/8 PASS): Evaluates an autoencoding Denoising Diffusion GAN architecture trained over a 4-step reverse diffusion chain on Gaussian data. The combination of multi-step diffusion sampling noise, paired latent autoencoding, and unadjusted discriminator gradient penalty prevents any recipe from achieving terminal reconstruction and stability thresholds within 2,500 updates.
2. **`model_ddgan_gaussian_chain`** (0/8 PASS): Evaluates a standard unconditional DDGAN generator with 4 reverse diffusion steps on Gaussian targets. Like its autoencoding counterpart, all 8 recipes fail to reach the terminal coverage gates under current default hyperparameters within 2,500 updates.

Because these tasks represent multi-step diffusion formulation challenges rather than standard adversarial smoke tests, they are prime candidates for reclassification to Tier 2 under Milestone 2.

---

## 4. Focus Families Baseline Breakdown (BISPF, Atlas, BCAP)

The original project scope specifically targets deep-dive evaluations and recipe improvements across three focus families: **BISPF**, **Atlas**, and **BCAP**.

### 4.1 BISPF (`bispf-optimized-quintic-v1`)
- **Tier 1 Distinct Outcome**: 8 PASS / 6 FAIL (23/29 view requirements).
- **Passing Tasks (8)**: `ae_gan_hold`, `clockfree_audit_measurement_v1`, `five_word_joint_smoke`, `gaussian1d_smoke`, `model_ae_reconstruction`, `ring16_acquisition`, `two_pole`, `unused_token_hold`.
- **Failing Tasks (6)**:
  - `model_ae_ddgan_gaussian_chain` (universal failure)
  - `model_ddgan_gaussian_chain` (universal failure)
  - `model_ae_gan_multimodal`
  - `model_vae_categorical`
  - `model_vae_hard`
  - `zero_tanh_smoke`
- **Family Profile**: BISPF demonstrates exceptional transport stability and multi-modal mode capture, passing `ring16_acquisition` and `two_pole`. However, its Smooth Born-Infeld potential and zero-mean gauge struggle with latent variational constraints in VAE tasks and saturated tanh boundaries.

### 4.2 Atlas (`atlas-particle-normal-v4`)
- **Tier 1 Distinct Outcome**: 8 PASS / 6 FAIL (18/29 view requirements).
- **Passing Tasks (8)**: `ae_gan_hold`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `model_ae_reconstruction`, `model_vae_categorical`, `model_vae_hard`, `unused_token_hold`, `zero_tanh_smoke`.
- **Failing Tasks (6)**:
  - `model_ae_ddgan_gaussian_chain` (universal failure)
  - `model_ddgan_gaussian_chain` (universal failure)
  - `model_ae_gan_multimodal`
  - `five_word_joint_smoke`
  - `ring16_acquisition`
  - `two_pole`
- **Family Profile**: Atlas exhibits strong compatibility with variational inference architectures and boundary constraints, cleanly passing `model_vae_categorical`, `model_vae_hard`, and `zero_tanh_smoke`. However, its Particle-Normal prior and stationary policy encounter difficulties on complex multi-modal cluster acquisition (`ring16_acquisition`) and transfer tasks (`two_pole`, `five_word_joint_smoke`).

### 4.3 BCAP (`recipe-prior-refactor-bcap-pure-v1`)
- **Tier 1 Distinct Outcome**: 7 PASS / 7 FAIL (22/29 view requirements).
- **Passing Tasks (7)**: `ae_gan_hold`, `clockfree_audit_measurement_v1`, `five_word_joint_smoke`, `gaussian1d_smoke`, `ring16_acquisition`, `two_pole`, `unused_token_hold`.
- **Failing Tasks (7)**:
  - `model_ae_ddgan_gaussian_chain` (universal failure)
  - `model_ddgan_gaussian_chain` (universal failure)
  - `model_ae_gan_multimodal`
  - `model_ae_reconstruction`
  - `model_vae_categorical`
  - `model_vae_hard`
  - `zero_tanh_smoke`
- **Family Profile**: BCAP pairs Bounded Critic Alignment with DualNorm optimization, securing passes on geometric transfer problems (`two_pole`, `ring16_acquisition`, `five_word_joint_smoke`). However, the critic alignment penalty currently impedes encoder-decoder reconstruction matching in `model_ae_reconstruction` and fails VAE/tanh smoke tasks.

### 4.4 Complementary Coverage Matrix Across Focus Families

The focus families exhibit remarkably complementary performance on Tier 1 tasks:

| Tier 1 Task | BISPF | Atlas | BCAP | Combined Focus Family Status |
|---|:---:|:---:|:---:|:---:|
| `ae_gan_hold` | PASS | PASS | PASS | **PASS (3/3)** |
| `clockfree_audit_measurement_v1` | PASS | PASS | PASS | **PASS (3/3)** |
| `five_word_joint_smoke` | PASS | FAIL | PASS | **PASS (2/3)** |
| `gaussian1d_smoke` | PASS | PASS | PASS | **PASS (3/3)** |
| `model_ae_ddgan_gaussian_chain` | **FAIL** | **FAIL** | **FAIL** | **FAIL (0/3)** — Universal failure |
| `model_ae_gan_multimodal` | FAIL | FAIL | FAIL | **FAIL (0/3)** — Passes GAN v3 & R1/R2 |
| `model_ae_reconstruction` | PASS | PASS | FAIL | **PASS (2/3)** |
| `model_ddgan_gaussian_chain` | **FAIL** | **FAIL** | **FAIL** | **FAIL (0/3)** — Universal failure |
| `model_vae_categorical` | FAIL | PASS | FAIL | **PASS (1/3)** |
| `model_vae_hard` | FAIL | PASS | FAIL | **PASS (1/3)** |
| `ring16_acquisition` | PASS | FAIL | PASS | **PASS (2/3)** |
| `two_pole` | PASS | FAIL | PASS | **PASS (2/3)** |
| `unused_token_hold` | PASS | PASS | PASS | **PASS (3/3)** |
| `zero_tanh_smoke` | FAIL | PASS | FAIL | **PASS (1/3)** |

Across all 14 Tier 1 tasks, the three focus families jointly cover 11 tasks with at least one PASS. Only three tasks fail across all three focus families:
- `model_ae_gan_multimodal`: Already solved by GAN v3 and R1/R2; accessible to focus families via formulation tuning.
- `model_ae_ddgan_gaussian_chain`: Universal failure across all 8 recipes.
- `model_ddgan_gaussian_chain`: Universal failure across all 8 recipes.

---

## 5. Tier 2, Tier 3, and Tier 4 Evaluation Breakdown

### 5.1 Tier 2: Quality & Transfer (23 Tasks Total)

In `discriminator_stability`, Tier 2 comprises 16 counted (required) tasks and 7 held tasks.

#### 5.1.1 Counted / Required Tier 2 Tasks (16 Tasks)

| Task ID | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate | Outcome Note |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `cover_leftover` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Universal pass |
| `five_word_joint_hold` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | 4/8 | Passes BCAP, BISPF, K3P, KA2 |
| `gaussian1d_stability` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (long-horizon noise stability) |
| `grid100` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (100-mode grid collapse) |
| `img_stripes2` | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL | 6/8 | GAN v3 & R1/R2 fail |
| `mid_scale_identity` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Universal pass |
| `mode_hold` | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 2/8 | Passes BISPF & Atlas |
| `model_ae_ddgan_bimodal_chain` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (bimodal DDGAN autoencoding) |
| `model_ddgan_bimodal_chain` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (bimodal DDGAN unconditional) |
| `residual_student` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | 7/8 | R1/R2 fails |
| `rotated100` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (rotated 100-mode ring) |
| `staggered100` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (staggered 100-mode grid) |
| `trajectory` | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | 5/8 | Atlas, E22, R1/R2 fail |
| `unipolar` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Universal pass |
| `vector_spiral` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Universal pass |
| `vector_two_broad` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 | Universal pass |
| **Total Tier 2 Counted Passes** | **9/16** | **10/16** | **8/16** | **7/16** | **7/16** | **9/16** | **9/16** | **5/16** | — | — |

#### 5.1.2 Held Tier 2 Tasks (7 Tasks)

Held tasks are executed and reported under standard binary evaluation, but are excluded from qualification totals pending satisfaction of declared re-entry conditions (e.g., a single recipe passing the held task alongside all counted Tier 1 and Tier 2 tasks).

| Held Task ID | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate | Core Challenge & Reason for Hold |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `img_bars4` | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | 2/8 | BCAP/BISPF generate only 2 of 4 bars; passers fail Tier 1. |
| `img_blobs4` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | 1/8 | Requires 4 clean corners ($29/32$); passers fail Tier 1. |
| `img_intensity2` | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | PASS | 3/8 | Two brightness levels; passers fail Tier 1 counted tasks. |
| `vector_anisotropic` | FAIL | FAIL | PASS | PASS | FAIL | PASS | PASS | PASS | 5/8 | Three tilted ellipsoids; TV distance bound exceeds 0.15 for BCAP/BISPF. |
| `vector_overlap` | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 3/8 | Overlapping Gaussian clouds; centroid drift exceeds 0.15 for BCAP/BISPF. |
| `vector_unequal_mass` | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 | 2% rare mode collapses in most candidates; passes Atlas. |
| `vector_unequal_width` | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 | Unequal variance modes; mass TV trade-off against unequal mass; passes Atlas. |

### 5.2 Tier 3: Endurance & Extreme Scaling (8 Tasks Total)

Tier 3 tests long-horizon stability (up to 14,000 updates), persistent mode retention, and dynamic perturbation recovery.

| Task ID | Role | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate | Evaluation Note |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `clockfree_audit` | Required (CC) | PASS | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 3/8 | Passes BCAP, BISPF, Atlas (clockfree_continuous view) |
| `grid100_14k` | Required (CC) | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (14,000-step 100-mode grid) |
| `ring_extension` | Required (DS) | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 | Passes BISPF (uninterrupted 1,200-check ring run) |
| `ring_hold` | Diagnostic | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 | Phase of `ring_extension` (counted via `ring_extension`) |
| `rotated100_14k` | Required (CC) | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (14,000-step rotated 100-mode ring) |
| `staggered100_14k` | Required (CC) | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (14,000-step staggered 100-mode grid) |
| `target_shift_recovery` | Required (CC) | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Universal fail (dynamic target centroid shift) |
| `zero_tanh_rail_escape` | Held (DS) | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | 2/8 | Held; passes E22 & K3P (dynamic rail escape) |
| **Total Tier 3 Passes** | — | **1/8** | **2/8** | **1/8** | **1/8** | **0/8** | **1/8** | **0/8** | **0/8** | — | — |

*(Note: CC denotes `clockfree_continuous` view; DS denotes `discriminator_stability` view).*

### 5.3 Tier 4: Stress & Architecture Formulations (5 Tasks Total)

All 5 tasks in Tier 4 are designated as `held` in the primary qualification view (`discriminator_stability`). As a result, Tier 4 contains zero required view-requirement instances in the standard qualification denominator.

| Held Task ID | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate | Declared Stress / Formulation Question |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `grid100_mlp16` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | 16-hidden-layer, 128-wide LeakyReLU MLP generator on 100-mode grid. |
| `grid100_res16_tanh` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | 16-layer residual MLP with $S=5.25$ bounded tanh head on 100-mode grid. |
| `misgan_block_neural_imputer` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | 3-pair MisGAN neural imputer on ambiguous single-sensor missingness. |
| `misgan_block_recovery` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | 2-pair incomplete-data MisGAN on 8D manifold block recovery. |
| `ring16_bigan_continuous` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 | Continuous 2D multi-modal joint BiGAN $(G, E, D)$ on 16-Gaussian ring. |

---

## 6. Complete 50-Task Master Binary Matrix

The master evaluation matrix below catalogs the binary status of all 50 tasks across all 8 candidates in alphabetical order:

| Task ID | Tier | Role | BCAP | BISPF | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 | Pass Rate |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ae_gan_hold` | T1 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `clockfree_audit` | T3 | Required | PASS | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 3/8 |
| `clockfree_audit_measurement_v1` | T1 | Diagnostic | PASS | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 3/8 |
| `cover_leftover` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `five_word_joint_hold` | T2 | Required | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | 4/8 |
| `five_word_joint_smoke` | T1 | Required | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | 6/8 |
| `gaussian1d_smoke` | T1 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `gaussian1d_stability` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `grid100` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `grid100_14k` | T3 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `grid100_mlp16` | T4 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `grid100_res16_tanh` | T4 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `img_bars4` | T2 | Held | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | 2/8 |
| `img_blobs4` | T2 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | 1/8 |
| `img_intensity2` | T2 | Held | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | PASS | 3/8 |
| `img_stripes2` | T2 | Required | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL | 6/8 |
| `mid_scale_identity` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `misgan_block_neural_imputer` | T4 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `misgan_block_recovery` | T4 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `mode_hold` | T2 | Required | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 2/8 |
| `model_ae_ddgan_bimodal_chain` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `model_ae_ddgan_gaussian_chain` | T1 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `model_ae_gan_multimodal` | T1 | Required | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | 2/8 |
| `model_ae_reconstruction` | T1 | Required | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 7/8 |
| `model_ddgan_bimodal_chain` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `model_ddgan_gaussian_chain` | T1 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `model_vae_categorical` | T1 | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 |
| `model_vae_hard` | T1 | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 |
| `residual_student` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | 7/8 |
| `ring16_acquisition` | T1 | Required | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 2/8 |
| `ring16_bigan_continuous` | T4 | Held | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `ring_extension` | T3 | Required | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 |
| `ring_hold` | T3 | Diagnostic | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 |
| `rotated100` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `rotated100_14k` | T3 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `staggered100` | T2 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `staggered100_14k` | T3 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `target_shift_recovery` | T3 | Required | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | 0/8 |
| `trajectory` | T2 | Required | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | 5/8 |
| `two_pole` | T1 | Required | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | 6/8 |
| `unipolar` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `unused_token_hold` | T1 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `vector_anisotropic` | T2 | Held | FAIL | FAIL | PASS | PASS | FAIL | PASS | PASS | PASS | 5/8 |
| `vector_overlap` | T2 | Held | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 3/8 |
| `vector_spiral` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `vector_two_broad` | T2 | Required | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 8/8 |
| `vector_unequal_mass` | T2 | Held | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 |
| `vector_unequal_width` | T2 | Held | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1/8 |
| `zero_tanh_rail_escape` | T3 | Held | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | 2/8 |
| `zero_tanh_smoke` | T1 | Required | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 6/8 |
| **Total Passes per Candidate** | — | — | **17/50** | **21/50** | **22/50** | **17/50** | **19/50** | **21/50** | **19/50** | **19/50** | **155/400** |

---

## 7. Methodological Details and Evaluation Rigor

To ensure absolute experimental reproducibility and adherence to repository governance standards, all evaluations summarized in this report comply with the following rigorous protocols:

1. **Protocol Screening Seed 0**:
   - Every candidate and task evaluation is executed using the repository's fixed protocol screening seed `0`.
   - Constructor, data streaming, training-noise, and evaluation RNG streams are strictly isolated through independent named generators (`torch.Generator`).
2. **Public Deterministic Initializer**:
   - Models are initialized using the repository's public deterministic orthogonal initializer (`deterministic_orthogonal`).
   - Fixed baseline fixtures (such as zero-particle fallbacks or stored host weights) are tracked in separate cohorts and never silently substituted for the shared baseline.
3. **Execution Hardware & Environment**:
   - Evaluations were performed on `NVIDIA RTX A6000` hardware with CUDA acceleration.
   - The primary evaluation view is `discriminator_stability` (revision 14) with source digest `2e8a97a0aa84`.
4. **Binary PASS/FAIL Evaluation Law**:
   - Evaluations admit no third or intermediate state.
   - Any constructor refusal, preflight blocker, missing dependency artifact, timeout, or numerical failure is recorded strictly as `FAIL`.
   - Passing requires satisfying all declared numerical gates, terminal checks, and stability bounds without exception.
5. **View Requirements vs. Distinct Tests**:
   - The repository tracks 6 primary evaluation views: `adaptation`, `clockfree_continuous`, `discriminator_stability`, `formulation_comparison`, `host_profile_transfer`, and `quality_coverage`.
   - Across these 6 views, there are **115 counted view-requirement instances** drawn from **36 counted distinct tests**.
   - The full evaluation catalog comprises 49 distinct tests plus 1 diagnostic test (`ring_hold`), yielding exactly 50 distinct task evaluations per candidate (400 candidate $\times$ task cells).

---

## 8. Roadmap for Subsequent Milestones

With the baseline evaluation report established, the research plan proceeds with the following milestones:

1. **Milestone 2: Tier 1 DDGAN Calibration & Retiering**:
   - Address the two universal Tier 1 failures (`model_ae_ddgan_gaussian_chain` and `model_ddgan_gaussian_chain`).
   - Provide empirical analysis demonstrating that 4-step reverse diffusion chain tasks exceed the smoke-test complexity appropriate for Tier 1.
   - Reclassify these tasks to Tier 2 in `configs/forge/views/discriminator_stability.json` with documented justification, opening a PR to `develop`.
2. **Milestone 3: Focus Family Progress & Tier 1 Resolution**:
   - Target failing Tier 1 tasks across BISPF, Atlas, and BCAP (`model_ae_gan_multimodal`, `model_ae_reconstruction`, `ring16_acquisition`, `two_pole`, `five_word_joint_smoke`, `model_vae_categorical`, `model_vae_hard`, `zero_tanh_smoke`).
   - Conduct deep-dive evaluations and formulation/recipe refinements to ensure that **every Tier 1 task has at least one candidate recipe passing**.
   - PR and merge verified recipe updates into `develop`.
3. **Milestone 4: Final Integration & Qualification**:
   - Execute full repository test suites (`pytest`), verify Forge readouts currency, and confirm clean qualification state without untracked logs.

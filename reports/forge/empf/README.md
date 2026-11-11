# Exponential Moving Prototype Flow (`EMPF`)

`EMPF` (`critic_formulation="empf"`, `prior_regularizer="empf"`, `prior_reg=0.01`, preset `"empf"`) is a clock-free, constant-learning-rate ParticleGAN formulation inspired by *e-means* (Bessa et al., arXiv:2611.09501). It replaces noisy raw-minibatch transport targets with a persistent online sufficient-statistics prototype bank $(S, N)$ checkpointed inside the critic step record, with no EMA critic network copy (`ema_critic=None`, `reg_anchor_weight=0.0`) and no step-count schedules.

## Four Formulation Pillars

1. **Online `e-means` Sufficient-Statistics Prototype Bank ([`update_emeans_prototypes`](../../../particlegan/_empf/prototypes.py))**:
   Maintains unnormalized feature sums $S \in \mathbb{R}^{K \times d}$ and assignment counts $N \in \mathbb{R}_+^K$ ($K = \min(K_{\max}, B_r)$) with exact centroids $C_k = S_k / N_k$. Instead of dataset-size or epoch-horizon schedules, EMPF adapts the EMA rate purely from the per-batch cluster hit ratio:
   $$\alpha_{\mathrm{eff}}(K, B_r) = \operatorname{clip}\!\left(\alpha_0 \left(\frac{B_r}{K}\right)^\eta,\; \alpha_{\min},\; \alpha_{\max}\right)$$
   Dead prototypes ($N_k < \tau_{\mathrm{rev}}\bar{N}$) revive deterministically via mass-conserving proportional splitting of the heaviest prototype ($S_{\mathrm{donor}}, N_{\mathrm{donor}} \to \frac{1}{2}$ each) along a low-discrepancy R2 perturbation direction, using zero RNG draws.
2. **Mass-Weighted Prototype-to-Fake Entropic Optimal Transport ([`mass_weighted_sinkhorn`](../../../particlegan/_empf/transport.py))**:
   Couples $C$ to detached generator samples $y = G(z)$ with non-uniform row marginals $w_k = N_k / \sum_j N_j$ and uniform column marginals $1/B_f$, preserving rare-mode mass instead of forcing uniform batch-to-batch row matching. Conditioned critic calls automatically bypass the marginal bank and evaluate fiber-aligned conditional chords.
3. **Prototype-Chord Elastic Born-Infeld Critic Penalty & Zero-Mean Gauge ([`EMPFCriticPenalty`](../../../particlegan/_empf/penalty.py))**:
   Bounds critic input gradients on real endpoints, fake endpoints, and stratified prototype-to-fake Sinkhorn chords using the smooth Born-Infeld potential $\phi_\beta(\|g\| - \kappa)^2$ plus a harmonic floor and zero-mean logit gauge $\frac{\gamma_{\mathrm{gauge}}}{2}\bar{D}^2$.
4. **Degree-of-Freedom-Aware EMPF Prior Regularizer ([`EMPFPriorRegularizer`](../../../particlegan/_empf/prior.py)) & Instance-Local Thermal Cooling ([`EMPFCriticAdam`](../../../particlegan/_empf/critic.py))**:
   Combines initial-moment tethering, scale-free $k$-NN log-repulsion, local scatter circularization, and a local median-normalized 1-NN revival tether for stranded prior particles while skipping global moment/covariance distortion on low-count discrete codebooks ($N \le \text{DoF}$, with a short-range one-sided prototype non-collision hinge on reference-anchored discrete codebooks). Thermal state travels exclusively on the instance-bound critic record (`bind_empf_record`), with zero module-global caches.

## Ordinary CUDA Qualification Summary (`empf-v1`, `discriminator_stability` Rev 16)

Executed at frozen commit `cb5e484b059d3474f7259c40beb7acb4b5bb8704` on `NVIDIA RTX A6000` (`40` executed attempts, `0` `INCOMPLETE`):

- **Tier 1 Qualification (`PASS`, `11/11` required + `1/1` `clockfree_audit_measurement_v1` diagnostic)**:
  Passes every required Tier 1 task (`ae_gan_hold`, `five_word_joint_smoke`, `gaussian1d_smoke`, `model_ae_gan_multimodal`, `model_ae_reconstruction`, `model_vae_categorical`, `model_vae_hard`, `ring16_acquisition`, `two_pole`, `unused_token_hold`, `zero_tanh_smoke`) and `clockfree_audit_measurement_v1`.
- **Tier 2 Screening (`FAIL`, `8/18` required PASS, `20/38` executed tasks PASS overall)**:
  Passes `cover_leftover`, `img_stripes2`, `mid_scale_identity`, `residual_student`, `trajectory`, `unipolar`, `vector_spiral`, and `vector_two_broad`. On the held imbalanced benchmark `vector_unequal_mass`, mass-weighted Sinkhorn ($w_k = N_k / \sum_j N_j$) preserves `2.98%` mass (`122/4096`) on the `2%` rare mode with `component_core_min_eigen_ratio = 0.0595` (`15x` BISPF's `0.004` and `6.6x` BCAP's `0.009`).
- **Identified Structural Bottlenecks for EMPF v2**:
  1. Fixed $K_{\max}=64$ prototypes bottleneck `100`-mode grids (`grid100`, `rotated100`, `staggered100`, where $K=64 < M=100$).
  2. Persistent prototype memory $(S, N)$ exhibits Mode Hold drop (`mode_hold` `hq=0.8411 < 0.90`, `5/8` modes) and shift reacquisition lag (`gaussian1d_stability` `mean_error_sigma=0.2869` after abrupt distribution shift).
  3. Unconditioned concatenated pairs $[x_t, x_{t+1}]$ in DDGAN chains (`model_ddgan_gaussian_chain`, `model_ae_ddgan_gaussian_chain`) pool across timesteps in a marginal prototype bank.
  No public-default adoption claim (`DEFAULT_RECIPE` remains `"bispf_optimized"`).

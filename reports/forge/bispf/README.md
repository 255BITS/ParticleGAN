# Born-Infeld Sinkhorn Particle Flow (`BISPF`)

`BISPF` (`critic_formulation="bispf"`, `prior_regularizer="sbnp"`, `prior_reg=0.01`) is a modality-agnostic, constant-learning-rate ParticleGAN formulation with no EMA critic copy (`ema_critic=None`, `reg_anchor_weight=0.0`) and no time-dependent LR or noise annealing schedules.

## Four Pillars

1. **Scale-Free Sinkhorn Relativistic Pairing (`SRpGAN` / [`sinkhorn_coupling`](../../../particlegan/bispf.py))**:
   Normalizes pairwise squared distances by the detached in-batch median squared distance $\bar{d}^2$ and computes a doubly-stochastic entropic Optimal Transport coupling $\Pi$ via 5 log-domain Sinkhorn-Knopp iterations.
2. **Elastic Born-Infeld Transport-Chord Regularizer (`EBI-TCR` / [`_BISPFPenaltyKernel`](../../../particlegan/bispf.py)) + Zero-Mean Logit Gauge**:
   Evaluates the two-regime Elastic Born-Infeld potential $\psi_{\mathrm{EBI}}(g) = \frac{\lambda_{\mathrm{floor}}}{2 d} g^2 + \frac{\lambda_{\mathrm{cap}}}{2} \operatorname{softplus}_\beta(g - \kappa)^2$ along real endpoints, fake endpoints, and stratified low-discrepancy Sinkhorn transport chords $x_{\mathrm{chord}} = (1 - u) x_f + u x_r^{(\pi)}$, paired with a zero-mean logit gauge $\frac{\gamma_{\mathrm{gauge}}}{2}\bar{D}^2$ that pins translational logit drift without an EMA critic network copy.
3. **Stratified Blue-Noise Particle Prior (`SBNP` / [`SBNPPriorRegularizer`](../../../particlegan/bispf.py))**:
   Combines an adaptive initial-moment tether (`capture_initial_moments`) that automatically disables covariance distortion when $N \le 2 k_{\mathrm{nn}} + d_z$ (preserving discrete 5-word codebooks on `five_word_joint_smoke` and `five_word_joint_hold`) with scale-free $k$-NN log-repulsion in latent space when $N > 2 k_{\mathrm{nn}} + d_z$.
4. **Harmonic Mode Capture Thermal Cooling (Regime II)**:
   Monitors the scale-free bidirectional 95th-percentile nearest-neighbor distance $\max(\operatorname{quantile}_{0.95}(\min_j d_{ij}/\bar{d}), \operatorname{quantile}_{0.95}(\min_i d_{ij}/\bar{d}))$ and applies straight-through gradient cooling below DualNorm's smoothing floor once all modes are captured inside the harmonic basin, preventing finite-batch Poisson jitter over long post-convergence horizons (`mode_hold`, `ring_hold`, `ring_extension`).

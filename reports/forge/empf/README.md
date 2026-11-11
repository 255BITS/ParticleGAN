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

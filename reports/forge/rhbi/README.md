# Relativistic Hellinger-Born-Infeld Transport (RHBI)

## Overview

**Relativistic Hellinger-Born-Infeld Transport (`rhbi`)** replaces unscreened adversarial generator objectives with a **Debiased Slack-Screened Unbalanced Sinkhorn** variational operator (`RHBILoss` / `RHBIGANLoss`) paired with a **Weyl-phased balanced Born-Infeld geodesic chord + endpoint Lipschitz cap** (`RHBICriticPenalty`), an **Ajtai-Komlós-Tusnády (AKT) empirical OT local horizon** ($\ell_t$), an **impedance-matched 5-channel Kantorovich-Hellinger thermal equilibrium controller**, and degree-of-freedom-aware SBNP particle prior regularization (`prior_reg = 0.02`, `prior_lr_mult = 2.9`) under strictly constant learning rates (`lr = 0.014`, `lr_floor = 1.0`, `network_lr_floor = 1.0`), zero EMA critic copies, and zero task-shape or step-count gates.

## Target Failure Modes

1. **Poisson Shot-Noise Floor ($B \lesssim M$) & Empty-Valley Bridging:** Linear Born-Infeld horizon screening ($\rho = 3.0$, `rhbi_rapidity_screening = False`) routes unmatched batch singletons and empty-valley midpoint particles ($\hat{c}_{ij} > \rho$) cleanly into the $(B+1)$-th Kantorovich-Hellinger slack reservoir rather than trapping them in 50/50 cross-mode active couplings, while `rhbi_slack_gain = 0.35` amplifies relativistic log-odds ascent in Proportion to total slack fraction $s_f$.
2. **1D Empirical OT Horizon Contraction ($d = 1$):** Nearest-neighbor spacings scale as $B^{-1/d}$ whereas empirical optimal transport fluctuations scale as $B^{-\min(1/d,\,1/2)}$ (Ajtai-Komlós-Tusnády). Scaling the 25th-quantile real NN distance by $B_r^{\max(0,\,1/d - 1/2)}$ (`rhbi_akt_horizon = True`) multiplies $d = 1$ by $\sqrt{B_r}$ and equals exact identity $1.0$ for all $d \ge 2$ without task-shape branching, keeping the entire 1D support inside quadratic $W_2$ reach.
3. **Intra-Mode Barycenter Shrinkage Without Singleton Inflation:** Feydy diagonal-included Sinkhorn self-coupling (`rhbi_self_coupling = "feydy"`) cancels intra-mode entropic variance contraction when 2+ particles share a mode while allowing isolated batch singletons to absorb self-mass along the zero-cost diagonal without intra-mode covariance inflation.
4. **Dead-Zone Starvation & Geodesic Chord Conditioning:** Combining real/fake endpoint gradient caps with a deterministic Weyl-phased balanced-Sinkhorn Born-Infeld geodesic chord (`rhbi_chord_weight = 0.25`, `ebi_beta = 12.0`, `rhbi_gamma_gauge = 0.03`) enforces unit Lipschitz slope across inter-mode valleys without consuming training RNG streams or gating on input tensor shape.
5. **Constant-LR Post-Capture Chatter & Nonstationary Reacquisition:** Under unit-norm DualNorm updates, uncooled post-capture steps execute finite limit-cycle chatter that drops populated modes or violates stationary/frozen KS tolerances. A clock-free 5-channel Kantorovich-Hellinger two-sample residual tracker (unbiased Energy Distance, connectedness-weighted squared barycenter error, spread discrepancy, 75th-quantile fake-to-real precision halo excess, and 90th-quantile real-to-fake coverage excess) smoothly scales both critic and generator gradients down to $s_{\min} = 10^{-6}$ at equilibrium and reopens in 1 step on distribution shifts.
6. **Singleton Continuous-Conditioned Fibres:** On continuous-conditioned critic tuples $D(c, x)$ (`rhbi_conditional_fibre = "slack"`), singleton conditional fibres ($c_i \neq c_j$) route directly through fibre-centered paired relativistic log-odds $\operatorname{softplus}(D(c_i, y_i) - D(c_i, x_i))$, cancelling per-fibre condition baselines $b(c_i)$ identically without collapsing conditional variance onto $\mathbb{E}[y \mid c_i]$.

## Formulation Summary

- **AKT Scale-Equivariant Local Horizon ($\ell_t$):** Zero-sync detached EMA ($\beta_\ell = 0.1$) of $B_r^{\max(0,\,1/d - 1/2)} Q_{0.25}(d^{\text{NN}}_Y)$, with scale-equivariant tie resolution on discrete multi-atom codebooks.
- **Overflow-Safe Born-Infeld Ground Cost:** Rationalized $c_{\text{phys}}(x_i, y_j) = \sqrt{\ell_t^2 + \|x_i - y_j\|_2^2} - \ell_t$ and dimensionless screening cost $\hat{c}_{ij} = c_{\text{phys}}(x_i, y_j) / \ell_t$, $C^\infty$ at $x_i = y_j$ and finite up to `FLT_MAX`.
- **Slack-Screened Sinkhorn Operator (`rhbi-v4`):** Deterministic $K = 8$ `logsumexp` iterations on the $(B_f + 1) \times (B_r + 1)$ kernel with linear active energy $\hat{c}_{ij}$, critic-potentiated slack borders $(-\frac{\rho}{2} \pm \frac{\tau}{2}\tilde{d})/\varepsilon$ ($\rho = 3.0, \tau = 2.0, \varepsilon = 0.1$), slack-gain boost $1 + \gamma_{\text{slack}} s_f$ (`rhbi_slack_gain = 0.35`), interior slack reservoir marginal $S = 2$, Feydy diagonal self-coupling debiasing, shift-invariant slack-screened relativistic pairing on joint BiGAN hosts, and fibre-centered paired relativistic log-odds on continuous-conditioned fibres.
- **Weyl-Phased Geodesic Chord Penalty & Zero-Mean Gauge:** `RHBICriticPenalty` combines real/fake endpoint Born-Infeld caps ($\beta = 12.0, \lambda_{\text{floor}} = 0.005$) with a deterministic golden-ratio Weyl-phased balanced-Sinkhorn geodesic chord (`rhbi_chord_weight = 0.25`) and zero-mean logit gauge (`rhbi_gamma_gauge = 0.03`), skipping batch-doubling concatenation on stateful batch-bound critic view modules.
- **Impedance-Matched 5-Channel Thermal Controller & Complete Continuation:** Both `_RHBIObserverStepRecord` and `RHBIStepRecord` persist schema-2 state (`ell_ema`, `ell_initialized`, `chord_phase`, 5 residual EMAs, and `thermal_scale`) with exact bit-for-bit continuation across restarts and tamper validation.
- **Hybrid Exact-Polar / Quintic DualNorm Optimization:** Pairs exact smoothed polar SVD on dense 2D weights via CPU LAPACK (`optimizer_dense_profile = "polar_cpu_v1"`, `optimizer_smoothing = 0.0061`) with the batched quintic spectral filter on wide CUDA convolution slices (`optimizer_convolution_rule = "quintic"`).

## Implementation Sources

- `particlegan/rhbi.py`
- `particlegan/_rhbi/primitive.py`
- `particlegan/_rhbi/loss.py`
- `particlegan/_rhbi/critic.py`
- `particlegan/_rhbi/penalty.py`
- `configs/forge/ideas/rhbi-v1.json`
- `configs/forge/ideas/rhbi-v2.json`
- `configs/forge/ideas/rhbi-v3.json`
- `configs/forge/ideas/rhbi-v4.json`
- `configs/forge/studies/rhbi-v1-study.json`
- `configs/forge/studies/rhbi-v2-study.json`
- `configs/forge/studies/rhbi-v3-study.json`
- `configs/forge/studies/rhbi-v4-study.json`

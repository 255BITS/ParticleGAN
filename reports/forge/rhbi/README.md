# Relativistic Hellinger-Born-Infeld Transport (RHBI)

## Overview

**Relativistic Hellinger-Born-Infeld Transport (`rhbi`)** replaces unscreened adversarial generator objectives and separate transport-chord penalties with a single **Debiased Slack-Screened Unbalanced Sinkhorn** variational operator (`RHBILoss` / `RHBIGANLoss`) paired with a smooth $C^\infty$ Born-Infeld endpoint input-gradient cap (`RHBICriticPenalty`, $\beta = 20, \lambda_{\text{floor}} = 0.005$) and degree-of-freedom-aware SBNP particle prior regularization (`prior_reg = 0.02`, `prior_lr_mult = 2.9`) under strictly constant learning rates (`lr = 0.014`, `lr_floor = 1.0`, `network_lr_floor = 1.0`), zero EMA critic copies, and zero task-shape or step-count gates.

## Target Failure Modes

1. **Poisson Shot-Noise Floor ($B \lesssim M$) & Empty-Valley Bridging:** Linear Born-Infeld horizon screening ($\rho = 3.0$, `rhbi_rapidity_screening = False`) routes unmatched batch singletons and empty-valley midpoint particles ($\hat{c}_{ij} > \rho$) cleanly into the $(B+1)$-th Kantorovich-Hellinger slack reservoir rather than trapping them in 50/50 cross-mode active couplings.
2. **Intra-Mode Barycenter Shrinkage Without Singleton Inflation:** Feydy diagonal-included Sinkhorn self-coupling (`rhbi_self_coupling = "feydy"`) cancels intra-mode entropic variance contraction when 2+ particles share a mode while allowing isolated batch singletons to absorb self-mass along the zero-cost diagonal without intra-mode covariance inflation.
3. **Dead-Zone Starvation & Relativistic Impedance Matching:** Batch-centered, hyperbolic-clamped critic log-odds $\tilde{d}(z)$ lower the slack barrier with unit relativistic gain ($\tau / 2 = 1.0 = \kappa$), matching the asymptotic unit velocity cap $[-1, +1]$ of physical Born-Infeld ground costs while soft sub-unit endpoint damping ($\lambda_{\text{floor}} = 0.005$) suppresses post-capture critic slope chatter.
4. **Singleton Continuous-Conditioned Fibres:** On continuous-conditioned critic tuples $D(c, x)$ (`rhbi_conditional_fibre = "slack"`), singleton conditional fibres ($c_i \neq c_j$) route directly through fibre-centered paired relativistic log-odds $\operatorname{softplus}(D(c_i, y_i) - D(c_i, x_i))$, cancelling per-fibre condition baselines $b(c_i)$ identically without collapsing conditional variance onto $\mathbb{E}[y \mid c_i]$, while preserving bit-identical joint Sinkhorn transport on all unconditional and discrete-class tasks.

## Formulation Summary

- **Scale-Equivariant Local Horizon ($\ell_t$):** Zero-sync detached EMA ($\beta_\ell = 0.1$) of the 25th-quantile real nearest-neighbor distance $Q_{0.25}(d^{\text{NN}}_Y)$, with scale-equivariant tie resolution on discrete multi-atom codebooks.
- **Overflow-Safe Born-Infeld Ground Cost:** Rationalized $c_{\text{phys}}(x_i, y_j) = \sqrt{\ell_t^2 + \|x_i - y_j\|_2^2} - \ell_t$ (exact quadratic $W_2$ moments inside modes and unit velocity cap in the tail across all coordinate scales) and dimensionless screening cost $\hat{c}_{ij} = c_{\text{phys}}(x_i, y_j) / \ell_t$, $C^\infty$ at $x_i = y_j$ and finite up to `FLT_MAX`.
- **Slack-Screened Sinkhorn Operator (`rhbi-v3`):** Deterministic $K = 8$ `logsumexp` iterations on the $(B_f + 1) \times (B_r + 1)$ kernel with linear active energy $\hat{c}_{ij}$, critic-potentiated slack borders $(-\frac{\rho}{2} \pm \frac{\tau}{2}\tilde{d})/\varepsilon$ ($\rho = 3.0, \tau = 2.0, \varepsilon = 0.1$), interior slack reservoir marginal $S = 2$, Feydy diagonal self-coupling debiasing, shift-invariant slack-screened relativistic pairing on joint BiGAN hosts, and fibre-centered paired relativistic log-odds on continuous-conditioned fibres.
- **Complete Checkpoint Continuation & Stateful Penalty Safety:** Both `_RHBIObserverStepRecord` and `RHBIStepRecord` persist `ell_ema` and `ell_initialized` with exact bit-for-bit continuation across restarts, and `RHBICriticPenalty` automatically skips batch-doubling concatenation on stateful batch-bound critic view modules.
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
- `configs/forge/studies/rhbi-v1-study.json`
- `configs/forge/studies/rhbi-v2-study.json`
- `configs/forge/studies/rhbi-v3-study.json`

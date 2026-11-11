# Relativistic Hellinger-Born-Infeld Transport (RHBI)

## Overview

**Relativistic Hellinger-Born-Infeld Transport (`rhbi`)** replaces unscreened adversarial generator objectives and separate transport-chord penalties with a single **Debiased Slack-Screened Unbalanced Sinkhorn** variational operator (`RHBILoss` / `RHBIGANLoss`) paired with a smooth $C^\infty$ Born-Infeld endpoint input-gradient cap (`RHBICriticPenalty`, $\beta = 20$) and global moment-tether particle prior regularization under strictly constant learning rates (`lr_floor = 1.0`, `network_lr_floor = 1.0`), zero EMA critic copies, and zero task-shape or step-count gates.

## Target Failure Modes

1. **Poisson Shot-Noise Floor ($B \lesssim M$):** On high-mode mixtures where mini-batch draws leave modes unoccupied, unmatched batch singletons route smoothly into the $(B+1)$-th Kantorovich-Hellinger slack reservoir rather than hallucinating cross-mode transport jumps.
2. **Intra-Mode Barycenter Shrinkage:** Active-fraction-matched hollow self-coupling debiasing ($P^{\star(XX)}$ with masked diagonal $L^{(XX)}_{ii} = -\infty$) cancels entropic variance contraction inside occupied modes while routing singleton self-mass harmlessly to slack.
3. **Dead-Zone Starvation & Logit Runaway:** Batch-centered, hyperbolic-clamped critic log-odds $\tilde{d}(z)$ lower the slack barrier to tunnel surplus fake particles across empty valleys, while physical Born-Infeld ground costs cap active transport velocities inside $[-1, +1]$ to match the unit Lipschitz critic slope ($\tau / 2 = 1$).

## Formulation Summary

- **Scale-Equivariant Local Horizon ($\ell_t$):** Zero-sync detached EMA ($\beta_\ell = 0.1$) of the 25th-quantile real nearest-neighbor distance $Q_{0.25}(d^{\text{NN}}_Y)$, with scale-equivariant tie resolution on discrete multi-atom codebooks.
- **Overflow-Safe Born-Infeld Ground Cost:** Rationalized $c_{\text{phys}}(x_i, y_j) = \sqrt{\ell_t^2 + \|x_i - y_j\|_2^2} - \ell_t$ (exact quadratic $W_2$ moments inside modes and unit velocity cap in the tail across all coordinate scales) and dimensionless screening cost $\hat{c}_{ij} = c_{\text{phys}}(x_i, y_j) / \ell_t$, $C^\infty$ at $x_i = y_j$ and finite up to `FLT_MAX`.
- **Slack-Screened Sinkhorn Operator:** Deterministic $K = 8$ `logsumexp` iterations on the $(B_f + 1) \times (B_r + 1)$ kernel with rapidity-screened active energy $2\rho\log(1 + \hat{c}_{ij}/(2\rho))$ ($\rho = 2$), critic-potentiated slack borders $(-\frac{\rho}{2} \pm \frac{\tau}{2}\tilde{d})/\varepsilon$ ($\tau = 2, \varepsilon = 0.1$), interior slack reservoir marginal $S = 2$, and shift-invariant slack-screened relativistic pairing on joint BiGAN hosts.
- **Complete Checkpoint Continuation:** Both `_RHBIObserverStepRecord` and `RHBIStepRecord` persist `ell_ema` and `ell_initialized`, rejecting tampered or mismatched checkpoints with exact bit-for-bit continuation across restarts.
- **Hybrid Exact-Polar / Quintic DualNorm Optimization:** Pairs exact smoothed polar SVD on dense 2D weights via CPU LAPACK (`optimizer_dense_profile = "polar_cpu_v1"`, preserving exact rank-2 bottleneck singular-value lifting without cuSOLVER kernel latency) with the batched quintic spectral filter on wide CUDA convolution slices (`optimizer_convolution_rule = "quintic"`).

## Implementation Sources

- `particlegan/rhbi.py`
- `particlegan/_rhbi/primitive.py`
- `particlegan/_rhbi/loss.py`
- `particlegan/_rhbi/critic.py`
- `particlegan/_rhbi/penalty.py`
- `configs/forge/ideas/rhbi-v1.json`
- `configs/forge/studies/rhbi-v1-study.json`

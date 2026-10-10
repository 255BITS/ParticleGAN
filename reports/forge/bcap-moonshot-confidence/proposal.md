# Current-batch confidence transport

This is one fixed global research candidate compared with the exact selected
Phase 2 incumbent on a common frozen implementation. It is a diagnostic of the
original sixteen numerical questions, with no ordinary qualification or public
default claim. No seed, architecture, prior, sampling law, target law, batch
sequence, update allowance, schedule horizon or scoring cadence changes.

The selected control inherits `bcap-three-phase-incumbent-v1` with no overrides.
The complete compound candidate delta is `kinetic_transport_weight=1`,
`kinetic_transport_local_weight=1`, `transport_mobility_mode=block_mmd_v1`, and
`optimizer_svd_backend=cpu`. Projection and the finite critic guard remain
disabled. CPU full SVD is an explicitly declared numerical change, motivated
by the original 900-second word allowance. The candidate uses both unchanged
transport terms, gradients through G and learned locations, uniform prior
masses and original protected adversarial/paired/joint losses. This pair cannot
isolate the new confidence rule from transport activation and SVD rounding.

## Statistical rule and physical motivation

Finite minibatches fluctuate even when the population laws match. The empirical
transport objective can then keep driving motion. Scale that auxiliary force by
the part of the current discrepancy that exceeds its estimated sampling noise;
preserve the original game field so acquisition is not wholly gated.

Flatten only the existing G-phase output panels X and Y. All rows participate
in deterministic contiguous blocks, with B=min(max(2,ceil(n/32)),floor(n/2)).
Nearly equal block sizes handle a remainder without dropping rows. Let
ell²=mean_i ||Y_i-mean(Y)||², clamped at dtype epsilon, and
k(a,b)=exp(-||a-b||²/(2 ell²)). Targets, bandwidth, statistics and multiplier
are detached. For block b of m rows,

    h_b = sum_{i != j} [k(X_i,X_j)+k(Y_i,Y_j)
                       -k(X_i,Y_j)-k(Y_i,X_j)] / [m(m-1)]
    mu = mean_b h_b
    se = std_unbiased(h_b) / sqrt(B)
    alpha = max(0,mu-se) / [max(0,mu)+se+dtype_epsilon]
    L_G = L_original + alpha * (L_sliced_W2 + L_local_v2)

Fewer than four rows have insufficient block uncertainty information and return
alpha=0. Matched identical panels return zero; repeated systematic shifted
blocks approach alpha=1. Negative unbiased MMD estimates are suppressed. The
alpha branch does not consume RNG, compare critic history, draw more data,
consult evaluator geometry, tune to a task, or change the serving distribution.
Audit counters record calls, rows, blocks, zero calls, alpha>=.9 calls and sums of alpha/mu/se;
the next decision never depends on those counters. Active checkpoints validate
counter clocks and resume exactly; disabled packets omit the new default/state.

[Zaremba, Gretton and Blaschko's B-tests](https://arxiv.org/abs/1307.1954)
motivate averaging MMD statistics over blocks to trade compute and variability.
Their asymptotic test results do not calibrate this controller: real-adaptive
bandwidths, finite block counts, learned shared parameters and dependent fake
rows violate an automatic iid testing interpretation. The one-SE multiplier is
a declared deterministic heuristic, not a p-value or GAN-training guarantee.
The analogy to fluctuation/dissipation is selective mobility, with no physical
temperature, detailed-balance or population-convergence assertion.

## Existing negatives and material difference

The original round-five source is
`653c38045618ad240524237a9c141c8d06b28c03:reports/forge/bcap-physics/round5/README.md`.
Its CG4 package filtered output forces through a damped empirical J J^T solve
before DualNorm; Gaussian/native gates failed and solver residuals were large.
This candidate neither solves that system nor changes force direction.
Prior-only routing improved a width endpoint but lost rare density and lacked
the five-check suffix; here both G and locations keep the original pullback.
Balanced assignment repaired width while harming rare/native shape; this rule
does not change the existing coupling or equal sample masses. Strict finite
projection was not the sole Gaussian-retention cause; this candidate gates an
auxiliary current-batch force and supplies no finite-projection causal claim.

The archived small-batch study at
`da3b0470918fc0045175441ce24e69777b5d5990:reports/toy100/lrfree-search/streaming-smallbatch-toy/RESULTS.md`
found moving-critic controllers harmful; its global guard mainly suppressed
actions and did not establish row-level direction. Our statistic uses output
geometry on the current panels, no payoff history, memory, extrapolation,
cloning, birth/death, mass adaptation or critic feature gauge. Global confidence
still cannot tell which local transport directions are reliable. Rare modes
and local width errors can be invisible to its coarse RBF scale, and suppressing
useful transport can leave the incumbent's adversarial drift untouched.

Output-marginal agreement cannot establish conditional correspondence: a
row-permuted target gives zero transport while violating identity. Trajectory,
residual and word retain all original paired/joint objectives and identity
gates; no marginal PASS is substituted for those checks.

## Frozen prediction and failure criteria

The candidate predicts Gaussian final KS <= .05 and complete Gaussian
stationary/deadline/shifted-hold gates, improved rare/broad shape, and preservation
of the original six Tier 1 passes. The registered numerical final signature is
Gaussian stability KS <= .05; KS > .05 is its formal falsifier. A passing
endpoint cannot certify the claim: any required stationary, deadline, hold or
terminal-suffix failure rejects the corresponding repair. The exact original
metrics and gates remain authoritative, including native precision .97 and
word reconstruction. Full failures or blocked dependencies stop this revision;
there is no post-result adjustment, extra arm, scientific retry or seed study.

Both own Gaussian and word producers must finish their full original update
counts and PASS before their hold runs. Failed/timeout producers leave holds
BLOCKED. All independent tasks in the registered diagnostic roster can finish.
The pair reserves 45,840 seconds (22,920/arm), with a 48,000-second paid ceiling
and a separate 300-second software allowance. Eleven original Tier 2 tasks are
outside the scope. Admission must be READY and all source/declarations committed
before either paid request is submitted.

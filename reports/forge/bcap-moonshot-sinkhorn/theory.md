# Frozen Sinkhorn research hypothesis

The selected Phase 2 incumbent passed all six original Tier 1 tasks. This pair
inherits its exact global recipe. The candidate enables only weight-one global
Sinkhorn transport and the explicit CPU full-SVD backend. Local transport,
direction projection and finite critic guarding remain disabled. CPU rounding is
a numerical trainer change, so this pair tests the complete compound package.

For equal empirical masses, define cost
`Cij = mean_d((xi-yj)^2) / (2 s)`, where `s` is the detached mean coordinate
variance of the complete original real panel, floored at machine epsilon.
The cross and both self terms share this one normalizer. With epsilon .1,
`Fε = minπ <π,C> + ε KL(π | a⊗b)` is entropically regularized free energy.
Use `Sε = Fε(x,y) - Fε(x,x)/2 - Fε(y,y)/2`. The fake self term differentiates
both occurrences of x; real targets and the normalizer are detached.

[Feydy et al.](https://arxiv.org/html/1810.08278v1), equations 1, 3 and 8,
derive the regularized cost, debiasing and dual formulation. Their exact
divergence removes the support-shrinking bias of cross-only entropic transport.
[Cuturi](https://arxiv.org/abs/1306.0895) provides the original entropic matrix
scaling approach. These sources motivate the hypothesis; their converged
positivity and population guarantees do not apply automatically to this finite
training implementation.

The implementation starts dimensionless dual f and g at zero. Each of exactly
24 rounds computes `Tg = -logsumexp_j(-C/ε + gj + log bj)` and its transpose
counterpart `Tf`, simultaneously, then uses `(f+Tg)/2, (g+Tf)/2`. Evaluate the
complete finite dual `ε(mean f + mean g - sum exp(f⊕g-C/ε+log a+log b) + 1)`.
Autograd differentiates all 24 rounds and the mass correction. There is no
detached-plan envelope derivative, warm start, stochastic solver, convergence
retry, extra panel or target oracle. Damped simultaneous mappings preserve the
symmetric finite solver; the real-owned scale makes the full normalized loss
asymmetric when swapping clouds with different variances. Negative finite losses
are retained, counted and reported instead of being silently clamped.

Every original row remains in the original adversarial and penalty computation.
Only the auxiliary term takes at most 128 evenly spaced existing rows, including
the endpoints: index k is `floor(k*(n-1)/127)` for n>128. Smaller panels use every
row. Auxiliary empirical weights are uniform. No data, prior or evaluation draw
is added, and the original public sampling law and learned prior are unchanged.
Call, input/auxiliary row, total iteration and negative-loss counters plus the
worst relative marginal residual are checkpointed. No dual solver state persists.

The original sustained width gate is the main falsifier: final covariance above
.85 or terminal passing suffix below five rejects repair, even if some widths
improve. Loss of any original incumbent Tier 1 gate is a regression. Native
precision, genuine-quality occupancy and spill remain independent original
gates. Finite-batch uncertainty, unresolved marginal residual, shared-network
gradient coupling and the global scale can defeat this hypothesis.

The earlier round-five balanced assignment (`653c38045618ad240524237a9c141c8d06b28c03`)
uses detached bijections in contiguous 128-row blocks, preserves every row once,
and repairs width while regressing rare/native density. This candidate has no
assignment, quantile matching, hard coupling or local-v2 objective; its self
costs supply a different spread force. It does not claim that entropic bias
caused the earlier unregularized assignment failure. The previous four-iteration
empirical output-kernel filter and prior-only routing remain separate negatives.

This single seed-zero pair measures all 16 original questions in the registered
diagnostic scope, with own fully passing Gaussian/word producers required for
holds. Reservations are 22,920 seconds per arm, 45,840 paired, within the 48,000
paid ceiling. Separate targeted software checks have 300 seconds. No scientific
retry, seed repeat, tuning or ordinary qualification follows.

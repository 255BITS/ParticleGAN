# Distinct anchors: a fixed-target coverage principle

The next useful test is whether an explicit allocation of distinct generated
particles to data-derived groups removes the wrong-subset attractor **before
testing a neural optimizer**. A lower GAN loss, a smaller G step, or a vector
with positive projection toward an empty mode is insufficient: the ideal
ratio diagnostic already shows that within-mode centering can satisfy the
last condition. The existing exact whole-map `C+Q` experiment also fails
cold support recovery, so its failure is not solely nonlinear pullback error.

For the proposed objective below, the cheapest falsifiers are: unstable
group recovery from halves of the same real batch; no nonlocal donor for
an uncovered group; nonzero field on a covered fixed cloud; failure to
restore a small displacement on that same cloud; or failure of the
subsequent actual neural update to decrease the declared objective. None
of these checks uses known host means to build the update. They are prior
to any full warm/cold GAN run.

## Objective and exact output-space result

Let `c_1,...,c_K` be distinct centers inferred from a fixed real-data bank,
and let `y_1,...,y_N` be free generated output vectors, with **N>K**. Let
`A` be all injective maps from the K centers to N distinct generated rows.
The proposed objective has the two unit-weight terms

```
L(Y) = min_(a in A) (1/K) sum_k ||y_a(k) - c_k||²
       + (1/N) sum_j min_b ||y_j - c_b||².
```

This is a support-coverage objective, not a probability divergence. Group
frequencies are not matched; extra particles may occupy any covered group.
That is deliberate and must remain explicit. Both normalizations are part
of the declared rule, not a gain selected after observing success.

**Lemma.** With fixed distinct centers and N>K, every local minimum of
`L` has `L=0`. In addition, any active quadratic branch with zero gradient
has `L=0`.

To see this, introduce a nearest-center label `b_j` for every particle.
Then `L` is the finite minimum of quadratic functions `q_(a,b)`. Every
branch has a positive-definite Hessian because every row appears in the
second term. If `Y*` locally minimizes `L`, every branch active at `Y*`
also locally minimizes its own quadratic: in a neighborhood,

```
q_(a,b)(Y) >= L(Y) >= L(Y*) = q_(a,b)(Y*).
```

Thus `Y*` is that branch's unique stationary point. For a particle assigned
as the anchor of center `c_a` and carrying nearest label `b`, stationarity
requires

```
y_j = (N c_a + K c_b) / (N + K).
```

If `a != b`, the distance of this point to `c_a` is
`K ||c_a-c_b||/(N+K)`, strictly smaller than its distance
`N ||c_a-c_b||/(N+K)` to `c_b`. That contradicts the active nearest-center
label. Hence each anchored particle is exactly at its assigned center.
Every unanchored particle is at its nearest center by its own stationarity
equation. Injectivity supplies an anchor for every group, so `L=0`.
The same calculation proves the active-branch statement without assuming
that `L` is differentiable at the point.

**Exact update.** Choose a minimizing injective assignment and current
nearest labels. Minimize that one active quadratic exactly: anchored rows
move to the weighted point above; other rows move to their nearest center.
This is a majorization-minimization update, with no Euler gain. The active
quadratic equals `L` at the old point and upper-bounds it everywhere. At
any positive-loss point it cannot already be stationary, by the lemma, so
its minimizer strictly decreases `L`. There are finitely many branches and
each branch has one fixed minimizer. Consequently, in exact free-output
arithmetic, repeated updates reach `L=0` after finitely many strict
decreases. This argument does not provide a useful small bound on their
number, and it does not describe a stochastic or neural training loop.

This is our direct finite-dimensional derivation for the declared
objective, not a theorem claimed by the drifting or GAN papers below.
Independent mathematical review found no gap under the stated premises.

## Exact bounded check and a necessary scope limit

The [rational-arithmetic check](anchor_coverage_geometry.py) uses N=3,
K=2 and fixed centers `{-1,+1}`. It enumerates all 48 combinations of
injective assignments and nearest labels, without optimization tolerance
or randomness. Twelve quadratic stationary points have active branch
labels; all twelve have objective zero. Starting with every particle at
`-1`, exact updates give

```
(-1,-1,-1), L=2
(-1, 1/5,-1), L=8/15
(-1, 1,-1), L=0.
```

A covered cloud rests exactly. Moving one redundant particle by `1/10`
produces positive loss `1/300`; the same update restores the covered cloud.
Thus this tiny fixed-target model has both exact rest and recovery without
requiring a positive motion floor or a clock-dependent step size.

Strict surplus is substantive for the simple active-branch algorithm.
With N=K=2, centers `{-1,+1}` and particles `(-1,0)`, one active branch
has zero gradient while `L=1`: the second particle ties between centers,
and selecting `-1` as its nearest label cancels its assigned-anchor force.
The point is not a local minimum, since moving it toward `+1` lowers `L`,
but deterministic first-index tie selection can stall there. This
counterexample is retained rather than extending the algorithm's scope.
K>N has no injective assignment at all.

Three focused tests pass in 0.04 seconds. The
[receipt](continuous-evidence/anchor-coverage-geometry/result.json) and
[manifest](continuous-evidence/anchor-coverage-geometry/manifest.json)
retain every branch, the counterexample, source hashes and test output.
This is `shared_gate_eligible=False`; no host training was run here.

## Why more accurate critic response alone is not the selected principle

The one-step total-gradient experiment is locally differentiated correctly,
but its ensuing saved-state filter failed. Merely increasing its unroll
depth has no new causal support. Even ideal unrestricted local density-ratio
guidance can restore a cluster into an already occupied mode, as documented
in the separate ideal-ratio three-mode diagnosis. Correct opponent tracking
and correct support allocation are different requirements.

An exact convex readout response would be a useful diagnostic, not an
automatic cure. With fixed hidden features `h(x)` and `D_w(x)=wᵀh(x)+b`,
the Rp logistic term is convex in `w`. So is `b_cap`: each input gradient
is linear in `w`, and squared positive excess of its Euclidean norm is a
convex function. The bias cancels. However, a finite unique minimum and
invertible response Hessian do not follow. For example, fixed clipped
features may be `+1` on every real sample and `-1` on every fake sample,
with zero input slopes at all samples. Then the entire empirical loss is
`softplus(-2w)`, the cap is zero, and the infimum is attained only as
`w -> infinity`. A pseudoinverse does not repair this missing optimum.
Any future readout-response test must certify its own residual and
identified curvature; adding a zero-centered ridge silently would change
both the objective and the user's requested mechanism.

## Recent primary research and limits

The August 2, 2026 preprint
[Wasserstein gradient flows of MMD with energy kernels](https://arxiv.org/abs/2608.01182)
is especially relevant to the distinction between particle and continuum
claims. It proves finite-particle convergence to a critical set and
constructs collision-free saddle equilibria; its stronger target-matching
statements concern absolutely continuous continuum laws under additional
bounds. It does not promise that twelve deterministic particles find a
global empirical optimum. This argues against declaring generic energy
or MMD descent a coverage solution without a finite-cloud test.

[(De)-regularized MMD gradient flow, JMLR 2025](https://www.jmlr.org/papers/v26/24-1574.html)
uses a covariance-adjusted kernel and an adaptive de-regularization
schedule to connect MMD flow with chi-square geometry. Its favorable
population results do not establish fixed uniform-particle HQ stability
or authorize replacing this experiment with another regularization
schedule. No DrMMD training arm was launched.

[Gradient Flow Drifting, March 2026](https://arxiv.org/html/2603.10592v1)
connects Gaussian-kernel drift with KDE score differences and discusses
mixed-divergence fields. Its distribution-level identifiability is not
equivalent to a field being zero at twelve generated support points.
The existing ideal-ratio and failed Sinkhorn diagnostics make that scope
distinction important; a local score can be uninformative about missing
groups despite a large distribution discrepancy.

[Gauss–Newton drifting, September 15, 2026](https://arxiv.org/pdf/2609.17167)
relates fitting an output drift by a linearized least-squares solve to a
natural-gradient metric. That supports separating the choice of an output
field from its neural realization. Its convergence result concerns an
implicit proximal scheme, a closed convex function class, increasing
network capacity and vanishing optimization error. Our fixed finite MLP,
nonconvex min-of-quadratics objective and bounded approximate pullback do
not satisfy that theorem. It is motivation for measuring actual nonlinear
landing error, not a convergence certificate.

## Remaining falsifiers before a training claim

The proved centers are fixed and distinct. A minibatch clustering rule
must establish that its inferred groups are reproducible rather than
outliers or arbitrary partitions. A covered empirical cloud at one batch's
centroids can still move under the next batch. The theorem says nothing
about component frequencies, overlapping distributions, conditioning,
or the probability of rare noisy batches over indefinite training.

For the neural pullback, record the full G+prior output Jacobian's relevant
rank, the best linearized target residual, and actual nonlinear decrease.
The earlier prior-only correction can miss a distant target despite local
rank two. A failed neural step does not refute the free-output lemma, while
a successful free-output step does not validate the neural algorithm.
Finally, adding the original adversarial update can push away from the
fixed support objective; only the actual combined host, with unchanged
quality gates and constant-rate/noise receipts, can establish acquisition
and sustained fixed-target quality.

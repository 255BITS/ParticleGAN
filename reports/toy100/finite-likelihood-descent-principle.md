# A finite objective with a response that does not decay with training age

For fixed positive target quadrature weights `a_k` summing to one, fixed
variance `v>0`, and `N` equally weighted Gaussian atoms at `y_j`, define

```
J(Y) = -sum_k a_k log[ (1/N) sum_j Normal(x_k; y_j, v I) ].
```

This finite weighted likelihood is the declared acceptance objective. Calling
it an approximation to a continuously blurred-target cross-entropy does not
make it an exact continuous-KL value. A finer numerical integration audit
remains separate. In particular, all intermediate steps of a cheaper proposal
solver need not decrease `J`; only the final applied state must do so.

For the current cloud, let `r_kj` be the normalized Gaussian responsibilities,
`m_j=sum_k a_k r_kj`, and `y'_j=sum_k a_k r_kj x_k / m_j`. All `m_j` are
strictly positive in exact arithmetic. The standard Jensen majorizer gives
the quantitative EM decrease

```
J(Y) - J(Y') >= sum_j m_j ||y'_j-y_j||² / (2v).
```

To see this, keep the old responsibilities fixed in the negative-log-mixture
upper bound. The bound touches `J` at the old cloud. Minimizing its independent
weighted quadratics gives the centroids above and reduces that bound by the
displayed amount. The new actual likelihood is no larger than its bound.
Thus a nonzero centroid displacement produces strict decrease; this response
does not contain the training clock, number of previous updates, or an LR
schedule. A cheaper proposed cloud can be accepted if it lowers `J`, with an
EM step on the actual acceptance quadrature as fallback when it does not.
Fallback must include equal-cost/unchanged cheap proposals: otherwise a coarse
solver could stop even though the acceptance objective still has a signal.

Every exact EM centroid lies in the convex hull of the target rows. For a
fixed finite dataset and fixed quadrature, full EM updates therefore remain
bounded. Also `J>=d/2 log(2πv)`, since a normalized equal-variance mixture has
density at most `(2πv)^(-d/2)`. Along a fixed-objective sequence consisting only
of descending accepted proposals and EM fallbacks, the objective values
converge and the sum of the displayed EM-decrease lower bounds is finite.

These are useful stability principles, with explicit boundaries:

- They do not prove the global optimum is reached. An EM fixed point may
  still lack a mode; observed-data donor proposals supply a separate global
  allocation mechanism that needs testing.
- They do not prove an HQ invariant or correct mixture masses and spread.
- Cumulative arrival of real samples changes the objective between updates.
  The running estimator has no count multiplier on model correction, but a
  streaming convergence argument requires additional assumptions.
- Approximate neural fitting must check the actual acceptance objective after
  realization. Small target-fit error alone is not a descent certificate.
- Bounding output targets does not bound redundant neural parameters. A fixed
  affine chart addresses that separate issue for the production affine model;
  nonlinear joint minimum-increment fitting has no such global theorem.
- Floating-point underflow, stopping tolerances and quadrature error remain
  numerical limits. The formula is an exact finite-objective statement, not an
  interval-arithmetic certificate for the implementation.

The [old quadrature filter](round8-forward-kl-cumulative.md) remains failed
under its original per-inner-step audit. A new acceptance/fallback algorithm
must earn fresh results. The [higher-order diagnosis](round8-forward-kl-quadrature-audit.md)
finds that the two affected whole updates still decrease the more accurately
evaluated cross-entropy, which motivates this final-state acceptance rule.

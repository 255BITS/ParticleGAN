# What the likelihood route can establish

The new free-output experiment evaluates the emitted Gaussian mixture rather
than only its clean centers. It smooths both the empirical target and emitted
law by the same fixed Gaussian, and lowers a positive finite-quadrature
approximation of forward KL. Global observed-data replacements address cold
allocation; fixed-weight EM refines locations. It has no training-age-dependent
correction multiplier. Accumulating data changes the distribution estimate,
not the ability to correct a model error against that estimate.

This direction is consistent with the July2026 revision of
[Zhu's inclusive-KL gradient-flow paper](https://arxiv.org/abs/2411.00214v2),
which relates MMD flows to smoothed inclusive-KL forces and develops
Fisher–Rao and local-estimator alternatives. Its continuum and varying-mass
results are not convergence theorems for twelve fixed-weight Gaussian atoms,
finite quadrature, or a neural realization. In particular, the observed MMD
mode loss cannot be dismissed by importing a distribution-space theorem.

[Bing, Kong and Li (July2026)](https://doi.org/10.1093/biomet/asag047) analyze
multi-component Gaussian-mixture EM under component-separation conditions.
Our model fixes equal atom weights and a narrower common covariance, so the
target distribution need not belong to that model family. Their recovery
result cannot certify this misspecified particle model. The
[2025 overspecified-EM analysis](https://link.springer.com/article/10.1007/s00362-025-01749-z)
also concerns a particular balanced two-component fit to a single Gaussian,
with local assumptions. These papers support checking the actual likelihood
and model assumptions; none removes the cold, omitted-bank, continued-quality,
or neural-fit gates.

For the finite positive quadrature actually optimized, one elementary bound
explains why likelihood can penalize missing support strongly. Let
`q(x)=C/N * sum_i exp(-||x-y_i||²/(2v))`, where `C=(2πv)^(-d/2)`.
If a set of target quadrature rows with total weight `w` lies at least distance
`r` from every atom, then

```
cross_entropy >= -log(C) + w*r²/(2v).
```

This follows from `q(x)<=C*exp(-r²/(2v))` on that set and `q(x)<=C`
elsewhere. Consequently, a sublevel set below this threshold excludes that
particular missing-support configuration. The bound is conditional on a
fixed target, positive variance and the stated distances. It is not an
eight-mode/HQ guarantee, does not imply correct masses or within-mode spread,
and must not use true mode centers inside training. As new data arrive the
objective changes, so descent within each update alone is not a stationary
streaming convergence proof.

EM monotonically lowers the declared finite weighted likelihood, but local
stationarity can be poor. The global replacement stage is finite-budget and
does not certify a global optimum. Agreement in objective-decrease signs
between5×5 and9×9 Gauss–Hermite rules is useful numerical evidence, not a
bound on integration error; their endpoint gradients already differ visibly
in the cold one-bank case. These distinctions remain part of qualification.

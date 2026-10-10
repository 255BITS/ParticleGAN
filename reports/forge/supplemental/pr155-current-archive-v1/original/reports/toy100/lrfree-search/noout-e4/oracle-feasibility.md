# Feasibility check before replacing E4's row test

This calculation uses no benchmark stream, model samples, critic, saved native
gradient, or gate score. It asks what an **oracle** row-mean test could detect
with 20,000 simultaneously tested rows and at most about 700 touches per row.
The oracle knows the gradient direction, covariance, and AR(1) correlation;
any implementable test that estimates them has less information. The generic
signal counts are one, 1% of rows, and 5% of rows. They are scenarios, not
estimates of native strays.

For `m` touches with unit marginal variance and AR(1) correlation `ρ`, the
information about a constant one-direction mean is
`I = (m(1−ρ)+2ρ)/(1+ρ)`. At per-row level `α`, the smallest standardized mean
shift with 90% power is
`(z_(1−α)+z_.90)/sqrt(I)`. The Benjamini–Yekutieli `k`th rejection level is
`α=q k/(N H_N)`, where `q=.05`, `N=20000`, and `H_N=10.4807`. Treating the
other `k−1` signals as already ranked above the row makes this optimistic.

| Correlation | Touches | One signal | 1% signals | 5% signals |
|---:|---:|---:|---:|---:|
| 0 | 128 | .56 | .46 | .42 |
| .5 | 128 | .96 | .79 | .73 |
| .9 | 128 | 2.28 | 1.87 | 1.72 |
| .9 | 700 | 1.03 | .84 | .78 |

Values are minimum mean shifts in units of the gradient's marginal standard
deviation. The complete grid, including Benjamini–Hochberg comparison levels,
is in [oracle-feasibility.json](oracle-feasibility.json); the calculation is
reproducible with [oracle-feasibility.py](oracle-feasibility.py).

Under strong correlation, a 128-touch disjoint-block test has little hope of
detecting weak persistent gradients at the required multiplicity threshold.
The originally proposed empirical-tail calibration is also impractical at its
first-rejection level of about `2.39e−7`. This rules out that *specific*
prospective detector plan as a useful next GPU experiment. It does not prove
that no gradient-only rate mechanism can work. A new mechanism needs a
separate, prospective argument and synthetic controls; E4's original `W=50`
result remains diagnostic.

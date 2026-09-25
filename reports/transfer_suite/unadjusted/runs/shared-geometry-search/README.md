# Shared-cap6 nonperiodic geometry discriminator study

**No new sustained live PASS:** 0 of 26 rare-component episodes. These are
incomplete D-only architecture screens of `vector_unequal_mass`; they cannot
change the 18/19 overall score. Every failed episode and its 24 live and EMA
observations are retained.

All cards use the unchanged `shared_c6` recipe: Rp logistic, b_cap coefficient
6, κ 1.25, prior spread .05, Adam `(0,.99)`, G/D learning rates .00425 and
particle rate .0085, held for 60% then cosine to 5%. The original G and
particle initialization, 256 particles, batch 128, 1200 steps, seed 0, data,
objectives and thresholds are unchanged. Only the pointwise discriminator
architecture varies, declared in each `discriminator_variant` card. No map is
fitted to target samples or benchmark metrics, and no feature center grid is
placed at target modes.

| Stage | Cards | Live passes | Closest final result | Evidence |
| --- | ---: | ---: | --- | --- |
| Nonperiodic coordinate maps, gates and activations | 15 | 0 | Random smooth ridges: min eigen ratio .11824 | [Screen](screen/README.md) |
| Initially-zero feature pathways | 9 | 0 | Quadratic score head: ratio .03174 | [Residual](residual/README.md) |
| Slowly saturating score readouts | 2 | 0 | Rational readout: ratio .00237 | [Readout](readout/README.md) |

The best overall card here, `geometry_ridgehinge2_lnsp96`, uses 48 fixed
random affine hyperplanes with Softplus sharpness 2 plus raw coordinates, then
a width-96, three-hidden-layer LayerNorm Softplus β4 critic. Its final mixture
mass error is .01667, high-quality fraction .94751, covariance error .72541,
and min normalized component eigenvalue ratio **.11824** against the required
**.15**. Its final five variance ratios are `.037, .067, .113, .099, .118`,
so it never meets the five-check sustained gate. The previous raw-input LN β4
lead reached .12525 on the last check; this new card does not surpass it.

Direct polynomial, bounded coordinate and sharper random ridge inputs often
lost a component. The nine follow-up variants preserve the previous LN β4
critic's initial weights, score and RNG state exactly, then learn zero-initialized
feature injections, score heads or activation corrections. They still contract
within-mode spread. Smooth asinh and rational score readouts were tested after
an independent study had already tried tanh readouts, avoiding duplicate runs.
These bounded failures support no claim about other geometry architectures or
future optimizer recipes.

Each stage contains its frozen plan, protocol, `source.tar.gz`, `index.json`,
individual compressed episodes and complete run log. The source archive is the
exact source used for that stage. [Validation](validation.json) recomputes every
verdict and checks source archive contents, per-episode hashes, unchanged D-only
specs, the common recipe, actual optimizer groups, 1200 G and D updates, and
all 24 observations. It found no training errors; EMA also passed zero cards.

Reproduce a stage with its source archive and plan; use a fresh output folder:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m \
  benchmarks.transfer_suite.shared_geometry_search \
  --plan benchmarks/transfer_suite/plans/shared_geometry_rare_screen.json \
  --output /tmp/shared-geometry-screen-replay
```

For the other stages, replace the module with
`shared_geometry_residual_search` or `shared_geometry_readout_search` and the
corresponding plan file. Three focused architecture invariant tests, plus the
existing research tests, pass (14 total tests).

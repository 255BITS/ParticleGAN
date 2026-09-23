# Fixed-recipe discriminator architecture study

This bounded study changes only the pointwise discriminator architecture under
`shared_c6`: Rp logistic, gradient cap coefficient 6 and kappa 1.25, prior spread
weight .05, no particle L2, Adam `(0, .99)`, absolute G/D LR .00425 and prior LR
.0085, cosine beginning at 60% and ending at 5%. Generator, samples, particle
count, batch, initialization seed 0, targets, thresholds and original budgets
remain fixed. No normalization depends on other examples in the batch.

The hypothesis is that reducing a shared periodic bottleneck or adding generic
smooth coordinate interactions may improve the critic's local spread feedback.
This is an architectural hypothesis, not an established causal conclusion for
the shared high-rate recipe. Fixed feature, branch, residual and score scales are
explicit architecture parameters; they change the model's parameterization even
though the optimizer's rates remain identical.

| Architecture | D parameters | Declared representation |
| --- | ---: | --- |
| raw_softplus96_l3 | 19009 | Raw coordinates; Softplus beta5; 96 x 3 |
| raw_silu128_l3 | 33537 | Raw coordinates; SiLU; 128 x 3 |
| quadratic_softplus96_l2 | 9985 | Raw plus .25-scaled x², xy, y²; Softplus beta5; 96 x 2 |
| quadratic_tanh96_l3 | 19297 | Same quadratic features; Tanh; 96 x 3 |
| residual_raw_softplus96_l3 | 19009 | Raw; smooth residual layers scaled by 1/sqrt(2) |
| residual_lowfreq_softplus96_l3 | 19393 | Raw plus one pi/4 axis harmonic; smooth residual layers |
| halfscore_fourier_skip96_l2 | 10467 | Two original harmonics; half-scaled Softplus score plus zero-initialized raw linear skip |
| additive_raw_fourier64_l2 | 5796 | Separate raw 64 x 2 and harmonic 32 x 2 branches; harmonic score scaled .25; raw linear skip |

All layers use ordinary PyTorch linear initialization; extra raw skip heads start
at zero. There are no data-derived centers, labels, frequencies or feature losses.
The generator initializes before the discriminator in the unchanged vector host.

The frozen initial screen runs all eight cards on unequal mass and unequal
width. At most four cards advance to anisotropic and overlapping distributions;
at most two complete the remaining broad and spiral cases. Every attempt stays
visible. A live pass requires all 24 observations and at least five passing
observations at the end. EMA has separate results and cannot determine selection.
Architecture support may be selected separately for each task; this does not
make an incomplete screen a full six-data profile or a 19/19 result.

Create a JSON plan with exactly `purpose`, `architectures` (names from the table)
and `tasks` (canonical vector task names), then run from the repository root:

```sh
python -u -m benchmarks.transfer_suite.shared_discriminator_search \
  --plan /tmp/architecture-plan.json --output /tmp/architecture-run \
  > /tmp/architecture-run.log 2>&1
```

The output directory must not already exist. It contains the frozen cards and
recipe, original and effective specs, exact source archive/hashes, optimizer
receipts, complete live/EMA curves and action traces, every failed trial, and an
incrementally updated readable matrix. `original_spec` always remains the
canonical reference host; `discriminator_variant` exposes the architecture change.

The runner depends on the shared discriminator-only validator
`shared_variants.py`. Importing a result into the primary leaderboard must verify
source bytes, recipe identity, the discriminator-only diff and actual optimizer
receipts; importing never substitutes recipe parameters by toy.

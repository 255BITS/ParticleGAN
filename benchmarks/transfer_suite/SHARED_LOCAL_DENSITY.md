# Shared-cap6 local-density discriminator study

This architecture study holds the shared recipe fixed: Rp logistic, b_cap6,
kappa1.25, prior spread .05, no particle L2, Adam `(0, .99)`, G/D rates .00425,
prior rate .0085, and the original delayed cosine schedule. Only the pointwise
discriminator changes. Generator, seed0, particles256, batch128, original data,
1200-step budgets, 24 observations, thresholds and required final passing suffix5
remain unchanged. EMA is reported separately.

The hypothesis is that generic local curvature features can represent within-mode
variance without sharing a periodic feature bottleneck. These are model
parameterizations; they introduce no feature matching, moment supervision,
target centers, extra loss, clipping, gradient scaling, normalization or optimizer
changes. Success or failure under this recipe does not establish a universal
discriminator architecture.

| Architecture | D parameters | Representation |
| --- | ---: | --- |
| local_rbf64_direct | 67 | Fixed Gaussian radial bank64 plus raw coordinates; signed linear head |
| local_rbf128_direct | 131 | Same, bank128 |
| local_rbf256_direct | 259 | Same, bank256 |
| local_rbf128_adaptive | 515 | Gaussian bank128 with learned centers and positive widths |
| local_cauchy128_direct | 131 | Fixed inverse-quadratic radial bank128 |
| local_rbf128_softplus64 | 12609 | Gaussian bank128 plus raw coordinates; Softplus beta5 MLP64x2 |
| local_quad16_width1 | 99 | Pointwise softmax-gated quadratic experts16, width1 |
| local_quad32_width1 | 195 | Same, experts32 |
| local_quad32_width05 | 195 | Experts32, width.5 |
| local_quad32_adaptive | 291 | Experts32, learned centers/positive widths |
| local_product_silu64_l2 | 8769 | Products of independently projected SiLU features, 64x2 |
| local_product_silu96_l2 | 19297 | Same, 96x2 |
| local_product_silu128_l2 | 33921 | Same, 128x2 |
| local_squared_silu64_l3 | 8577 | Squared SiLU activations, 64x3 |
| local_squared_silu96_l3 | 19009 | Same, 96x3 |
| local_product_softplus96_l2 | 19297 | Products of Softplus beta5 features, 96x2 |

Every radial/expert center comes from the same generic `N(0, 2² I)` local seed0
stream, never from target examples. Radial widths cycle through [.25,.5,1,2].
Trainable widths use `exp(log_width)` without extra constraints. Radial heads and
MLPs use ordinary PyTorch linear initialization. Local quadratic coefficients
start at zero with an ordinary raw linear head. Local coordinates are displacement
divided by width; the six expert features are 1,dx,dy,dx²,dx*dy,dy². Expert gating
normalizes across features for each individual sample, never across samples.

The frozen first round tests all16 cards on unequal mass and unequal width.
No per-task recipe is permitted. All failures are retained, including final-bound
passes that do not sustain five checks. The registry wrapper reuses the audited
shared-D runner unchanged, including actual optimizer receipts, architecture-only
validation, source snapshots, full live/EMA curves and action traces.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u \
  -m benchmarks.transfer_suite.shared_local_density_search \
  --plan /tmp/local-density-plan.json --output /tmp/local-density-run \
  > /tmp/local-density-run.log 2>&1
```

The JSON plan contains `purpose`, `architectures` (names above), and `tasks`
(`vector_unequal_mass`, `vector_unequal_width`). The output directory must be new.
Each episode keeps its canonical `original_spec` plus explicit
`discriminator_variant`, and its resolved recipe is always `shared_c6`.
No result from another formulation or another optimizer can fill this row.

The initial32 episodes produced no sustained pass. The approved bounded followup
therefore keeps a full raw-coordinate MLP and adds a zero-initialized local
quadratic correction. The six cards combine each existing raw SiLU128x3 and raw
Softplus96x3 base with32 experts of width1,64 experts of width1, or32 experts of
width.5. The centers again use generic `N(0, 2² I)`, fixed local seed0. Only the six
quadratic coefficients per expert are added as trainable parameters. Gaussian
softmax gates are fixed pointwise functions; the standard cap sees the summed
score and there is no additional objective.

The main MLP is constructed first. Zero branch coefficients and a local buffer
initialization preserve its initial weights, score and global RNG state exactly;
tests verify all three and the cap gradient into the correction. This separates
local curvature capacity from replacing the MLP's existing mode feedback. D
parameter counts are33729/33921 for the SiLU base and19201/19393 for the Softplus
base. No initializer or seed is swept.

Run the second frozen plan through
`benchmarks.transfer_suite.shared_residual_curvature_search` with the same CLI
arguments. Its source archive includes the original round unchanged, plus the
two new refinement modules. Every initial failure remains in its original
artifact; the followup has a separate plan, source manifest and episode folder.

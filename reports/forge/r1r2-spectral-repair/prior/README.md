# Prior table versus generator noise geometry

**No distinct prior-update change has concrete support in these endpoints.**
Excess tangent variance already appears in the generator's transformation of
fixed within-component noise. The endpoint prior tables have similar global
spread; that spread alone cannot explain their different output shapes.
This is a physical diagnosis, not a new recipe proposal or training result.

The [source-bound JSON](table-geometry.json) and [reproducer](table_geometry.py)
analyze saved m000, m110, full spectral and critic-only spectral Ring16 states.
All four retain 256 prior rows, latent dimension 4, independent target/prior
sigma .1, protocol seed 0 and the shared public initializer. The six endpoint
gate metrics reproduce their archived grades. No models, gradients, draws or
updates occur; Python and Torch CPU RNG states remain unchanged.

Variances below are in target component variance units, averaged equally over
all 16 target components. Latent traces are population covariance traces.

| Endpoint | Global latent trace | Within-component latent trace | Observed tangent | Between mapped centers, tangent | Local noise, tangent | Local noise, radial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m000 | 4.975 | 1.836 | 3.306 | .748 | 2.335 | .853 |
| m110 | 5.454 | 1.711 | 2.327 | .432 | 1.720 | .256 |
| Full spectral | 5.102 | 2.097 | 2.586 | .300 | 2.171 | .199 |
| Critic-only spectral | 5.139 | 1.706 | 2.626 | .329 | 1.927 | .481 |

Local tangent noise exceeds target variance in 15/16 m000 components and 16/16
for every other endpoint. The local-noise proxy is 71–84% of each recipe's mean
observed tangent variance; this ratio is **not an exact variance partition**.
Between-row center dispersion adds a smaller positive contribution. Fixed
prior noise has covariance `.01 I`, independent of the table's global radius;
its local output covariance is `.01 J Jᵀ`. Thus the saved Jacobians already
stretch tangent noise while some radial directions are compressed.

The global latent traces differ by at most 9.6%. m110 has the largest trace but
the smallest local tangent noise. Full spectral has greater within-component
latent spread than m000, yet less mapped-center tangent spread. These endpoint
contrasts weigh against a simple larger-table/larger-output-variance story.
Initial prior receipts share the exact original hash, but archived initial
location values are unavailable. No initialization replay is authorized, so
absolute expansion from initialization remains **UNAVAILABLE**.

Crossing each saved G with each other saved table increases its mean squared
Jacobian Frobenius norm: every G has its smallest value at its own table. Own
values are 3.163/1.988/2.356/2.404; crossed values range 3.844–5.254. Crossed pairs
also place 159–199 mapped centers outside 3 target sigmas, versus 0–13 for own
pairs. These untrained functional combinations reveal endpoint coadaptation;
they are separate diagnostics and confer no gate credit or causal attribution.

A bounded, zero-training falsifier is available for the claim that removing
between-row center dispersion alone repairs density. Set that contribution to
zero in the first-order proxy while retaining the saved local noise covariance:

| Endpoint | Noise-only proxy covariance error | Minimum noise-only eigen ratio |
| --- | ---: | ---: |
| m000 | 1.018 | .263 |
| m110 | .774 | .023 |
| Full spectral | 1.016 | .078 |
| Critic-only spectral | .802 | .051 |

Every proxy still violates covariance <=.85 or minimum eigen ratio >=.15.
This rejects a center-dispersion-only explanation **within this approximation**,
not a future joint trajectory that also changes G. Finite-noise activation
crossings, nearest-mode reassignment and unknown consumed row IDs prevent an
exact additive sample decomposition. A LeakyReLU G's Jacobian is constant
inside each activation cell, so prior repositioning changes local noise only
through activation crossings or subsequent generator learning.

Preserve the prior LR 1→.25 negative: candidate `r1r2-prior-slow-v1`, attempt
`ed5c0491f733440f845806e3e59ec96a`, endpoint covariance 3.175129, HQ .475342,
12 modes, original FAIL. Its complete revision/source/checkpoint identities
remain in the JSON. Full spectral already changes prior updates to sampled-row
normalization and still fails Ring16. The supported initial-reference proximal
policy is restricted to zero-momentum full DualNorm; no measured initial-table
drift or directional restoration benefit motivates it here. Faster prior,
freezing, another LR point or a prior regularizer grid does not follow from
this diagnosis.

Old evidence retains its original execution commits and source. New spectral
states bind execution `220d37827e910c4f39c0d1c7afb1f337627c53ab`; all four use
scientific source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Actual candidate revisions, artifacts, initialization receipts and failures are
retained in the JSON. Current whole-campaign collection remains root-owned.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/prior/table_geometry.py > reports/forge/r1r2-spectral-repair/prior/table-geometry.json
```

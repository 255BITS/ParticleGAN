# Fixed-scale network spectral clipping diagnostic

This study tests whether G/D's unit-normalized motion prevents the 1-D Gaussian
from settling while the learned prior retains its existing update rule. It is
one independent mechanism comparison in the three investigations requested by
the user. The source branch depends on open PR [#317](https://github.com/255BITS/ParticleGAN/pull/317).

The public Recipe option `network_update="spectral_capped"` replaces each G/D
matrix's polar direction `U Vᵀ` with `U diag(min(s / 0.1, 1)) Vᵀ`. It retains the
original matrix aspect factor `sqrt(max(1, out/in))`. Vector/scalar biases use
`g / max(norm(g), 0.1)`. Small singular values retain their gradient magnitude,
including zero singular values in rank-deficient matrices; large values keep
the previous spectral motion cap. The fixed scale 0.1 is the same for G/D and
both tasks. It was selected before spending, with no trained-checkpoint tuning.
The nominal learning rates remain constant: G .012, D .018, prior .03.

The learned prior still uses the original sampled-row unit normalization.
This investigation changes G/D only. It therefore cannot show that shrinking
prior and network motion together would or would not work. The three timing
arms are ordinary D-then-G, simultaneous joint gradients, and extrapolation
from the past. The latter uses the same exact direction helper for its preview
and correction. Its cache and optimizer configuration are checkpointed.

## Frozen conditions and gates

The [protocol](protocol.json) freezes seed 0, the public deterministic initializer,
batch 128, 256 learned uniform MoG locations, sigma .1 and no standardization.
Gaussian is N(2, .5²), z dimension 2, width 32/depth 2, with Fourier critic order
2. Ring16 retains z dimension 4 and width 64/depth 2. Each timing arm uses the
same initial networks, prior, optimizer state, real batches and isolated,
checkpointed constructor/data/training/evaluation streams within each task.

All six stationary arms run 4000 updates. Gaussian must acquire five terminal
full passes at update 1000 and pass all 72 scheduled retention checks. Ring
must acquire at 1600 and retain all 144 subsequent checks. Gaussian bounds are
mean error ≤ .2 target sigmas, standard-deviation ratio .8–1.2, KS ≤ .05 and
finite output. Ring retains every original coverage, precision, balance and
local covariance bound. No gates or deadlines are softened.

Three Gaussian continuations shift the target mean from 2 to 3 at update 4000,
retaining target sigma .5 and all optimizer/cache history. Reacquisition requires
five terminal full passes at 5000 and all 24 remaining retention checks through
6000. Each active arm has a separately frozen checkpoint copy receiving exactly
the same evaluation draws. It takes no updates. A failed stationary gate
precludes a continuous-learner pass even if the shifted endpoint looks good.

The finite allowance is nine trials, 30,000 new updates, 4500 reserved loop
seconds and zero scientific retries. All neural construction, training,
sampling and mechanism tests use CUDA. The original CPU target law, saved-array
numerical scoring, metadata tests and rendering retain their separate roles.
This explicitly scoped diagnostic confers no ordinary qualification credit.
Existing PR317 controls retain their original evidence identities and cost.

## Reproduction

Training uses `/usr/bin/python`; rendering uses the project Python environment
with Matplotlib/Pillow. Logs are ignored and easy to tail:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python -u -m benchmarks.toy_audit.gaussian_network_magnitude run \
  --device cuda:0 --output runs/api/gaussian-network-magnitude-v1 \
  > runs/api/gaussian-network-magnitude.run.log 2>&1
tail -f runs/api/gaussian-network-magnitude.run.log
```

The adapter reuses the established public GANTrainer trial runner with scoped
protocol and initial-proof bindings. The archived PR317 driver/protocol remain
byte-identical; scientific replay uses the declared source hashes and commit.
Its software tests now use explicit metadata fixtures for the current API and
confer no archived-replay credit.

The initial-only field audit disables both optimizer corrections, computes one
public joint field per task, records gradient spectra and implied motion, then
restores the complete context exactly. It performs zero parameter updates and
cannot alter the scale or scientific recipe. Publication recomputes the saved
sample metrics and renders actual-training GIFs, including failures. Verification
restores every final CUDA context and checks constant effective rates, optimizer
counts and frozen control immutability.

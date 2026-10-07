# Fixed-scale network spectral clipping diagnostic

G/D spectral clipping improves the Gaussian substantially, but **does not solve
the continuous-learning gate** at this fixed scale and budget. Past extrapolation
passes 43/96 stationary checks (41/72 retention), reaches its first five-pass
window at update 3875, and ends the shifted problem with KS .03837. Acquisition,
strict retention and shifted reacquisition still fail. Every ring arm fails all
240 checks, so the adopted ring recipe should remain unchanged. All nine CUDA
trials complete 30,000 new updates for **362.227 loop seconds**, with zero
scientific retries.

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

## Numerical readout

All rows below fail the combined acquisition-and-retention gate. These are
diagnostic readouts, with no qualification credit or new generated leaderboard.
[Complete compact metrics](results.json) and [original source-bound controls](control-reuse.json)
preserve the final endpoints, deadlines and original evidence identities.

| Gaussian stationary | Full passes / 96 | Retention / 72 | Longest streak | Final mean error / sigma | Std ratio | KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alternating | 16 | 11 | 2 | .06916 | 1.02670 | .04838 |
| Simultaneous | 0 | 0 | 0 | .05956 | .71069 | .13628 |
| Past extrapolation | 43 | 41 | 5 | .01287 | .99263 | .06083 |

Each acquisition terminal suffix is zero. Alternating's final snapshot passes,
but only 11 of 72 retention checks do. Past extrapolation's first five-pass
window arrives at 3875, and the final snapshot fails KS. The matching original
past-extrapolation control passed only 3/96, longest streak 2, final KS .26799;
matching original alternating passed 3/96 and ended with KS .09594. Joint timing
alone still passes none. The clipping mechanism improves the Gaussian's live
trajectory while leaving substantial variability.

| Ring stationary | Full passes / 240 | Retention / 144 | Final component covariance error | Minimum eigen ratio | HQ | Mass TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alternating | 0 | 0 | 2.01671 | .36483 | .97510 | .06787 |
| Simultaneous | 0 | 0 | 1.85419 | .32088 | .97632 | .06665 |
| Past extrapolation | 0 | 0 | 2.05553 | .77600 | .87378 | .06274 |

Ring retains 16 modes at every final endpoint, but component covariance fails
its ≤ .85 bound. The new rule is a regression from the original alternating
ring's acquisition PASS and 144/144 retention. Endpoint coverage and local core
spread cannot replace the unchanged full covariance gate.

| Gaussian mean-2-to-3 shift | Full passes / 48 | Retention / 24 | Longest streak | Final mean error / sigma | Std ratio | KS | Frozen KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alternating | 0 | 0 | 0 | .11330 | .87208 | .07497 | .69716 |
| Simultaneous | 0 | 0 | 0 | .29254 | .74436 | .23232 | .76401 |
| Past extrapolation | 17 | 8 | 3 | .03319 | .92232 | .03837 | .69229 |

Every shifted reacquisition suffix is zero; every frozen control passes 0/48.
Past extrapolation adapts and its final endpoint passes all bounds, while the
frozen checkpoint remains near the old mean. Its 17 full passes and final
endpoint are useful response evidence, but 8/24 retention and longest streak 3
fail the declared continuous-learning claim. No history was reset.

## What the mechanism audit shows

The [initial-only audit](initial-gradient-audit.json) leaves all parameters and
optimizer histories unchanged and restores the whole context exactly. At the
Gaussian initializer, G's hidden-matrix smallest singular value is 6.2e-13 and
D's is 9.5e-11. The original polar rule would complete these small directions to
unit singular values. The new mean singular motion fractions are .04383 for G
and .08373 for D. Their Frobenius direction norms are 18.2% and 25.8% of the
original matrix motion cap. Ring G's hidden-matrix mean fraction is only .00682,
which already predicts slower initial network response; the fixed scale was
not changed after this observation.

The [saved-cache audit](saved-cache-audit.json) uses no new gradients or model
samples. At Gaussian update 4000, G/D hidden-matrix directions use only 2.84%
and 7.23% of their original cap norms. The 97 active prior rows still move a mean
.02997 per update, close to the unchanged .03 nominal prior rate. At shift end,
network utilization remains 2.99%/8.12%, prior motion .02999. The new rule reduces
network motion, while sampled prior motion remains substantial; these different
parameter units do not establish a causal magnitude ratio.

The [saved-shape readout](saved-shape-and-field-audit.json) also improves alternating
Gaussian fitted-normal KS from the original .17296 to .05782 and skew from 1.61
to -.273. Past extrapolation ends near the target mean/width, with fitted-normal
KS .05775 and skew .098. Its remaining KS failure therefore includes shape
error, beyond mean and width. These posthoc diagnostics explain observed motion
and shape; they do not change any gate.

Stop this exact scale-.1 recipe as a global smoke-test repair. Keep the network
cap as an opt-in research capability, retain the adopted ring recipe, and compare
this evidence with the separately executed frozen-prior and magnitude-sensitive
prior studies. If their outcomes justify combining prior and network response,
that needs its own frozen comparison and budget. This experiment does not test
that combination, other cap scales, extra steps or seeds.

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

## Verification and artifacts

All **67 software checks** pass: 23 CUDA mechanism/protocol fixtures and 44
metadata checks. The CUDA cases verify small/large/rank-deficient fields, exact
correction/preview equality, unchanged prior-row ownership, first-past-step
identity with simultaneous timing, exact resume, unsupported rules and default
optimizer equality with the pinned original source. All nine final CUDA contexts
restore exactly, with constant effective rates and correct optimizer counts;
all three frozen training states and caches remain unchanged. Publication
recomputes **1308 saved live/frozen sample sets**, reproduces every grade, and
renders nine actual-training GIFs, each with nine frames.

The initial field audit's first software preflight rejected transient sampled-row
and critic-observer bookkeeping. Its repaired check verifies unchanged parameter
histories and exact whole-state restore; it consumed zero parameter updates.
The first publication invocation lacked the declared one-thread environment and
failed an exact floating-point metric comparison. Repeating saved-array publication
with one thread reproduces all metrics. Both failed stdout logs are preserved;
no scientific training trial was repeated.

Scientific source: `4335e2bd4806e218d0d6e544973c53078d344c1a`.
Protocol SHA256: `bd3ab3141ae30538667bfe43d3eebfce38124038cdc138e1159fb4ee77b43f59`.
[Provenance](provenance.json), [verification](verification.json),
[exact restores](restore-proof.json), [stationary curves](stationary.svg) and
[shift curves](shift.svg) bind the implementation, source, budget and artifacts.
The measured loop time includes scheduled scoring, excludes construction,
checkpoint serialization and publication. Raw logs, curves, checkpoints and
tensors remain ignored under `runs/api/gaussian-network-magnitude-v1/` and its
content-hashed archive.

Actual-training media:

- Gaussian stationary: [alternating](alternating-gaussian1d_acquisition-stationary.gif), [simultaneous](simultaneous-gaussian1d_acquisition-stationary.gif), [past](extrapolation_from_past-gaussian1d_acquisition-stationary.gif).
- Ring stationary: [alternating](alternating-ring16_acquisition-stationary.gif), [simultaneous](simultaneous-ring16_acquisition-stationary.gif), [past](extrapolation_from_past-ring16_acquisition-stationary.gif).
- Gaussian shift: [alternating](alternating-gaussian1d_acquisition-shift.gif), [simultaneous](simultaneous-gaussian1d_acquisition-shift.gif), [past](extrapolation_from_past-gaussian1d_acquisition-shift.gif).

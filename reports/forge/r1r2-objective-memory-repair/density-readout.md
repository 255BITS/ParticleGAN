# Saved density audit for global transport and AMSGrad on m110

Both new global recipes pass five of six ordinary tasks plus their own clock,
but fail sustained Ring16. The [audit](density-audit.json) verifies **192 saved
Ring panels**, 96 per recipe. All six original gate metrics reproduce exactly,
with maximum absolute difference zero. Each recipe has **0/96 covariance-bound
passes**, **0/96 full passes** and terminal suffix **0/5**. Qualification is
unchanged; neither recipe completes the repair.

The [reproducer](density_audit.py) imports the unchanged, hashed
[five-track equations](../r1r2-five-track-repair/density_readout.py) for metrics,
windows and functional saved-state geometry. No models or optimizers are
constructed; no new draws, updates or grades are produced. Python and Torch CPU
RNG states remain unchanged. Reference-target/global-moment/SW1 scoring is
excluded. Each exact cached parent keeps its original metrics and execution;
only its saved endpoint geometry is additionally diagnosed.

The original gate requires at least 4,096 samples, all 16 HQ modes, mass TV
<=0.15, HQ >=0.85, component covariance error <=0.85 and minimum eigen ratio
>=0.15 for five terminal observations within 1,600 updates.

| Global recipe | Exact parent | Best covariance error (step) | Endpoint covariance | HQ | HQ modes | Mass TV | Min eigen ratio | Endpoint failed bounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cached Adam | — | 2.098129 (1217) | 2.386427 | 0.842285 | 16 | 0.063477 | 0.433621 | HQ, covariance |
| Global transport weight 1 | Adam | 2.072274 (1550) | 2.352042 | 0.760498 | 16 | 0.033447 | 1.164918 | HQ, covariance |
| Cached m110 | — | 0.904193 (1584) | 1.083997 | 0.923340 | 16 | 0.068115 | 0.111723 | covariance, minimum spread |
| AMSGrad on m110 | m110 | 1.535922 (1467) | 1.679120 | 0.879150 | 16 | 0.046875 | 0.159521 | covariance |

**AMSGrad on m110 has the better density of these two new recipes, but worsens
density relative to its own m110 parent.** It repairs that parent's Gaussian
and word acquisition failures. Global transport slightly reduces covariance
error relative to Adam while reducing endpoint HQ; HQ fails on all 96 panels.
Its five terminal covariance errors are 2.429770 / 2.072274 / 2.430521 /
2.237538 / 2.352042. AMSGrad's are 1.837218 / 1.614646 / 1.723391 /
1.645077 / 1.679120; its first four terminal panels also fail minimum spread.
Passing the endpoint's other bounds supplies no sustained Ring pass.

## Acquisition and endpoint quality are distinct

The original word task confirms acquisition after one passing observation;
it still executes the full 20,001 updates. The original
[publication](publication/results.json) retains these gates and evaluator
summaries without tightening or regrading them.

| Recipe | Original word gate | Passing observations | First confirmation | Endpoint modes | Endpoint quality | Endpoint TV | Exact inverse reconstruction | Minimum token probability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global transport | PASS | 1/24 | 834 | 2 | 0.395508 | 0.612500 | 0 | 0 |
| AMSGrad on m110 | PASS | 24/24 | 834 | 5 | 1.000000 | 0.018945 | 1 | 0.998372 |

Transport's sole acquisition confirmation occurs at step 834; its saved
evaluator explicitly reports `endpoint_passed=false`. Its PASS therefore
establishes acquisition, while the terminal model loses coverage and inverse
reconstruction. AMSGrad on m110 preserves all 24 confirmations and passes the
word endpoint. Both recipes pass Gaussian, pole, token/AE holds and clock;
Ring is the remaining original task failure.

## Remaining component-shape defect

Geometry uses original float32 assignments, float64 population moments and
the actual saved generator/prior tensors. Variance ratios use target component
variance units; target total trace is 2. All 16 components are available.

| Recipe | Mean radial variance (range) | Mean tangent variance (range) | Mean local Jacobian noise trace | Mean between-center trace | Prior means outside 3 sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached Adam | 2.215 (0.435–16.485) | 3.306 (1.290–6.467) | 3.188 | 1.898 | 3/256 |
| Global transport | 3.116 (1.165–9.976) | 3.178 (2.508–4.119) | 3.873 | 2.314 | 12/256 |
| Cached m110 | 0.575 (0.133–2.496) | 2.327 (1.600–3.466) | 1.976 | 0.721 | 1/256 |
| AMSGrad on m110 | 0.413 (0.188–1.031) | 3.251 (2.054–4.782) | 2.963 | 0.460 | 0/256 |

Transport remains broad in both directions in every component. AMSGrad on
m110 instead has compressed radial variance in 15 components and broad
tangential variance in all 16, despite all mapped prior means being within
three target sigmas. Coverage and center placement do not establish the target
component shape. Its larger local Jacobian trace and tangent variance than
m110 accompany the worsened covariance error. These saved-state comparisons
do not identify a unique historical gradient, generator or prior cause;
first-order Jacobian and center traces are not an exact additive variance
partition. Neither arm supports adoption as a complete repair.

## Actual update law and provenance limits

Both recipes use native PyTorch Adam for every G/prior/D group, with rate
.0085, epsilon 1e-8, beta2=.99 and all parameters at step 1,600. Both retain
the learned, unregularized prior and constant R1/R2 gamma .1 every update.
The audit compares each complete task-bound recipe to its own cached parent:
transport changes only `kinetic_transport_weight`, from 0 to 1; AMSGrad on
m110 changes only `amsgrad`, from false to true.

Transport retains beta1=0 in all roles, local transport weight 0 and 32 fixed
projections. The immutable public source binds its active loss through
`GANTrainer.step -> Recipe.kinetic_transport_loss(fake_g, real_g)`, using the
existing generator-phase batches and no added RNG draws. The saved source,
recipe and completed updates establish this binding; per-update auxiliary
values and force decompositions are not stored.

AMSGrad on m110 retains G/D beta1=.5 and prior beta1=0. Actual saved maxima
exist in every optimizer state, have the correct shapes and are at least
their current second moments. G/D first moments average history and cannot
recover instantaneous gradients. Beta1=0 prior buffers are terminal
gradients, but no historical sampled-row ownership or unsampled displacement
is inferred from gradient support. No missing gradient is synthesized.

The audit verifies execution `2b0ad9ec7c5b2c1e91ab0d70751da0b7e36a7b42`,
scientific source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`
and runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Candidate revisions/job keys, protocol seed 0, every frozen source file,
original task/scorer bindings, artifact hashes, complete state digests and
every saved named RNG stream match their receipts. Captured actual initial
tensors and ordered consumed target batches match each exact cached parent;
this verifies receipt consistency rather than reconstructing initialization.
Adam retains execution `078f3897e672bfae29504d40019d99c7c5c91ba9`, Ring attempt
`27c75f3d768e4b789a488a3fdc5a2e90`; m110 retains execution
`6f634cec5c3ec6e0dcda7c1508a2eafaf9fb2bfd`, Ring attempt
`20aca4f16e8a4c31900b652e2c71ee3b`. The audit records their exact revisions.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-objective-memory-repair/density_audit.py --execution-commit 2b0ad9ec7c5b2c1e91ab0d70751da0b7e36a7b42 > reports/forge/r1r2-objective-memory-repair/density-audit.json
```

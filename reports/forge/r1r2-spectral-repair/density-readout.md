# Saved density audit for full and critic-only DualNorm

Both new global recipes fail sustained Ring16. The [audit](density-audit.json)
verifies **192 saved panels**, 96 per recipe. All six original gate metrics
reproduce **exactly**, with maximum absolute difference zero. Each arm has
**0/96 covariance-bound passes**, **0/96 full passes** and terminal suffix
**0/5**. The original qualification results are unchanged.

The [reproducer](density_audit.py) imports the unchanged, hashed
[five-track equations](../r1r2-five-track-repair/density_readout.py) for gate
metrics, windows and functional endpoint geometry. No models or optimizers are
constructed; no new draws, updates or grades are produced. Python and Torch CPU
RNG states remain unchanged. Reference-target/global-moment/SW1 scoring is excluded.

The original gate requires at least 4,096 samples, all 16 HQ modes, mass TV
<=0.15, HQ >=0.85, component covariance error <=0.85 and minimum eigen ratio
>=0.15 for five terminal observations within 1,600 updates.

| Global recipe | Best covariance error (step) | Endpoint covariance | HQ | HQ modes | Mass TV | Min eigen ratio | Endpoint failed bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cached Adam parent | 2.098129 (1217) | 2.386427 | 0.842285 | 16 | 0.063477 | 0.433621 | HQ, covariance |
| Full DualNorm | 1.128805 (1584) | 1.180543 | 0.897461 | 16 | 0.045166 | 0.288719 | covariance |
| Critic-only DualNorm | 1.149246 (1550) | 1.212486 | 0.830322 | 16 | 0.041748 | 0.144263 | HQ, covariance, minimum spread |

**Full DualNorm is the stronger density arm.** All five of its terminal panels
pass every bound except covariance. Their covariance errors are
1.226326 / 1.345509 / 1.185637 / 1.128805 / 1.180543, each above 0.85.
Critic-only DualNorm also improves covariance over the cached parent, but its
last two panels lose HQ and minimum spread. These improvements supply no
sustained pass. Both recipes pass Gaussian, pole, token/AE holds and their own
clock, but fail word acquisition, as the [publication](publication/results.json)
records. Full DualNorm ends with four word modes and inverse reconstruction 0;
critic-only ends with one mode and inverse reconstruction 0.

## Remaining saved shape defect

All geometry uses original float32 sample assignments, float64 population
moments and the saved generator/prior tensors. Ratios use target component
variance units; target total trace is 2. The parent is a separate endpoint
diagnosis retaining its original gate metrics, with no parent gate recomputation.

| Recipe | Mean radial variance (range) | Mean tangent variance (range) | Mean local Jacobian noise trace | Mean between-center trace | Prior means outside 3 sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached Adam parent | 2.215 (0.435–16.485) | 3.306 (1.290–6.467) | 3.188 | 1.898 | 3/256 |
| Full DualNorm | 0.635 (0.317–0.973) | 2.586 (1.912–3.464) | 2.371 | 0.718 | 0/256 |
| Critic-only DualNorm | 0.623 (0.147–1.220) | 2.626 (1.253–4.962) | 2.409 | 0.452 | 13/256 |

Every full-DualNorm component remains radially compressed and tangentially
broad despite all mapped prior means lying within a 3-sigma target region.
Coverage and center placement therefore do not establish the correct component
shape. A uniform contraction would also shrink the compressed direction; these
saved panels do not support it as a complete repair. The full/critic-only
contrast bundles G and prior update-law changes and changes later critic
feedback, so it cannot identify a unique generator or prior cause.

The matrix polar law improved density relative to both Adam and the earlier
direction-preserving Ada arm, but is insufficient to meet the original gate.
Jacobians and center traces diagnose output geometry; they are not an exact
additive variance partition or a measurement of historical update spectra.
No globally useful fixed smoothing scale is established by this audit.

## Actual optimizer law and proof limits

Saved groups prove full `dualnorm` on G/D matrices and vectors, with `rownorm`
on the sampled prior and `sampled_rows_required=True`. All these states contain
only step counters: declared beta2=.99 is inactive in their update laws.
Critic-only family `dualnorm_D_only` has native Adam algorithms and both moment
buffers on G/prior, with the same DualNorm critic law. All parameters have
1,600 saved steps; rates are .0085, effective epsilon 1e-8, momentum/smoothing
zero, native SVD backend, no convolution variant and AMSGrad disabled.

**Raw gradient limits:** full DualNorm stores no raw gradients or first/second
moments, so actual terminal G/D/prior gradients cannot be recovered. The
critic-only G/prior first moments are terminal gradients because beta1=0;
its critic still stores none. No missing gradient is synthesized.

**Row-ownership limits:** the unchanged, hashed public training/policy hooks
bind actual generator-side draws to the full prior's required-row law.
Optimizer steps clear those sampled IDs. Endpoint checkpoints and RNG receipts
cannot independently reconstruct every historical sampled row set or prove
individual unsampled-row displacements. No gradient-support mask substitutes
for actual row ownership.

The audit verifies execution `220d37827e910c4f39c0d1c7afb1f337627c53ab`,
scientific source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`
and runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Candidate revisions/job keys, protocol seed 0, every frozen source file,
original task/scorer bindings, all artifact hashes, complete state digests and
every saved named RNG stream match their receipts. Captured actual initial
tensors and ordered consumed target batches match the verified cached parent;
this is receipt consistency rather than independent initialization reconstruction.
The parent retains original execution `078f3897e672bfae29504d40019d99c7c5c91ba9`
and attempt `27c75f3d768e4b789a488a3fdc5a2e90`.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-spectral-repair/density_audit.py --execution-commit 220d37827e910c4f39c0d1c7afb1f337627c53ab > reports/forge/r1r2-spectral-repair/density-audit.json
```

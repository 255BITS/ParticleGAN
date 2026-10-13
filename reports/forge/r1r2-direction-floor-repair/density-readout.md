# Saved density audit for direction and critic epsilon

All three new global recipes fail sustained Ring16. The [audit](density-audit.json)
verifies **288 saved panels**, 96 per recipe: all six original gate metrics
reproduce **exactly**, with maximum absolute difference zero. Each arm has
**0/96 covariance-bound passes**, **0/96 full passes** and terminal suffix
**0/5**. The original grades and qualification records are unchanged.

The [reproducer](density_audit.py) imports the unchanged, hashed
[five-track equations](../r1r2-five-track-repair/density_readout.py) for metrics,
windows and functional endpoint geometry. It constructs no models or optimizers,
makes no sampling draws and performs no training. Python and Torch CPU RNG
states remain unchanged. Reference-target/global-moment/SW1 scoring is excluded.

The original gate requires at least 4,096 samples, all 16 HQ modes, mass TV
<=0.15, HQ >=0.85, component covariance error <=0.85 and minimum eigen ratio
>=0.15 for five terminal observations within 1,600 updates.

| Global recipe | Best covariance error (step) | Endpoint covariance | HQ | HQ modes | Mass TV | Min eigen ratio | Endpoint failed bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cached Adam parent | 2.098129 (1217) | 2.386427 | 0.842285 | 16 | 0.063477 | 0.433621 | HQ, covariance |
| Adam, critic eps 1e-4 | 2.294437 (1584) | 2.635454 | 0.846191 | 16 | 0.039307 | 0.297133 | HQ, covariance |
| AdaNSGDA | 2.647227 (1584) | 4.570228 | 0.432861 | 14 | 0.074219 | 0.697502 | modes, HQ, covariance |
| AdaNSGDA, critic eps 1e-4 | 2.661903 (1500) | 7.055577 | 0.258057 | 6 | 0.061523 | 1.129127 | modes, HQ, covariance |

**Adam with critic epsilon is the best new density arm**, but both its best
and endpoint covariance errors exceed the exact cached parent's. Its final
five errors are 3.378588 / 2.980432 / 2.354288 / 2.294437 / 2.635454.
The epsilon arm preserves the parent's passing Gaussian, pole and hold gates;
words still fail. Both Ada arms likewise pass Gaussian, pole and holds but
fail words and Ring16. All three own clock gates pass, as recorded in the
[whole-recipe publication](publication/results.json). None supplies a Tier 1 repair.

## Actual saved geometry

The table uses all 16 assigned components, population sample covariance and
the saved G/prior tensors. Variances and traces are in target component
variance units; the target total trace is 2. The parent geometry is a separate
saved-endpoint diagnosis, with zero recomputed parent gate metrics and its
original results retained.

| Recipe | Mean radial variance (range) | Mean tangent variance (range) | Mean local Jacobian noise trace | Mean between-center trace | Prior means outside 3 sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached Adam parent | 2.215 (0.435–16.485) | 3.306 (1.290–6.467) | 3.188 | 1.898 | 3/256 |
| Adam, critic eps 1e-4 | 2.971 (0.305–26.938) | 2.894 (1.581–5.185) | 3.062 | 2.395 | 8/256 |
| AdaNSGDA | 2.621 (0.725–13.307) | 6.538 (2.692–12.398) | 3.562 | 5.977 | 123/256 |
| AdaNSGDA, critic eps 1e-4 | 4.986 (1.240–26.472) | 9.057 (4.146–18.762) | 5.165 | 8.375 | 185/256 |

Preserving each parameter tensor's raw-gradient direction did not repair the
shape failure. Ada retains the Adam-derived tensor step norm and changes all
trainer roles; its trained feedback trajectory differs from Adam. Its much
broader tangent variance and displaced prior means weigh against the hypothesis
that removing within-tensor gradient rotation alone would solve Ring16.

The surviving Adam-derived update magnitudes and their distribution across
tensors remain plausible research targets. The broad functional Jacobian
noise traces and center dispersion support inspecting the resulting geometry
alongside update magnitudes or spectra. This audit does **not** measure update
singular-value histories or establish an update-spectrum cause. It cannot
isolate magnitude, role interaction, prior transport or feedback as the unique
cause. The critic epsilon result also rejects this particular denominator
floor as a sufficient repair. These observations identify no qualified replacement
recipe. Jacobian traces are first-order diagnostics, not an exact additive
partition of sampled variance.

## Verified ownership and provenance

The checkpoints independently verify native Adam for `adam_eps` and
`NormalizedOptimizer` family/algorithm `ada_nsgda` for `ada`/`ada_eps`, with
generator, sampled-prior and critic groups correctly bound. Every saved
parameter has 1,600 steps, beta1=0, beta2=.99, LR=.0085 and AMSGrad disabled.
G/prior epsilon is 1e-8; D epsilon is 1e-4 only in the epsilon arms. Ada states
have second moments without first-moment buffers; native Adam retains both.
Ada uses D epsilon in both the Adam denominator and the norm-graft denominator.

New evidence belongs to execution `a7f141c9c73d7aead972f2d7a382b2f52a59ac69`,
scientific source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`
and runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
The audit verifies admitted candidate revisions/job keys, every frozen source
file, task/scorer bindings, request/raw/graded/panel/checkpoint file hashes,
complete state digests and every saved named RNG stream. Actual initial-tensor
and consumed target-batch receipts match the verified cached parent. This is
receipt consistency, not independent reconstruction of initialization.

The cached parent retains original execution
`078f3897e672bfae29504d40019d99c7c5c91ba9`, candidate revision
`35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`
and attempt `27c75f3d768e4b789a488a3fdc5a2e90`.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-direction-floor-repair/density_audit.py > reports/forge/r1r2-direction-floor-repair/density-audit.json
```

# Saved prior and role interaction

The existing role factorial contains the useful prior contrasts: **m001** adds
only `prior_betas=(.5,.99)` to cached m000; **m110** removes only prior momentum
from cached m111, retaining G/E and D beta1=.5. These use supported public
Recipe fields and have not been measured. No additional candidate is needed.
Both must pass all six original Tier 1 tasks and clock; the saved density
improvement alone supplies no qualification.

[Exact readout](analysis.json) and [read-only reproducer](analyze_saved.py) bind
the terminal Ring16 states to execution commit
`9ad31157b8327044f9364b5fc458659cced0c8f0`, scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`,
candidate revisions and artifact SHA256s. Reproduction consumes zero draws and
updates; Python and Torch CPU RNG states remain unchanged.

| Terminal saved-state diagnostic | Global momentum | AMSGrad, beta1=0 |
| --- | ---: | ---: |
| Moving prior rows /256 | 256 | 105 |
| Off-batch moving rows, lower bound (G batch128) | 128 | 0 |
| Prior latent covariance minimum eigenvalue, unbiased | .942036 | .926676 |
| G center displacement, mean norm | .024692 | .063108 |
| Prior center displacement, mean norm | .003556 | .002713 |
| Opposing G/prior center displacement rows | 117/256 | 48/105 |
| Prior-only activation pattern changes /256 | 9 | 3 |
| Mean local radial variance /target variance, 256-row weighted | .372493 | .455934 |
| Mean local tangent variance /target variance, 256-row weighted | 1.721494 | 3.166012 |

Full-rank latent covariance does not support a collapsed-table explanation for
the compressed output radial direction. Global momentum moves every row,
including at least128 rows outside the current G batch. That residual transport
may help coverage or disturb density; these endpoint states cannot establish
which effect dominates. AMSGrad's same-moment denominator suppresses median
prior displacement to about14% of ordinary Adam's algebraic displacement, but
the two experiments have different state trajectories.

For a LeakyReLU generator, fixed-weight prior motion inside an activation cell
changes the mapped center while leaving its local Jacobian unchanged. The
reconstructed final step changes activation patterns on only9/256 momentum
rows and3/256 AMSGrad rows. G displacement is about7× and23× prior displacement,
respectively. Prior momentum therefore tests row transport and player
interaction; it is not a direct radial-Jacobian correction. The simultaneous
G/D changes in m111 prevent assigning its better covariance to the prior.

Actual task bindings in the JSON come from public `task_formulation_context`
without model construction. `Recipe.prior_betas` overrides the sampled prior
group, `Recipe.betas` controls G/E, and `Recipe.d_betas` overrides D. Prior
controls are inapplicable to the direct-coordinate pole and unsampled token
fixtures; pole uses G betas. There is one shared `Recipe.amsgrad` switch and no
supported role-specific AMSGrad switch. The registered global
AMSGrad+all-momentum contrast covers a separate interaction hypothesis.

The displacement is recovered algebraically from the final native Adam
moments with bias correction and epsilon. Subtracting it from float32 terminal
parameters gives an approximate inverse, **not a saved pre-update checkpoint**.
No fresh gradient or future update is inferred. Local variance ratios use
Jacobians at256 row centers and final nearest-target radial frames; finite-noise
activation crossings and mode reassignment are omitted. These descriptive
ratios have a different weighting from the component-level gate geometry in
the [full density readout](../../r1r2-five-track-repair/density-readout.md).
The declared full and sustained gates remain authoritative.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/prior/analyze_saved.py > reports/forge/r1r2-role-repair/prior/analysis.json
```

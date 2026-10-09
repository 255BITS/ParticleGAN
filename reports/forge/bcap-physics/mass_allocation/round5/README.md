# Round five: conservative center allocation

Preregistered, unmeasured: replace local-v2's sliced W2 term with joint-coordinate
balanced assignment in globally fixed contiguous blocks of128. Retain weight1,
exact local moments weight1, fullDualNorm and winning BCAP recipe unchanged.
Every source row and target row participates exactly once; no evaluator geometry,
labels, prior mass changes, routing, additional noise/draws or serving change.
Within a block the assignment minimizes squared Euclidean transport cost;
normalization uses dimension and the detached whole-real-batch coordinate variance.
This is exact empirical W2 within each block, not full-batch W2 across blocks.
For native2048 there are16 blocks; all other declared batches are128.

The [saved-state receipt](prior-diagnostics.json) shows native center TV .1459,
served TV .148 and only9/100 diagnostic three-sigma quality modes reaching half
their target mass. Equal-width rare centers are141/77/32/6; local-v2 already passes
that full task, so the new-source matched primary control retains it.
Saved128-row pairings cross diagnostic nearest labels less under assignment than
under sliced pairing: native .922→.781, rare .0984→.0469, broad .0991→.0313,
width .2087→.1094. These are output-space endpoint diagnostics, not training flux
or G/prior pullbacks. Archived finite G/prior role probes retain their exact source
and endpoint scope in the receipt. Oracle labels/centers/covariance enter diagnosis
only. Genuine served full gates remain authoritative.

[Fatras et al.](https://proceedings.mlr.press/v108/fatras20a.html) analyze minibatch
OT and its loss of the population distance property; this motivates an explicitly
bounded test, not a convergence claim. [SciPy's assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
provides the minimizing square bijection. Assignment ties select one branch.
CPU solving requires the existing optional SciPy dependency and synchronization;
shared-device timings supply accounting only.

Two ready schema-v3 arms share one frozen source/runtime and seed0. Full original
native7000, rare/broad/width1200, Gaussian smoke1000 and own-stability5000 additional
updates, task architecture/data/prior/sampling and scoring cadence remain fixed.
Own failed-smoke dependencies remain authoritative. Frozen vector/Gaussian hosts
reuse the same actual real tensor for D/G. Public deterministic initialization
and all named streams remain isolated and checkpointed.

[Ready freeze](freeze.json) reserves19440 worker seconds,9720 per arm, plus three
separate120-second ancillary allowances within21600. One candidate, no tuning,
seed experiments, second proposal or promotion. Scientific prediction is native
precision≥.97; full sustained mass, shape, radial, Gaussian retention/reacquisition
and guardrails decide any repair. Diagnostic profile remains provisional.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/logs/driver.log
```

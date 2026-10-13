# R1/R2 saved-evidence diagnosis

The ordinary Ring16 failure is dominated by tangential within-component stretch.
All 16 components have variance along the ring tangent between **2.553 and 5.190
times** the target variance; their radial variance is **0.226–1.138 times** the
target variance. The recipe finds every mode and balances their mass; most radial
variances are below target while every tangential variance remains broad.
The clock failure has a separate mechanism: the recipe explicitly schedules LR,
Adam beta2 and the R1/R2 coefficient from the external completed-step label and
horizon. Removing only one schedule cannot establish clock independence.

This analysis consumes saved output panels and the saved endpoint parameters.
It launches **zero updates and zero sampling draws** and grants no qualification.
The [source-bound baseline](../r1r2-baseline/README.md) remains unchanged.

## Original Ring16 evidence

Source commit: `a0493ca3c213d559523eadc76e244ec9f9c1f81f`.
Source digest: `a5eafc73e4c4a995cc388b9be00bea5aec289fd35863099e369008e7def0ab14`.
Attempt: `a54bc96e82f34c9e98e7fa1f1d268a33`.
The original archive is
`/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/queue/recipe-prior-refactor-selected-leaders-v1/a54bc96e82f34c9e98e7fa1f1d268a33`.

| Saved artifact | SHA256 |
| --- | --- |
| `observed-samples.pt` | `874ecc14c1ab0dde54b29d50f6a1e47f9b4359bbb6c1290c83e421498317fd58` |
| `provenance/provenance-state.pt` | `8897396157ecb7e0daf6e9899c93c840a873a73ca0ed527312196bab2ba799cd` |

The first artifact contains all 96 original output panels of 4,096 samples. The
second is the saved public trainer at update 1,600, including G, D, learned MoG
locations, optimizer state and consumed named streams. Its recorded recipe has
schedule horizon **400**, execution cap **1,600**, sigma **0.1**, 256 rows,
latent dimension 4, batch 128, width 64, two layers and discriminator Fourier 2.
LR is already at **0.00085** from update 400 onward; beta2 is **0.99** and the
R1/R2 coefficient **0.1** from update 80 onward.

The endpoint has all 16 modes, mass TV **0.051514**, global covariance error
**0.051335**, but mean component covariance error **2.066388** and HQ **0.844238**.
Core covariance error **1.331667** still exceeds the ordinary full-covariance
bound **0.85**, showing that simply clipping the most distant samples would not
solve the shape failure. The gate uses the mean component covariance error;
individual component errors below are diagnostics, not new per-component gates.

The table recomputes geometry from the saved endpoint output panel on CPU in
float64. Variances divide by the target variance 0.01; radial means the direction
from the ring origin to that component, and tangent is perpendicular. Component
assignment is nearest declared target mean, as in the original scorer.

| Component | Output samples | Full covariance error | Radial variance / target | Tangent variance / target | Saved prior means mapped here |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 275 | 1.455724 | 0.610804 | 3.020557 | 17 |
| 1 | 341 | 2.470242 | 0.789100 | 4.479736 | 20 |
| 2 | 286 | 2.040013 | 0.298351 | 3.798109 | 18 |
| 3 | 238 | 2.114865 | 0.300604 | 3.907527 | 15 |
| 4 | 268 | 2.163502 | 0.656697 | 4.031258 | 17 |
| 5 | 288 | 2.145094 | 1.137837 | 3.988477 | 17 |
| 6 | 251 | 1.217186 | 0.372195 | 2.552901 | 16 |
| 7 | 260 | 1.706942 | 0.510102 | 3.363120 | 17 |
| 8 | 264 | 2.026349 | 0.913551 | 3.861317 | 16 |
| 9 | 161 | 2.769078 | 0.404820 | 4.863412 | 10 |
| 10 | 205 | 1.936438 | 0.481014 | 3.686047 | 15 |
| 11 | 260 | 2.415418 | 0.439384 | 4.360544 | 16 |
| 12 | 253 | 1.395787 | 0.359265 | 2.854701 | 14 |
| 13 | 234 | 2.107951 | 0.225660 | 3.875695 | 15 |
| 14 | 239 | 3.016356 | 0.300132 | 5.190073 | 16 |
| 15 | 273 | 2.081258 | 0.721025 | 3.842620 | 17 |

A deterministic forward evaluation of saved G at each saved MoG location maps
**10–20 rows** to each component. Only **1/256** output means lies beyond the
nearest component's 3-sigma radius; none lies beyond 4 sigma. This static mapping
is a diagnostic panel, not the task's clean MoG sampling law. It does not substitute
atomic sampling for the positive-width prior or alter any grade.

A first-order local approximation at each row uses
`0.1² × J_G(z) × J_G(z).T`. Its mean component trace is **3.724123** in target
variance units; the corresponding between-row output-mean covariance trace is
**0.485784**, while the desired total component trace is **2**. These are
unweighted averages over target components. The larger local-noise term supports
within-row generator stretch as the dominant explanation over simple dispersed
prior means. Noise crossing LeakyReLU boundaries and reassignment between modes
can invalidate a local approximation; it is not an exact variance decomposition
or proof that R1/R2 itself causes the stretch. No saved row identities accompany
the output samples, so a measured sample-by-row covariance decomposition remains
unavailable.

The following zero-draw reproduction uses only the two hashed artifacts. It
implements the archived two-layer LeakyReLU(0.2) forward/Jacobian directly from
saved weights, without constructing randomly initialized modules. Run in the
recorded Python environment; the unweighted mean trace values reproduce the
reported local approximation.

```python
from pathlib import Path
import torch

archive = Path('/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/queue/'
               'recipe-prior-refactor-selected-leaders-v1/a54bc96e82f34c9e98e7fa1f1d268a33')
torch.set_num_threads(1)
x = torch.load(archive / 'observed-samples.pt', map_location='cpu',
               weights_only=True)[-1]['samples'].double()
models = torch.load(archive / 'provenance/provenance-state.pt', map_location='cpu',
                    weights_only=False)['trainer']['models']
w = lambda i: models['G'][f'net.{i}.weight'].double()
b = lambda i: models['G'][f'net.{i}.bias'].double()
z = models['prior']['z'].double()
a1 = z @ w(0).T + b(0)
a2 = torch.nn.functional.leaky_relu(a1, .2) @ w(2).T + b(2)
y = torch.nn.functional.leaky_relu(a2, .2) @ w(4).T + b(4)
j = torch.einsum('oi,bi,ij,bj,jk->bok', w(4), torch.where(a2 >= 0, 1., .2),
                 w(2), torch.where(a1 >= 0, 1., .2), w(0))
local = .01 * (j @ j.transpose(1, 2))
theta = torch.arange(16, dtype=torch.float64) * (2 * torch.pi / 16)
means = 3 * torch.stack([theta.cos(), theta.sin()], 1)
sx, sy = torch.cdist(x, means).argmin(1), torch.cdist(y, means).argmin(1)
between, within = [], []
for k in range(16):
    u, v = x[sx == k], y[sy == k]
    centered, row_centered = u-u.mean(0), v-v.mean(0)
    cov = centered.T @ centered / len(u)
    row_cov = row_centered.T @ row_centered / len(v)
    radial = means[k]/3
    tangent = torch.stack([-radial[1], radial[0]])
    print(k, len(u), float(radial @ cov @ radial/.01),
          float(tangent @ cov @ tangent/.01), len(v))
    between.append(torch.trace(row_cov)/.01)
    within.append(torch.trace(local[sy == k].mean(0))/.01)
print('mean traces:', torch.stack(between).mean(), torch.stack(within).mean())
print('means outside 3/4 sigma:',
      int(((y-means[sy]).norm(dim=1) > .3).sum()),
      int(((y-means[sy]).norm(dim=1) > .4).sum()))
```

The original trajectory improves slowly after its LR floor: full covariance
error **6.036732 / 3.202252 / 2.381681 / 2.066388** at updates
**400 / 800 / 1,200 / 1,600**. All 96 original full-gate observations fail.

## Separate duration and schedule evidence

The user's existing main-checkout publications
`reports/forge/ring16-duration-20261011/results.json` and
`reports/forge/ring16-schedule-20261011/results.json` use the archived public API,
source and runtime. They are explicit ring-only diagnostics with
`qualification_input=false`. Their original baseline, budget and denominator
remain separate from ordinary Tier 1.

| R1/R2 diagnostic | Update 1,600 | Update 6,400 | First five-check passing window |
| --- | --- | --- | ---: |
| Original horizon 400; continue unchanged | FAIL: covariance 2.066388, HQ 0.844238 | FAIL: only 3 terminal passes; endpoint covariance 0.525637, HQ 0.947754, minimum eigenvalue ratio 0.221104 | 5,217 |
| Horizon 400 → 6,400 from initialization | FAIL: covariance 1.816648, HQ 0.780762 | PASS: 5 terminal passes; covariance 0.423491, HQ 0.977539, minimum eigenvalue ratio 0.157181 | 3,834 |

The longer-horizon arm matches saved initial models/prior, streams and actual
full data sequence. It also matches the serialization/reconstruction boundary
at update 1,600. Nevertheless, it bundles LR, beta2 and penalty schedules, takes
four times the ordinary update budget, and retains clock dependence. At update
3,200 it temporarily collapses to **5 modes** and HQ **0.229248**. Its eventual
PASS supports learnability of this architecture/prior; it cannot repair the
ordinary 1,600-update failure or identify a single schedule cause.

The original-horizon continuation's first five-check window and longest passing
streak **19** do not override its failed terminal suffix **3/5**. The full-horizon
arm's longest streak is **14**, with **69/384** passing observations, against
**63/384** for the unchanged-horizon continuation. Neither is a retained-quality
or broader-suite qualification claim.

## Research basis

Primary sources reviewed on 2026-10-12 support a bounded test, not a prediction
that this finite nonlinear suite must pass:

- **Keep positive zero-centered penalties; test a fixed law.** Mescheder,
  Geiger and Nowozin's Theorem 4.1 proves local convergence with R1 or R2 for
  sufficiently small simultaneous/alternating gradient steps under realizability
  and regularity assumptions near equilibrium. The theorem does not require
  coefficient annealing. Its assumptions and update law do not certify our
  from-initialization, stochastic Adam training with a learned MoG prior or
  guarantee acquisition within 1,600 updates.
  [ICML 2018 paper, §4.2](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf).
- **Retain both penalties every update.** R3GAN extends the local argument to
  paired relativistic loss, but its empirical demonstration finds R1 alone
  unstable away from equilibrium. Appendix E reports convergence failures from
  lazy penalties on Stacked MNIST and several 2D toys. Appendix D uses constant
  LR for Stacked MNIST, while annealing gamma and beta2; removing those remaining
  schedules is our new hypothesis. Full mode coverage there does not establish
  local covariance fidelity here.
  [R3GAN paper, §§2.3–2.4 and Appendices D–E](https://arxiv.org/html/2501.05441v1).
  The [pinned official trainer](https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/R3GAN/Trainer.py)
  confirms discriminator penalties on both real and detached fake samples.
- **Conditioning is a substantive later hypothesis.** Odena et al.'s Jacobian
  clamping bounds directional output/input quotients and improves average
  image-quality scores in their experiments. This motivates controlling both
  excessive stretch and lost spread, rather than minimizing Jacobian norm
  indiscriminately. Their empirical evidence supplies neither exact singular-value
  bounds nor Ring16 qualification; an adaptation would need its own public API
  implementation and declared global regularizer.
  [Generator-conditioning paper, §4](https://arxiv.org/pdf/1802.08768).
- **Extrapolation addresses game dynamics, with different assumptions.** Gidel
  et al. develop extragradient/past-extrapolation variants, including Adam, and
  evaluate them on GANs. Their convergence results assume monotone operators
  (with stronger conditions for linear rates); general neural GANs need not
  satisfy these. A later repair would need explicit moment/stream checkpointing
  and equal update/gradient-work accounting. It cannot inherit a BCAP result.
  [Variational-inequality paper, §§3–5](https://arxiv.org/pdf/1802.10551).

Fixed LR, beta2, R1/R2 coefficients and role multipliers are already supported
by the public recipe. The last two mechanisms are research directions requiring
separately scoped implementation and admission, not ready qualified recipes.

## Bounded next comparison

The first justified global trainer change is to remove all three externally
clocked schedules while keeping Adam, paired relativistic loss and both R1/R2
penalties. Set fixed LR scales, `beta2_end=null` and `reg_coeff_end=null`;
keep `reg_every=1`, all task-owned tensors/resources, learned prior locations,
fixed prior width/uniform weights, `prior_reg=0`, `prior_l2=0`, clean live sampling
and disabled added noise. Fixed start settings (base LR **0.0085**, beta2 **0.9**,
R1/R2 coefficient **1**) are a direct starting-law control; fixed terminal
settings change the starting optimizer/penalty law and require a separately
named whole recipe. Choose each recipe once globally across the complete Tier 1
suite, including the clock audit, before selecting it.

The predictions are exact step-label/horizon parity and faster component-shape
fitting without losing pole/embedding/AE or confirmed Gaussian/word acquisition.
The unchanged 1,600-update Ring16 gate, minimum spread and five terminal checks
are required. Failure of density or any incumbent pass falsifies a complete
repair, even if clock parity succeeds. A second, separately registered structural
option would address generator local stretch with a magnitude-sensitive update
or local geometry mechanism; avoid inferring that an isotropic contraction alone
will work when the radial eigenvalue is already small.

Existing negatives in [compiled memory](../EXPERIMENT_MEMORY.md) constrain the
choice: historical LR **0.00425**, coefficient **1→0.1** fails pole movement
(**0.145772 < 0.3**); LR **0.0085**, coefficient **0.1→0.01** fails pole gradient
(**1.020285 > 1**). The earlier low global repair fails movement
(**0.039444 < 0.3**). These retain their original source/runtime/init contracts;
they warn against assuming a small LR or penalty is safe, and supply no grades
for a new constant-law candidate. Historical frozen-prior and budget/noise
repairs in other formulations likewise do not establish an R1/R2 solution.

Use the exact current task declarations and public deterministic initializer at
seed 0 for new ordinary comparisons. The original baseline source must be
identified separately from any runtime/source amendment; verify matched initial
model/prior tensors and actual data batches before attributing output changes to
the recipe. Do not restore an archived provenance checkpoint as an ordinary
qualification prerequisite, lengthen the budget, substitute a diagnostic
initializer/prior, or pool the ring-only diagnostics with current grades.

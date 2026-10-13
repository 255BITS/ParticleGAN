# Prospective epsilon analysis

Saved denominator scales support **critic-only `d_eps=1e-4` on the constant
parent** as a focused, unmeasured stabilization hypothesis. They do not establish
that epsilon repairs Ring16 radial spread or any original gate. Do not launch an
epsilon grid from this readout. Await the current role campaign before deciding
whether this one additional mechanism deserves a bounded registration.

The [JSON](denominator-analysis.json) covers 12 archived terminal states and 36
actual optimizer groups: constant, AMSGrad and global momentum on Ring16,
Gaussian, words and the 80-update pole. Exact artifact SHA256s match their
published receipts, and the three relevant public binding sources match the
frozen scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
The [reproducer](analyze_denominators.py) constructs no models, makes no gradient
calls, draws or updates, and preserves Python/Torch CPU RNG states.

For each coordinate, `q=sqrt(v/(1-beta2**t))`, using `max_exp_avg_sq` for
AMSGrad. Holding saved moments and LR fixed, changing epsilon scales the last
Adam displacement by exactly `(q+1e-8)/(q+epsilon)`. The JSON reports denominator
quantiles, all-coordinate and active-moment fractions below 1e-6, 1e-5, 1e-4 and 1e-3,
the fraction of original squared displacement carried by those coordinates,
absolute original displacement, and exact same-moment L1/L2 retention.
These are diagnostic thresholds, not proposed experiment arms.

At epsilon 1e-4, the retained last-displacement L2 fractions are:

| Recipe | Task | G or direct coordinates | E | Sampled prior | D |
| --- | --- | ---: | ---: | ---: | ---: |
| Constant | Ring16 | .8592 | — | .8932 | .6894 |
| Momentum | Ring16 | .7398 | — | .7900 | .5907 |
| AMSGrad | Ring16 | .9332 | — | .9698 | .9361 |
| Constant | Gaussian | .6677 | — | .2744 | .5440 |
| Momentum | Gaussian | .8282 | — | .6893 | .8074 |
| AMSGrad | Gaussian | .8508 | — | .5964 | .8884 |
| Constant | Words | .0295 | .5664 | .9920 | .5983 |
| Momentum | Words | .0016 | .8030 | .9941 | .7705 |
| AMSGrad | Words | .6366 | .6911 | .9924 | .6663 |
| Constant | Pole | .9958 | — | — | .9436 |
| Momentum | Pole | .9957 | — | — | .9151 |
| AMSGrad | Pole | .9959 | — | — | .9423 |

The direct pole denominator median is about .023 across all three recipes,
well above the tested diagnostic thresholds. Its final direct-coordinate step
would change by only about 0.4% at global epsilon 1e-4. With **D-only** epsilon,
its direct-coordinate epsilon stays 1e-8, so the same-moment direct step stays
exactly unchanged. Critic updates and subsequent generator gradients can still
change; terminal states cannot certify the earlier 80-update trajectory or the
original sustained pole gate. Observed final mean-absolute positions are .8196,
.7966 and.8183 for constant, momentum and AMSGrad, respectively; all original
pole gates passed. Those outcomes belong to the measured recipes.

The useful critic contrast is strongest without AMSGrad:

| Constant-parent role | Active denominators below 1e-4 | Original displacement energy below 1e-4 |
| --- | ---: | ---: |
| Ring16 G | 43.77% | 10.89% |
| Ring16 prior | 0% | 0% |
| Ring16 D | 41.94% | 33.87% |
| Gaussian D | 45.56% | 55.80% |
| Word D | 59.49% | 40.47% |
| Pole direct coordinates | 0% | 0% |
| Pole D | 2.08% | .32% |

Thus D epsilon changes appreciable Ring16 critic motion while damping a much
smaller fraction of pole critic motion. This suggests a coordinate-dependent
critic stabilization mechanism rather than another uniform rate reduction.
The Gaussian and word critic effects are substantial too; passing those tasks
cannot be assumed. AMSGrad already retains large denominators: only .94% of
active Ring16 D coordinates lie below 1e-4, carrying .47% of its displacement
energy. Adding 1e-4 there changes D L2 by only 6.4%, so these states provide weaker
motivation for an AMSGrad-plus-small-epsilon hypothesis.

Tiny word-generator denominators alone are misleading. Constant and momentum
word G terminal displacement L2 is only 4.08e-16 and 1.16e-8, respectively. Their
large relative epsilon suppression acts on essentially inactive terminal
motion; it does not show impaired acquisition or harmful oscillations.
AMSGrad word G displacement L2 is 1.17e-4. No earlier denominator history or
directional link to radial/tangent error is recovered here. Small denominators
do not by themselves establish harmful amplification.

Actual public applicability is explicit:

- `Recipe.eps` supplies G/E defaults, sampled-prior defaults and direct pole
  coordinates; `Recipe.d_eps` overrides every critic group.
- `Recipe.prior_eps` overrides sampled priors in `Recipe.make_optimizers` and
  behavior-host prior groups. It does not control direct pole coordinates.
- The word fixture manually builds its prior group with LR/betas but no
  `prior_eps`. Its prior therefore consumes global `eps`; a whole-suite claim
  that `prior_eps` is honored everywhere would be unsupported on current source.
  This analysis makes no host edits or declaration to conceal that gap.

The focused prospective delta is only `d_eps: None -> 1e-4`, retaining
`eps=1e-8`, `prior_eps=None`, paired R1/R2 every update, all trainer roles and all
task/protocol contracts. It avoids repeating measured critic LR×2 or prior
LR×.25 recipes, but still lacks causal evidence of radial repair. Qualification
would require every unchanged ordinary Tier1 gate and clock under that one
global recipe.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/prior/analyze_denominators.py > reports/forge/r1r2-role-repair/prior/denominator-analysis.json
```

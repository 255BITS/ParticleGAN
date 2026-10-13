# Native least-squares review

**Defer this loss arm.** It is an existing, globally supported structural
contrast, but the saved evidence supplies no stronger density prerequisite than
the recipe-owned prior-freeze lead being reviewed. This note admits nothing and
selects no loss grid. Both closed energy recipes retain their original failures.

The possible sole delta on the fresh-source NS AMSGrad m110 control is
`loss: non_saturating → least_squares`, with **inherited** labels
`(fake, real, generator) = (0, 1, 1)`. Keep `energy_distance_weight=0`, gamma `.1`,
both R1/R2 every update, all role optimizer laws and checkpoint coverage, learned prior and every
task/protocol field unchanged. The actual law on raw scores `r,f` is
`LD = .5 mean(r−1)² + .5 mean(f)² + .05(E||∇D(real)||² + E||∇D(fake)||²)`;
scalar `LG = .5 mean(f−1)²`; joint word `LGE` additionally includes
`.5 mean(r)²` for the reversed encoder stream. Existing task-owned objectives
remain alongside these terms.

The [primary LSGAN paper's extended analysis](https://arxiv.org/html/1712.06391)
distinguishes this classification variant (Eq. 14, used in the original
conference paper) from its Pearson-divergence variant. The latter requires
`b−c=1` and `b−a=2`, which `(0,1,1)` fails. Its divergence or anti-collapse
argument therefore does not establish a density repair here. Least squares can
correct score overshoot: G seeks score 1 instead of monotonically increasing
the score; the joint E stream seeks 0 instead of monotonically decreasing it.
Its critic curvature is constant rather than logistic and vanishing at extremes.

[Mescheder et al.](https://proceedings.mlr.press/v80/mescheder18a.html) motivate
positive zero-centered penalties and prove local convergence under stated
realizability, regularity and sufficiently small-step assumptions. These results
do not qualify the fixed finite-budget AMSGrad recipe. R1/R2 operate on raw
critic input derivatives independently of `GANLoss`, so software compatibility
does hold; the unchanged numerical gamma has a different balance against the
new classification curvature.

The [saved-only probe](probe.py) and [receipt](diagnosis.json) verify the exact
fresh control revision, scientific execution `67920d909`, all 1,216 source files,
checkpoint/complete-state and saved-panel hashes. All seven task-bound recipes
differ only in `loss`: Gaussian and Ring use `GANTrainer`; pole/token/AE use the
original public component binding; words retain `joint_g_loss` and both G/E
streams; clock uses the same trainer factory. No loss branch adds data/prior
draws. No models, autograd, optimizer steps, samples or grades were added; Python,
NumPy and Torch CPU RNGs remain unchanged.

On 4,096 saved NS Ring outputs, scores range **−.181252 to .172537** and the
critic input-gradient norm averages **.169471**. LS ascent weights remain
positive, averaging **1.002511** versus NS **.500628**. Both input-force proxies
expand radial variance in only **6/16** components and contract tangent variance
in **16/16**; mean tangent drift changes from **−.0251621** to **−.0504955**.
Thus reject generator starvation or endpoint force-direction repair. This proxy
omits G Jacobians, optimizer preconditioning and future critic refitting.

The distinct, weak-confidence from-initialization hypothesis is critic
quadratic anchoring: common-shift curvature is **2** versus NS **.499260**,
and LS equilibrium scores are `.5` rather than NS logits `0`. The measured
shift gradients **−.978813** versus **.005288** use declared target centers as
an explicit stand-in fixture, not saved training pairs. They show a different
game, not a passing trajectory.

Risks remain global: word E forces reverse when real joint scores cross 0,
fake forces reverse above 1, squared outliers can produce large gradients,
and categorical softmax saturation remains. Identical zero-origin pole rows
receive identical force; LS introduces no symmetry-breaking or repulsion.
The five-terminal-check pole gate and all sustained word bounds must still pass.
Any future LS hypothesis would be falsified by one ordinary/clock failure or a
Ring terminal suffix below five, even if early acquisition improves.

The publication retains **231 attempts / 6,970.914170 paid seconds**, leaving
**1,029.085830** under the original 8,000 cap. Reuse of the exact fresh NS control
would avoid a new-source control, but the full 900-second word reservation must
still fit after peers; a complete comparison is not guaranteed. New science
would need fresh control and treatment with honest full reservations. No task
shortening, source pooling, seed study or automatic follow-up is implied.

Reproduce without training:

```sh
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-distance-repair/research/least-squares/probe.py
```

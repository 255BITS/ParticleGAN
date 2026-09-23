# Research handoff: make training robust to generator form

## User objective and branch scope

This is a **separate ParticleGAN research line**, branch `deep100gaussians`,
based on `master` at `d162e4c4d4d6ac7b93ef41882f290eb54dee085b`. Keep it separate
from develop and releases until the user asks otherwise. The provisional
HyperGAN PR #384 was closed unmerged; this research now lives here.

The objective is **a method that trains deep generators reliably across depth
and architecture changes**, not a hand-tuned rescue of this one deep MLP.
The user's working analogy is a stack of lenses: layers may amplify some
directions and suppress others until a useful critic signal produces an
unhelpful generator update. Treat that as a hypothesis to test, not a diagnosis.

The user explicitly rejected memorizing fixed latent/image pairs as the next
test. Keep this a distribution-learning problem with the learned prior and
b-cap. Do not replace the objective with per-example reconstruction.
Never run the same experiment with different seeds. Prefer metrics to pictures.
Make logs easy to tail, preserve failed results, and report measured costs.

## What is ready

- `run.py`: one-thread CPU adaptation of the native 100gaussians loop with
  depth as the structural variable and real Adam-update diagnostics.
- `recipe.json`: all resolved settings from the measured master recipe. This
  avoids silently inheriting different defaults if the research is moved later.
- `summarize.py`: validates matching identities/horizons and prints the table.
- `results/2026-09-23-native-depth/`: complete numeric results, configurations,
  provenance, interpretation, and the exact original runner source snapshot.
- `tests/test_deep100gaussians.py`: native-loop parity, diagnostic non-interference,
  common initialization, metric sanity, and archived recipe/model identity checks.

From the repository root:

```sh
python -m pytest tests/test_deep100gaussians.py -q
python experiments/deep100gaussians/summarize.py \
  experiments/deep100gaussians/results/2026-09-23-native-depth
python -u experiments/deep100gaussians/run.py \
  --output /path/to/new-experiment > /path/to/deep100gaussians.log 2>&1
tail -f /path/to/deep100gaussians.log
```

Torch and NumPy are needed; no plotting or GPU is needed for the runner. In the
current environment the tested interpreter is
`/home/martyn/dev/hypergan/training-runs/transgan-128-env/bin/python`.
That interpreter is incidental, not a HyperGAN package dependency.

Do not rerun the unchanged baseline just to start the next session. Use its
saved evidence; declare the next intervention first. The CLI above documents
reproduction, not an instruction to run another unchanged experiment.
There are no trained checkpoints; a continuation would need explicit checkpoint
support. Changing `--steps` changes the annealing horizon, not just truncation.

## Completed evidence

All cases: same 100-Gaussian target, learned 20,000-particle table, Fourier-2 D,
RpGAN logistic, every-step exact b-cap, Adam, EMA, original seed1234, 7,000 steps.
G width stays128; the hidden-layer count is3/8/16. Shared G tensors, D/prior
initialization, training streams, and monitoring inputs are matched.

| Hidden layers | EMA modes /100 | EMA HQ | EMA sliced W1 | CPU seconds |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 100 | 98.94% | .1582 | 74.5 |
| 8 | 85 | 49.29% | .2252 | 92.1 |
| 16 | 35 | 28.49% | .4363 | 119.8 |

HQ is mass within3 target standard deviations of a center. All20,000 monitoring
IDs and the real bank are fixed. Sliced W1 is a distribution proxy, not exact W1.
HQ alone can reward variance compression; a serious success claim should add
the existing within-mode calibration metrics from ParticleGAN's research tools.

Sixteen layers initially have156× less output variation than three layers. The
first hidden activation receives73× less gradient on the first update although
the output gradient magnitude is similar. The learned prior's adversarial
gradient also passes through G and is initially attenuated. Later gradients
recover: initial attenuation is not an explanation of the entire trajectory.

The deepest first-step request is already99.39% shared energy, versus98.82%
shared energy in realized output motion. Therefore this observation does not
show G corrupting differentiated requests. The shallow successful run also has
88.29% shared first-step movement. Neither shared fractions nor good cosine
alignment alone is a success/failure score.

Deeper models improve late during annealing. Say **worse learning within the
same budget**, not irreversible collapse. Adding layers also changes the initial
function and parameter count; there is no function-preserving depth control yet.

## Why prior and b-cap do not settle the question

A learned prior changes where inputs are sampled; the network remains continuous.
It relieves some distribution-mapping pressure but does not make the map or its
optimizer invariant to architecture. Prior updates themselves depend on `dG/dz`.

B-cap is `relu(norm(grad_x D)-kappa)^2`: a soft penalty on overly steep critic
gradients at sampled points. It does not enforce a nonzero slope, its semantic
usefulness, or any bound on the generator's response. Do not equate zero penalty
with either a missing gradient or a solved adversarial game.

For fixed latent inputs and critic, let `X=G_theta(Z)`, `q=dL/dX`, and
`J=dX/dtheta`. A small plain-SGD step gives
`delta_X ~= -eta * J * J.T * q`. This is a local chain-rule identity, not an
infinite-width assumption. Shared parameters couple samples. Adam's state and
finite steps alter the response, so the runner measures actual movement.
Frozen-state descent is still not a guarantee of distribution improvement.

## A route toward a general method

The promising abstraction is **control the effect of an update in output space**,
using the generator's measured response, rather than assigning a magic rate to
a particular layer name or depth. There are several distinct problems to separate:

1. **Bad initial signal propagation.** Use a function-preserving depth insertion
   control (e.g. identity-initialized residual blocks) to separate changed initial
   outputs from a deeper parameterization. This is a structural control, not
   evidence that residual blocks are a universal fix.
2. **Excessive overall update gain.** A damped step rule based on measured output
   displacement/local linearization error is an architecture-independent candidate.
   A tiny step that learns nothing must fail the distribution-progress criterion.
3. **Unequal gain across directions.** A scalar learning-rate reduction cannot
   repair severe directional imbalance. A local, damped least-squares update
   `min_delta ||J delta + alpha*q||^2 + lambda*||delta||^2` is a useful reference:
   can matching the requested movement help? JVP/VJP products permit matrix-free
   approximations; the tiny toy also permits exact checks on small batches.
   This is a research candidate, not a validated production optimizer. Damping,
   rank deficiency, finite-step validity, compute cost, and critic error matter.
4. **Adversarial dynamics.** Even a faithfully realized request can be unhelpful
   because D is wrong or the alternating game is unstable. A successful local
   response diagnostic cannot exonerate that interaction.

An output-space reference update can temporarily move a fixed batch of generated
points directly along the same frozen critic gradient, then compare with G's
realized movement. That is a diagnostic of parameterization, not a replacement
generator and not supervised fitting to assigned targets. It also cannot prove
that the critic's direction improves the target distribution.

Keep G-only and prior-only effects distinguishable before trying joint control.
The existing report already separates G-only from combined G/prior movement;
it does not yet compute a standalone prior-only counterfactual or Jacobian spectrum.

For a general solution, choose a rule on a declared development case and freeze
it before testing other depths/forms. Require preservation of the shallow control,
improvement across more than one deep form, quality/coverage/calibration gains,
and acceptable wall-clock overhead. Report equal-update and equal-time comparisons.
Do not fit a different rule for each depth and call the result robust.

## Connection back to images

The original motivation was a working32px TransGAN recipe and collapsing larger
logos generators. Image diagnostics showed growing shared residual activations,
then tanh saturation suppressing visible latent variation. The current toy has
plain LeakyReLU layers and an unbounded output; its initial problem is contraction.
We have reproduced depth sensitivity, **not established the same failure mechanism**.

After the basic response test, bridge one structural feature at a time toward
the image generator: normalized residual blocks, bounded output, then token/spatial
structure if necessary. Keep a successful simple control throughout. Do not
extrapolate a repaired2D MLP directly to a128px GAN.

The migration ran validation tests only; the archived three7k experiments were
not rerun. Original source hashes and paths remain unchanged in provenance.
The old measured runner is included so its implementation can be inspected
without recovering the former HyperGAN research branch.

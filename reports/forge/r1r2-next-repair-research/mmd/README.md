# Fixed RBF mixture: saved-state prerequisite, not a new experiment

**Decision:** a finite-scale MMD auxiliary is a credible, low-confidence research
lead. The analytic prerequisite passes, but it does not establish a complete
Tier1 repair or justify paid admission. The remaining **831.751939 seconds**
cannot cover even one unchanged 900-second word reservation. No candidate, API
field, gate, task, queue entry or budget was changed **at this research receipt's
original scientific source 2b04**. Later reversible API preparation is a separate
software cohort and receives no trained evidence or qualification from this note.

Read `EXPERIMENTATION.md`, compiled experiment memory and the closed energy and
prior-freeze readouts first. Energy broadened Ring beyond its covariance gate;
freezing the prior retained that failure. Their saved output laws motivate a
finite length scale, rather than another optimizer/prior/gamma grid.

The single mathematical law checked here is the equally weighted Gaussian kernel
mixture with **standard deviations (.0625, .125, .25, .5, 1, 2, 4)** in original
task-native output units. Its hypothetical addition has coefficient 1 to the
fresh non-saturating AMS m110 parent with energy weight 0 and learned prior.
Coefficient 1 and these widths are a finite hypothesis, not an inferred optimum.
There is no unit-invariance claim or task-dependent normalization.

For existing generator outputs X and detached existing targets Y, use

`L = mean(i != j) k(x_i,x_j) + mean(i != j) k(y_i,y_j) - 2 mean(i,j) k(x_i,y_j)`.

Differentiate both fake arguments; remove only diagonal indices, retaining
different rows with identical values. Preserve the signed squared-MMD U-form:
no square root, clamp, jitter, additional draws, component labels or balancing.
Both R1/R2 penalties, the original adversarial and paired reconstruction/hold
losses, native AMS histories and all task contracts remain necessary.

[Gretton et al.](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
derive the fixed-kernel U statistic, its possible negative estimates and
distribution identification by characteristic kernels. Fixed positive Gaussian
mixtures inherit that population identification; this does not give a finite
training convergence guarantee. [Li et al.](https://proceedings.mlr.press/v37/li15.pdf)
support equally weighted multiscale Gaussian moment matching for generators;
their published biased/square-root objective is different from this signed U
law. These papers do not validate our coefficient or seven widths.

A target-batch median was rejected for this prerequisite. The token host's eight
targets are identical, giving median zero. Selecting the bandwidth from the same
target batch also forfeits unbiasedness for a fixed population kernel; detaching
that bandwidth does not restore the statistical guarantee. The
[median-heuristic analysis](https://arxiv.org/abs/1707.07269) does not identify a
universally optimal training scale. Fixed widths avoid those two problems while
remaining sensitive to original coordinate units.

## Exact saved Gaussian-law calculation

Each of the three endpoints uses **all 4096 original saved Ring outputs**, with
empirical fake/fake U pairs. Cross and real/real terms integrate the **original
uniform 16-Gaussian law**, radius 3 and covariance .01 I, analytically. For kernel
variance s², its target expectation at x is

`mean_c (s²/(s²+.01)) exp(-||x-mu_c||²/(2(s²+.01)))`.

The real/real expression replaces .01 by .02 and averages all center pairs.
Raw-energy forces integrate the identical Gaussian target through the exact
two-dimensional Gaussian-norm mean derivative, using scaled Bessel I0/I1.
They do not replace Gaussians by mean points or use target draws.

Gradients are n times derivatives of the mean objective. The table reports
first-order mean radial/tangent variance drift under **output-space descent
alone**; component labels are diagnostic axes only. Negative means contraction.

| Saved output law | MMD radial / tangent | Energy radial / tangent | MMD radial expanding / tangent contracting modes |
|---|---:|---:|---:|
| Fresh NS parent | +.001812 / -.002084 | +.000880 / -.001425 | 16 / 11 |
| Energy, learned prior | -.001050 / -.004195 | +.000261 / -.003715 | 5 / 14 |
| Energy, frozen prior | -.000197 / -.014913 | -.000220 / -.012219 | 7 / 16 |

The learned-energy endpoint supplies modest evidence of a restoring scale:
MMD contracts mean radial width where energy still expands it. Both expand five
individual modes there. At the NS endpoint their direction counts coincide;
at the frozen endpoint both already contract mean widths. These results do not
show MMD will prevent overdispersion. Native NS force on its own saved endpoint
has mean drifts -.000321 / -.025162, expanding radial width in 6/16 modes.

This is population-target diagnosis with a large saved evaluation panel. The
actual independent 128-row Ring training pairs were not archived. Future native
minibatch forces would retain random mode-mass differences; no exact coupling or
equal-mode target sampling may be substituted. No G Jacobian, parameter sharing,
AMS preconditioner, future critic, assignment-boundary changes or historical
gradient is reconstructed. Original six bounds and terminal suffix stay closed
under their original evidence; this calculation does not rescore them.

## All-host risks and remaining prerequisites

| Original host | Preserved data/output contract and risk |
|---|---|
| Gaussian | Existing 1D sampled targets/prior outputs; finite scales can miss tails. Passing acquisition does not imply the separate endpoint KS bound. |
| Pole | Twelve zero direct coordinates versus deterministic balanced +/-1 with +/- .05 spread. Smooth even kernels give zero auxiliary force at exact symmetry; native adversarial force must still move them. |
| Token | Eight repeated concept outputs versus repeated [0,1]. The finite kernel gives a concept direction; shared parameters also move the unused slot, so its original matched hold cannot be replaced. |
| AE | Existing unconditional 2D generated batch versus sampled two-anchor Gaussian data. Marginal matching cannot certify paired inverse reconstruction; keep its existing reconstruction branch. |
| Ring | Existing independent Gaussian-mixture batches, 128 rows, unchanged .1 target widths and prior sampling. Width sensitivity competes with stochastic component-mass error and center displacement. |
| Words | Existing five atomic 6x28 one-hot targets sampled with replacement, categorical-softmax generated probabilities, five-row prior and joint inverse encoder. Use probability coordinates only, preserving the joint adversarial inverse stream. Duplicate atoms remain in U sums. Saturated logits can eliminate parameter gradients despite finite output force. |
| Clock | Its own original scalar public-API cohort, warmed branches, state restore and RNG/budget checks remain required; a new output term needs actual consumer/state coverage. |

The explicit fixtures retain a negative matched U value (-.024771), near-zero
symmetric pole force (9.32e-19), and token shared/concept force [0,-.409840].
At word logit gap 80 the output-gradient norm is .016920 but the manual softmax
logit-gradient norm is 1.41e-36. This illustrates an unresolved conditioning
limit, not trained word behavior. Empirical MMD, Gaussian-target MMD and exact
Gaussian energy gradient finite-difference errors are respectively 4.25e-11,
3.94e-11 and 3.16e-10, below the declared 2e-9 bound.

There is currently **no output-MMD public recipe field**. The latent
`prior_regularizer='initial_mmd'` is a different objective. Any later integration
needs separate all-host consumer, target-detachment, zero-path, signed-state,
strict resume, original-source pin, initialization/RNG and resource prerequisites,
then legitimate fresh-source controls. The falsifier for this narrow mechanism
is persistent Ring covariance/HQ/suffix failure or a lost original host pass;
endpoint force signs alone cannot qualify it. Keep this as a research lead,
with no automatic source addition or paid comparison under the closed cap.

`probe.py` and `diagnosis.json` bind source digest
`2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12`,
all 1216 original science files, exact publications/checkpoints/complete state
digests and saved output hashes. No models, autograd, draws, optimizer updates or
qualification mutations were performed; Python/NumPy/Torch RNGs are unchanged.

Reproduce from the isolated report worktree:

```sh
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-next-repair-research/mmd/probe.py --source-root /mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue/snapshots/2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12 > /mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012/mmd-probe.log
tail -f /mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012/mmd-probe.log
```

The archived stdout is a mathematical receipt, not a live experiment log or a
new leaderboard.

The first read-only execution verified all 1216 then-live files before any MMD
integration. Its immutable archived receipt is `mmd-probe-before-integration.log`,
SHA256 `87279753d26cf0bf33cf7a709e5beda779d33cae5b6d03cb0ff1b8d593c0582a`.
Reproduction after later core changes verifies the original B/E scientific
snapshot via `--source-root`; it never attributes these saved panels to changed
HEAD. This flag changes verification location only, not calculations or inputs.

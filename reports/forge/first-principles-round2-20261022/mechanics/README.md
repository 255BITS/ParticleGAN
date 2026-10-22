# Same-batch normalized G/prior secant correction

Candidate `bispf-secant-extragradient-v1` changes **only**
`bispf_generator_step=secant_extragradient` relative to the output-energy winner.
It retains energy weight 1, learned prior locations, fixed task-owned initial
prior law/capacity/kernel widths, SBNP coefficient .01, critic update, thermal
gradient hook, DualNorm rates/smoothing/quintic convolution and final original
anchor DirectionBlend protection. Sampling, data, initialization, gates,
cadence and committed update budgets do not change. The parent study owns
declarations, one current comparison and the common frozen-source execution.
No scientific training was launched from this worktree.

The [completed first round](../../three-theme-repairs-20261022/README.md)
measured energy at 6/15 PASS against 4/15. It restored rare support but not
sustained rare-axis shape, width or overlap mean. At rare checks 1050/1100/1150,
the sole failed rare bound was minimum eigenvalue ratio (.129/.030/.064 against
.15). That motivates testing finite nonlinear G/prior overshoot after support
recovery. It does **not** establish overshoot as the historical cause: original
training-batch force tensors were not saved. This candidate changes parameter
integration rather than output force, pullback metric or density law.

## Defined operator and finite work

After the ordinary D update, freeze D. Let x contain G and learned prior
parameters, and L_B be the unchanged combined generator objective on the
existing G-phase real batch, sampled row IDs and fixed MoG perturbations.
Let N_x(g) be an ordinary **unprotected** normalized optimizer proposal from
the complete anchor optimizer state, including actual rounded parameter writes
and existing sampled-row ownership. Define

```
p0 = N_x(gradient L_B(x)) - x
y_a = lerp(x, x+p0, a)              # retain actual proposal endpoint tensors
p_a = N_x(gradient L_B(y_a)) - x    # restore anchor before normalizing
a ||p_a-p0|| <= .5 ||y_a-x||
```

Take the first accepted a in 1, 1/2, ..., 1/256. The right side uses the actual
rounded predictor displacement. Apply the original anchor DirectionBlend
protection to the accepted predictor gradient's normalized proposal, then
interpolate that protected endpoint by a. Thus the certificate describes the
**pre-protection normalized field**, not a guaranteed final projected-field
contraction. If all nine candidates fail, leave G/prior bits unchanged and
commit their optimizer clock once. Zero proposal skips trial forwards. A
nonfinite anchor loss/gradient/proposal aborts and restores the whole public
step; nonfinite trial values consume their bounded trial and are rejected.

Each trial uses the same real tensors, critic state and latent perturbations;
there are no additional data or sampling draws. A sampled prior row's replay
input is its original latent plus current-minus-anchor row location. Each
probe restores model/optimizer/controller/record/stream/public checkpoint
state, pending sampled rows, gradients, model modes and transient policy state.
Only the final optimizer operation advances G/prior clocks and local/global G
step hooks. Interpolation preserves exact endpoints at zero/one, including
untouched signed zeros. Returned stats measure actual movement after policy
completion, raw/protected proposal norms, protection change and the secant
bound. Parent execution records those returned stats after the actual step.

The cost bound is nine extra G/critic forward-backwards and transient normalized
proposals per G phase, with whole-state copying/restoration overhead. The
existing task wall-time caps still apply. No tuning of trial count, margin,
seeds or task-specific parameters is allowed. Known stochastic/running-buffer
models and incompatible policies/noise are rejected before execution. Unknown
RNG consumption or model-buffer mutation fails closed during replay; immutable
Fourier buffers are allowed. Public state rollback cannot undo an earlier D
optimizer hook's external write or caller callback side effects. It is not an
atomic transaction over external observers.

## Rejected first operator: protection hides reversal

The initial pre-admission definition put fixed-anchor DirectionBlend **inside**
N for both secant proposals. The unchanged convex quadratic software fixture
falsified it: the bias norm grew from 1.4142e-5 to 2.8261e-5 although its
consistency test accepted a=.25 (third trial). This is a mathematical failure,
not merely floating-point noise, and is retained as negative evidence.

In the smoothing limit, raw P(x)=-c*x, y=(1-a*c)*x. When y<0 the raw predictor
force reverses; anchor protection reflects it to -c*abs(y)/2. The old test
accepts a*c in [2,4], but its final iterate is
`x[1-a*c*(a*c-1)/2]`. At a*c=3 it gives -2*x while the secant difference is zero.
The sole pre-admission operator revision excludes protection from both secant
proposals and retains it once at final application. Constants and fixed
controls were unchanged. No further revision or scientific threshold search
was used to obtain admission.

## Fixed controls and limits

[controls.json](controls.json), reproduced by [controls.py](controls.py), is
an explicitly separate public-API software cohort, with deterministic seed 0.
Its overridden analytic loss is not a learned-quality task or gate result.

- The unchanged quadratic fixture (smoothing .001, rate .012, initial bias
  [1e-5,1e-5]) accepts a=1/32 after six trials. Its norm contracts by
  .765620172; lhs 1.98814e-6 <= rhs 2.65139e-6. Protection is inactive here.
- For L(x,y)=xy with frozen opponent y=1, the G field is constant. The public
  implementation takes a=1, secant lhs zero and exactly matches ordinary G
  and prior tensors. It provides **no correction of joint bilinear game
  circulation**. Its hypothesis is nonlinear frozen-critic G/prior variation.
- Opposed and nonfinite trial fixtures reject after nine trials, preserve
  exact parameter bits and commit only one clock. Null force remains exact.
- Public checks require identical consumed streams and real-callback/sample
  counts to ordinary training, single optimizer hook notification, exact full
  checkpoint continuation, full public-state exception rollback, rejection of
  stochastic/mutable models, static-buffer support and finite-anchor guards.

All 64 mechanics/allocation/settling/consumer checks passed in 2.02 seconds;
75 existing trainer/Forge API/field-boundary checks passed in 2.18 seconds.
The compact numerical reproduction used 1.17 CPU seconds. These checks and
their limited reruns remain below the assigned 120 CPU-second screen allowance.
They establish implementation behavior, not task-level repair, stability of a
nonconservative game, or Lyapunov descent. Learned-quality evidence must come
from the complete matched panel and its actual-training GIFs; no software
fixture silently substitutes for that denominator.

## Prior art, distinction and scientific falsifiers

[Same-sample stochastic extragradient](https://proceedings.mlr.press/v108/mishchenko20a.html)
motivates reusing a sample to approximate implicit updates, while noting the
extra iteration cost. [Malitsky's operator-based line search](https://arxiv.org/abs/1601.04001)
motivates a local secant check; its monotone-VI guarantees do not apply to this
normalized, protected GAN implementation. [Gidel et al.](https://arxiv.org/abs/1802.10551)
place adversarial optimization in the VI framework. Even
[nonmonotone EG theory](https://arxiv.org/abs/2210.13831) requires structural
assumptions not established here. This G-only replay is not joint-player EG.

The [compiled negative evidence](../../EXPERIMENT_MEMORY.md) and
[BCAP three-phase audits](../../bcap-three-phase/README.md) already reject raw
historical optimism (PR378: stale gradients across moving critics and sampled
prior rows), bounded historical secant scaling (PR381: floors dominate and
local density still fails), and universal claims from normalized gradients.
This proposal uses current same-batch predictor gradients and realized
unprotected normalized proposals, no cross-update history, no positive scale
floor and no D lookahead. StrictProgress protects finite original adversarial
losses with objective probes on conflicts; this secant rule runs on every G
step and does not promise finite adversarial descent. Hydraulic output-travel
controls and [round-five CG4](../../bcap-physics/round5/README.md) remain negative
for global transfer; this method uses no output-radius cap, Jacobian inverse,
force filter or balanced allocation. These distinctions justify one bounded
comparison, not expected success.

On the unchanged full 15-task panel compare against a **source-matched energy
control**, and retain the original BISPF control separately. Primary scientific
prediction: rare component minimum-eigenvalue and overlap translation bounds
retain quality over the required terminal checks, while energy's anisotropic
and blobs gains and all four original passing controls survive. Width,
Gaussian stability and native image gates remain in the same denominator.
An endpoint-only improvement does not pass a sustained gate. A lost existing
gate, inability to preserve matched streams, many zero-step rejections, or
no improvement in the targeted sustained bounds falsifies a global repair.
Secant acceptance, smaller step norms or prettier images alone cannot rescue
that verdict. No task-specific combination, changed gate, budget extension,
seed sweep, automatic successor or default adoption follows this screen.

Reproduce software fixtures from the repository root:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/test_bispf_mechanics_replay.py
OMP_NUM_THREADS=1 python reports/forge/first-principles-round2-20261022/mechanics/controls.py \
  --output /tmp/bispf-mechanics-software-controls.json
```

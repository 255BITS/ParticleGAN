# Four-step DDGAN memory scout

Status: **cancelled before completion, superseded by the core real-memory
handoff experiment at the user's request.** CPU tests and CUDA smoke passed;
there is no completed substantive DDGAN result or final training checkpoint.
Retain this implementation as an exploratory option, not evidence of a better
formulation. Do not resume or promote it before the core handoff is tested.
The stopped queue's preserved log is:

```sh
tail -f runs/memory_path/scout_round4/train.log
```

Run configuration: `experiments/configs/memory_ddgan/gru_ddgan4.json`.
Trainer: `experiments/memory_ddgan_scout.py`.
Incomplete output: `runs/memory_path/scout_round4/runs/gru_ddgan4`.

## Formulation

One persistent 32-value GRU memory belongs to D. There is no private recurrent G
state. A fixed learned particle z is reused through the entire trajectory,
including every inner denoising call. At each trajectory step, Gaussian noise
initializes a temporary 2D diffusion state. Four clean predictions and public
`DDGAN.reverse` calls produce the emitted point. Memory stays fixed throughout
those four calls, then receives exactly one write containing the final point.
Both deployment memory and training rollout memory start at zero; there is no
real-time expert, real prefix, analytic projection, or supervised circle loss
in autonomous generation.

D owns a path head and a transition head, with one shared writer. The transition
G reads state computed from a real prefix. D recomputes numerically equivalent
prefix state when scoring each transition pair. Autonomous G uses its own
zero-initialized rollout state updated by that same writer; the path head replays
the trajectory to obtain its states. Thus writer parameters and state values
are shared/equivalent at corresponding prefixes, but one live mutable memory
object is not handed from G rollout into D scoring. G receives the
weighted mean of their adversarial losses. Default weights are one each. D
receives the same weighted mean of their adversarial losses and B-cap penalties.
This keeps the aggregate loss scale comparable, but divides the relative weight
of the existing path objective by two. It is not a pure architecture-only
comparison against the one-shot baseline.

- Path objective: the existing 64-point flat critic scores full real or fully
  generated trajectories. G backpropagates through every generated point and
  memory update; only D updates the writer's parameters.
- Transition objective: uniformly select one real trajectory position and one
  diffusion timestep per batch entry. Build M from points strictly before that
  position. The real candidate is the coupled forward sample x_(k-1); the fake
  candidate is `reverse(G(z,M,x_k,k),x_k,k,eta)`. Real and fake share the exact
  same real-prefix memory condition, noisy x_k, and timestep. There is no
  current-point or future-point leakage into M.
- Transition B-cap: differentiate candidate coordinates only, holding the real
  prefix, noisy point, and timestep fixed, following `docs/api.md`. Memory is
  recomputed inside D, so adversarial writer gradients are preserved. The
  candidate gradient graph also supports exact higher derivatives.

The schedule is explicitly `[1, .9, .5, .05, .0001]`. Terminal and posterior noise
are fresh Gaussian draws for every emitted point; diffusion noise is separate
from learned particles. The last reverse step has zero posterior variance and
emits G's clean prediction. Fresh noise may make persistent direction and circle
identity harder to maintain; this scout tests that issue rather than suppressing
noise or adding a second persistent state.

## Protocol and validation

The scout uses the public DDGAN recipe with scalar conditioning, no class labels,
512 learned particles, batch 128, length 64, and 2,000 updates on the same fixed
10,000-update learning-rate schedule as prior scouts. Optimizer, loss, particle
regularizer, and exact autograd B-cap use public defaults. No clipping or EMA.
G initialization is isolated from D, writer, and prior initialization. This is
one configuration, not a seed sweep. The added diffusion inputs and head mean
it is not identical in parameter count to the one-shot baseline.

On successful completion, saved artifacts include input and resolved config, source snapshots and hashes,
training logs, full optimizer/model/RNG checkpoint, trajectories, and metrics.
Resume checks architecture and training settings and restores all RNG streams.
Evaluation uses the same circle metrics, early-fit criterion, 256/1,024 lengths,
zero/shuffled memory interventions, and first 128 particles as earlier scouts.
Baseline and memory interventions reuse the same diffusion noise stream.
One diffusion-noise realization is evaluated; it is not a sweep over seeds or a
held-out particle split.

Five CPU tests passed: fixed z and memory through four inner calls; exactly one
write per emitted point; late-to-early feedback gradients and writer ownership;
strict-prefix conditioning; active conditional exact B-cap; noise replay and
final clean transition; and exact resumed training. A two-update CUDA smoke with
full evaluation passed in 2.52 seconds. Smoke artifacts are under
`runs/memory_path/scout_ddgan_smoke_round4` and are not scientific results.

The substantive process was terminated without final metrics. No scientific
conclusion is drawn from intermediate losses or the smoke run.

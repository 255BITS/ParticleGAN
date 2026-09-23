# Gym world model: three-generator Lunar Lander

Status: implemented on `feature/gym-world-model`. See [the implementation guide](gym-world-model.md)
and the new [baseline board](../reports/gym/lunar_lander/baseline/README.md).
User selected Lunar Lander and explicitly wants to build on the three-generator
solution. The design below records the agreed initial protocol.

## Objective and first demo

Learn individual state/action/next-state records, then compose predictions into
an animated lander responding to engine commands. Test whether joint generation
helps conditional prediction as well as producing plausible transitions.

Use Gymnasium `LunarLander-v3` with `continuous=True`, default gravity, and wind
disabled. Observations contain position, velocity, angle, angular velocity, and
two leg-contact indicators (8 coordinates); actions contain main and lateral
engine commands (2 coordinates). Render from numerical observations rather than
learning pixels. Continuous engines have activation thresholds, so command
coverage and evaluation must distinguish inactive and active regions.
[Environment documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

The first visual payoff is a real lander and a learned ghost receiving identical
commands: falling, firing the main engine, rotating, approaching the pad, and
making contact. A landing scene driven by a reference controller demonstrates
prediction. Landing under a controller using the learned model is a later,
separately evaluated milestone.

## The three-generator solution is the main model

```text
Same sampled z from a learned 1,024-component MoG:
G1 -> st       # lander observation: 8 coordinates
G2 -> at       # main and lateral engine commands: 2 coordinates
G3 -> st+1     # successor observation: 8 coordinates

Observed input:
E(st, at) -> z_hat -> G3 -> predicted st+1
                    G1 -> reconstructed st
                    G2 -> reconstructed at

Synthetic composition:
G1 -> st --+
G2 -> at --+-> E(st, at) -> z_hat -> G3 -> st+1

Discriminators:
D_joint(st, at, st+1)
D_action(at)
D_state(st, current_role)
D_state(st+1, next_role)   # shared state-critic weights
```

Each G is an independent MLP. Preserve coordination through the same MoG draw,
including its noise. G3 continues to take a latent code; do not replace the main
architecture with a direct `(st, at) -> st+1` network. Such a network is only a
supervised comparison. G2 learns the sampled action distribution; it is not yet
a policy. Chosen controls enter the prediction path through E.

Provide the observed ground polyline as fixed-size context to every G, E, and D.
Extract and normalize terrain geometry from the simulator with a version-tested
adapter. This is additional privileged scene information beyond the standard
8-coordinate observation; it allows the rendered ground to influence predictions.
Never provide an episode ID, future state, future reward, or future contact.
Group evaluation by unseen episode/terrain. State this context augmentation in
results rather than claiming an unmodified observation-only benchmark.

Carry over MoG1024, z_dim32, width128 G/E, width256 joint D, width128 marginal Ds,
particle_ae routing and bounded offset, EMA, Rp logistic, bcap, optimizer groups,
and prior spread. Calibrate the fixed prior sigma by the existing procedure.
Remove the old route-class gain and time inputs. Use training-only normalization,
with one common state scaler across the two roles. Include a current/next role
label in the shared state D: sharing weights does not require equal marginals.
Keep the joint loss plus the mean of three marginal roles.

Retain real triple reconstruction and synthetic state/action reconstruction,
with the existing detached synthetic targets and live input gradient path.
For dimensions 8/2/8, normalize losses by role rather than giving large blocks
more weight solely because they have more coordinates. E never receives st+1.
The simulator supplies data and evaluations; it is not differentiated through.

Six state coordinates have continuous outputs; the two contacts have logits.
Use standardized continuous MSE and mean contact BCE with separately logged,
explicit weights, identical across learned comparison arms. Average the three
real reconstruction roles; average two synthetic state/action roles. Bound G2's
commands with tanh. Define contact sampling consistently before benchmarking:
use hard Bernoulli samples in generated adversarial records, with a documented
straight-through gradient for G updates, and retain probabilities for calibration
metrics. This prevents D winning solely by distinguishing binary real contacts
from soft fake contacts. Synthetic contact targets are detached sampled bits.

## Simulator complications we must handle explicitly

The implementation applies random engine dispersion even with wind disabled.
Its Box2D world also contains articulated legs and contact/sleep state not fully
represented in the observation. Terrain is generated on reset. Consequently,
we should treat the first conditional path as an approximate observation predictor,
not assert deterministic, fully observed dynamics.
[Simulator source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py).

Preserve standard engine randomness in the main dataset. The current deterministic
E -> G3 path makes continuous point predictions; the prior sampler can model a
joint distribution, but that alone does not establish calibrated conditional
uncertainty. A later explicit conditional-noise extension would be a named
architecture change. Contact probabilities do not solve all hidden-state effects.

Do not restore the simulator by assigning its eight observations to a body.
For counterfactual actions, reset to a recorded episode seed and replay the saved
prefix, restoring the full world through simulation. Validate reproducibility
including contacts and RNG behavior in the installed version. A snapshot shortcut
is allowed only after equivalence to replay is demonstrated. These prefixes are
collector/evaluator metadata, never training sequences or model inputs.

Preserve terminated and truncated separately and stop stepping after either.
Time limits are environment-wrapper bookkeeping, not physical crash labels.
A learned reward/termination model is outside the first architecture. Recursive
prediction is evaluated up to reference termination, with lengths/counts reported;
longer autonomous episodes require an explicitly designed stopping rule/model.

## Finite dataset and action coverage

Start with a fixed mixture of reference heuristic, perturbed heuristic, and
exploratory continuous controls. Collect both approaches and failures, including
low-altitude and contact transitions. No policy training is needed to collect.
Freeze mixture weights and phase sampling after a small collection pilot, before
full model runs; record both natural visitation and selected phase counts.

Target budget: 32,768 training triples (8,192 anchor states with four actions),
4,096 validation triples, and 8,192 test triples. Split whole source episodes
before selecting anchors or branching actions. No prefix, sibling action, terrain
instance, or episode crosses splits. Use multiple fixed dataset episodes; this is
not permission for repeated training experiments that differ only in random seed.

For each anchor, use replay to obtain four alternative commands. Stratify main
engine off/on and lateral off/left/right cases across anchors, including combined
firing and command magnitudes. Include the behavior command where appropriate.
Store parent episode, anchor step, command, terrain, observation, successor,
reward, flags, and replay provenance. Stop anchors before terminal states.
The collector traverses episodes; the trainer only sees shuffled single records.

Measure replay cost on a small pilot before committing the dataset build. If the
proposed collection budget is too costly, revise and record the protocol before
training rather than silently calling unrelated states matched-action examples.
Count simulator calls, unique training triples, and repeated optimizer draws
separately. All learned arms receive exactly the same finite data.

Freeze additional held-out counterfactual probes using identical replayed worlds
and shared pre-step RNG state for alternative commands. They measure action
response under coupled random disturbances. Repeat a small evaluation-only noise
probe from the same physical world with varied engine RNG to characterize outcome
variation; report it separately from model error, not as a proven Bayes-error
floor. This is a simulator diagnostic, not a seed-repeat training experiment.

## Bounded first comparison

| Arm | Purpose |
| --- | --- |
| Persistence: predicted st+1 = st | No-training sanity baseline |
| Direct supervised predictor | Practical conditional-prediction comparison only |
| Three Gs + E + prior, reconstruction only | Same generative graph without adversarial losses |
| Three Gs + E + joint/marginal Ds | Main model, adapted from our winning recipe |

Match the full and reconstruction-only arms' G/E/prior capacity exactly. Both
retain real and synthetic reconstruction and prior spread. Use the same next-state
supervised loss in the direct control; choose its size before runs to approximate
E+G3+prior parameter count. Report inference-path and full-model counts separately,
along with compute/draw differences. This is not an exact compute match.

Start width128, batch256, and 10k updates per learned arm with the MoG schedule
adapted to that budget. Save 1k/2.5k/5k/10k checkpoints. Run correctness and timing
smokes, then three learned arms sequentially. Aim for minutes per training run,
with simulator collection timed separately; do not promise a runtime until measured.
Use validation for checkpoint selection and also report final checkpoints.
No automatic 28k extension, width sweep, or seed repeats.

## New leaderboard

Primary: standardized next-state MSE over the six continuous observation fields
on held-out real (st, at, terrain), with fixed training-only scales. Report position,
velocity, angular errors, p95, and improvement over persistence. Keep contact
Brier score, BCE, and precision/recall separate so rare contacts cannot disappear
inside aggregate MSE. Report flight, approach, and contact-transition subsets.

Required secondary results:

- Action response on matched anchors: predicted changes between engine commands
  versus simulator changes; separate threshold crossings and inactive regions.
- Recursive prediction at 1/5/20/50 steps under identical saved commands, without
  refreshing real observations. Report drift, survival counts, and contact errors.
  Use probabilities for one-step scoring and a disclosed fixed contact decoding
  rule for recursive feedback, shared by learned arms. Never snap the ghost to
  the ground or silently repair a predicted crash.
- Original and composed joint SW1/coverage, conditioned on fixed held-out terrain
  contexts with enough reference samples per context. Record context weighting.
  Score contact-combination frequencies and phase coverage alongside distances.
- G1/G2 reconstruction, routing/offset usage, nonfinite predictions, out-of-range
  states, training time, and inference throughput.

An arbitrary generated observation cannot be uniquely installed as a Box2D world.
Therefore do not claim exact simulator-consistency scores for all generated
triples. Use replayable real anchors for physical conditional checks; use
reference-distribution metrics for unrestricted prior samples. Any later canonical
world reconstruction is an explicitly approximate diagnostic.

Freeze evaluation references/definitions and pin run provenance on a new Lunar
Lander board. The old route leaderboard remains unchanged. If generation improves
but supervised prediction wins, report both. Good-looking motion alone does not
establish useful action response or successful model-based control.

## Demo progression

1. **Engine what-if:** choose a recorded airborne/approach/contact anchor, vary
   main and lateral commands, and compare predicted next-state ghosts with
   replayed simulator outcomes. Show thrust, velocity, and rotation clearly.
2. **Real versus learned:** synchronized landers under the same recorded commands,
   with explicit one-step and recursive modes, model/baseline toggle, and error
   trace. Show the actual terrain. Predicted leg poses are schematic: two contact
   bits do not recover articulated leg geometry. Do not hide divergence.
3. **Three-generator gallery:** sample G1/G2/G3 with shared z and terrain context;
   display generated state, engine commands, and outcome. Toggle the composed
   E -> G3 outcome. These are independent generated transitions, not a policy.
4. **Model-based landing:** after prediction checks, add a short-horizon sampling
   planner through E -> G3 with an explicit landing objective and terminal design.
   Compare the same planner with the direct predictor and simulator, plus the
   reference heuristic. Evaluate real-environment return, landing/crash rates,
   and truncations. G2 can later propose actions, but it is not already a planner.

A precomputed standalone HTML replay plus video/GIF is the first shareable artifact.
Arbitrary live counterfactuals require a local checkpoint-inference and simulator
backend; do not present interpolated cached frames as live model computation.
Choose scenes by a fixed documented rule and expose failures as well as successes.

## Implementation order after planning

1. Optional Gymnasium/Box2D dependency setup, version pinning, collector pilot,
   and deterministic prefix-replay tests. Verify command thresholds against the
   installed implementation, including exact boundary values.
2. Dataset and provenance artifacts, split/leakage checks, phase/action coverage
   report, persistence metrics, and fixed evaluation protocol.
3. Dimension-aware three-generator trainer with terrain context, E, shared state
   D, binary contact treatment, comparison configs, and checkpoint replay tests.
4. Timing smoke, bounded training round, leaderboard/readout/recommendation.
5. Verified checkpoint demo; evaluate model-based landing as a separate milestone.

Suggested paths: `lib/gym_transition.py`, `experiments/train_gym_transition.py`,
`examples/gym_world_model.py`, `configs/gym/lunar_lander/`,
`reports/gym/lunar_lander/`. Ignore raw data/checkpoints under
`results/gym/lunar_lander/`. Give each run a fresh directory, flushed log.txt,
metrics.jsonl, config/source hashes, and checkpoint. Maintain a stable log alias:

```bash
tail -F results/gym/lunar_lander/live.log
```

Implementation and the bounded baseline round follow this plan. Model-based
landing remains a separate milestone. Preserve unrelated changes and existing benchmarks.

# Next experiment: one state encoder, trained from scratch

Status: completed. Both matched runs, fresh paired evaluations, cross-prediction
diagnostics, and the six-controller viewer are implemented and verified. See
[the readout](../reports/gym/lunar_lander_state_control/READOUT.md). The original
pre-run plan below is retained for provenance; it does not request another run.

The user proposed removing the action input entirely and using one useful
encoder, with G1/G3 either measuring the representation or helping train it.
This supersedes the previous readout's suggested DAgger experiment as the next
task. Do not launch DAgger, RL, or an automatic hyperparameter sweep.

## Workspace and constraints

- Repository `/home/martyn/dev/ParticleGAN`, branch `feature/gym-world-model`.
- Preserve all existing uncommitted work and unrelated user changes. No commit,
  PR, or merge requested for Lunar Lander yet.
- Use `.venv/bin/python`. Experiments use **GPU 1**; GPU 0 belongs to the user.
- User previously requested subagents for implementation; that authorization
  persists. Delegate independent bounded implementation/evaluation work.
- AGENTS.md: no repeated training experiments differing only in seed; flushed
  logs easy to tail; finish with metrics, leaderboard, explanations, recommendations.
- Existing evaluation protocols hash model/training sources. Prefer new modules
  and scripts instead of modifying frozen implementations. Preserve old reports.

## What we learned in the completed round

Read `docs/gym-control.md` and
`reports/gym/lunar_lander_control/READOUT.md` for implementation/results.

| Controller | Test landings | Mean return |
| --- | ---: | ---: |
| Hand-written Gym heuristic | 50/50 | 287.29 |
| Pretrained E_control + G2, imitation fine-tune | 50/50 | 286.76 |
| Pretrained joint three-generator fine-tune | 12/50 | 74.87 |
| Original pretrained paired encoder reused for control | 0/50 | -374.53 |

Both fine-tunes started from the same existing adversarial world-model checkpoint,
not random initialization. They copied E_pair to E_control, used 47 expert
training episodes / 9,297 transitions, and trained 2,500 updates with batch256.
Both selected update2,500 by validation landing rate then mean return.

Original E_pair learned `(state, current action)` but the original live prototype
fed it the previous action. The fine-tunes corrected that input meaning by
training a separate E_control on `(state, previous expert action) -> current
expert action`. They still had the demonstration/live difference of expert
previous actions during training versus their own previous actions in playback.

The new state-only encoder removes that explicit previous-action input. It does
not remove the general distribution shift between expert and learner states.

## Agreed graph and comparison

```text
                    +-- G1 -> reconstructed st
st -> E -> z -------+-- G2 -> at
                    +-- G3 -> predicted st+1

actual simulator.step(at) -> observed st+1
repeat using the observed state
```

Terrain still enters E and the generators. There is one encoder, no action input,
no previous-action input, and no trajectory unrolling. Retain MoG1024, z32,
bounded encoder offsets, independent G1/G2/G3, and the familiar width128 setup.
Initialize all learned parameters from scratch. Do not load pretrained weights
into either new arm. Normalization statistics are data preprocessing, not weights.

Compare two matched runs:

1. **Action-only with probes:** E -> z -> G2 learns the expert action. G1/G3 are
   also trained, but receive `stop_gradient(z)`. Their reconstruction/prediction
   losses update only their own weights. Held-out probe scores measure what can
   be decoded from the action-trained representation; they cannot shape E/prior.
2. **Joint auxiliary training:** same initialization and graph, but G1/G3 losses
   also backpropagate through z into E and the prior. This tests whether asking
   the representation to explain states and outcomes helps action selection.

Core task losses:

```text
L_action = standardized current expert-action MSE
L_state  = current-state reconstruction loss
L_next   = observed successor prediction loss

joint: L_action + lambda_state * L_state + lambda_next * L_next
probes: same head targets, but G1/G3 use detached z
```

Use continuous-coordinate MSE and separate binary-contact BCE, matching existing
state loss conventions. There is no action reconstruction through E(state,action),
no synthetic reconstruction loop, and no discriminator in this initial isolation
experiment. Joint/marginal GAN losses are a possible later separate comparison;
do not conflate this new auxiliary-loss arm with the old adversarial joint arm.

## Implementation defaults to freeze before running

These are practical defaults carried from the previous round, rather than extra
user-specified constraints. Record any necessary adjustment before full runs:

- Same 9,297 training expert records from actual behavior episodes; no alternative
  branch actions. The expert current action is a target, never an encoder input.
- Retain the original training-only scaler for comparability; never fit test data.
- 2,500 updates, batch256, checkpoints250/1,000/2,500, same data draw sequence.
- Lambda_state=lambda_next=1 initially; record precise per-head reductions.
- Train the MoG prior in **both** new arms under the same rule. Action-only prior
  gradients come from action learning; joint additionally gets auxiliary gradients.
  Use the same default MoG prior regularizer in both and report it separately from
  the task losses. No auxiliary-head gradient may leak into the action-only prior.
- Same optimizer/LR/EMA conventions in both arms, including probe head updates.
  All corresponding initial weights must match. Separate RNG streams prevent
  auxiliary/probe work from changing action minibatches or shared initialization.
- Run meaningful correctness smokes, then the two full arms sequentially on GPU1.
  No automatic budget extension or seed repeats.

Tests should prove input/target separation, matched initialization, and gradient
scope: action loss reaches E/G2/prior; detached probe losses reach G1/G3 only;
joint auxiliary losses reach E/prior as well. Confirm real-simulator playback
uses only current observed state plus terrain and does not use recorded targets.

## Interpretation of G3

G3 has no alternative-action input. On expert demonstrations it learns successors
under expert behavior. The intended use is to predict the transition associated
with the action produced from the same latent. This is **not** an unrestricted
counterfactual world model. If G2 differs from the expert, consistency with its
own action must be measured; the shared latent does not guarantee physical validity.

Use held-out expert transitions to score the auxiliary heads. Also compare G3's
prediction against the actual successor during learner rollouts if practical.
Do not present the old four-action counterfactual benchmark as a fair test of an
architecture that cannot receive the alternative action. Never restore arbitrary
eight-dimensional observations as complete Box2D state; use the actual live env.

## Evaluation and deliverables

Freeze fresh paired validation/test worlds before candidate evaluation, disjoint
from original collection and previous control rounds. Suggested ranges591000–591019
and691000–691049; verify provenance rather than assuming disjointness. These are
evaluation worlds shared across methods, not seed-repeat training experiments.

Keep landing rate primary, tie-break validation mean return; test only selected
and final checkpoints. Preserve explicit simulator success/crash/bounds/time-limit
conditions, Wilson intervals, per-episode returns, engine usage, paired comparisons,
and action traces. Report inference and training costs and G1/G3 metrics separately.

Re-evaluate the successful pretrained imitation controller and heuristic on the
same new worlds as references. Its old50/50 score is historical, not a paired
score on the new test. Do not let reference performance select new test worlds.

Suggested separate paths:
`configs/gym/lunar_lander_state_control/`,
`results/gym/lunar_lander_state_control/`,
`reports/gym/lunar_lander_state_control/`.
Provide a stable flushed `results/gym/lunar_lander_state_control/live.log`.

Update the playable viewer to include both new controllers, preserving old
options. Choose the default from comparable validation scores, never test scores.
Maintain Play/Pause/Step/Reset, reset/pause on switching, and real native frames.
Finish with the leaderboard, what auxiliary training did or did not improve,
its compute cost, and a metrics-grounded next recommendation.

## Current live viewer

Server at http://localhost:8787, CPU inference, default imitation controller,
paused on seed391000 at the end of the previous task. Exec session14310 was left
running. Verify process ownership before restarting. Leave unrelated services on
ports8765/8766 alone. CLI `examples/gym_lander_live.py`; implementation
`lib/gym_lander_live.py` / `.html`; manifest
`reports/gym/lunar_lander_control/controllers.json`.

Old round: full suite322passed +27subtests,4opt-in skips; subsequent focused
provenance/probe tests passed. Chromium tested all four controllers, resets,
playback, successful landing and terminal stop. Artifacts and hashes are in the
old reports; do not rewrite them to incorporate this new round.

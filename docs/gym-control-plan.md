# Next round: does joint three-generator learning help control?

Status: implemented and evaluated. See `docs/gym-control.md` and
`reports/gym/lunar_lander_control/READOUT.md`. Imitation landed 50/50 held-out
episodes; joint three-generator training landed 12/50. The viewer defaults to
the validation-selected imitation controller and retains all four modes.
Current branch: `feature/gym-world-model`. Preserve existing uncommitted work,
the frozen world-model leaderboard, and unrelated user changes. User authorizes
subagents. Use GPU **1** for experiments; GPU 0 belongs to the user.

## User's objective

Keep the three-generator approach, train the action-selection path explicitly
using expert data, establish control metrics/baselines, then update the live
simulation to compare controllers with a Play button. The scientific question is:
does learning states, actions, and outcomes jointly improve imitation/control
over training the same action-selection path alone?

Do not silently replace this with a conventional direct policy network or SAC.
DAgger and reward fine-tuning remain subsequent options, not this initial round.

## Starting point and known limitations

- Existing main checkpoint: `results/gym/lunar_lander/adversarial/best.pt`, update
  1,000 selected by validation. Full frozen old board and findings:
  `reports/gym/lunar_lander/baseline/`, `reports/gym/lunar_lander/READOUT.md`.
- Core graph: MoG1024, z32, independent G1/G2/G3, width128, particle_ae E,
  bounded offsets, EMA, Rp logistic + bcap, joint/action/shared-role state Ds.
- Live prototype: `examples/gym_lander_live.py`, `lib/gym_lander_live.py`,
  `lib/gym_lander_live.html`. Started at http://localhost:8787 on CPU. Ports
  8765/8766 are unrelated user services; leave them alone. Verify server status
  after compaction. Logs: `results/gym/lunar_lander/live_controller_server.log`.
- Current controller reuses E(state, **previous** action) -> G2. Original E was
  trained on state and **current** action to reconstruct that action. This is
  explicitly a prototype, not a trained action-selection policy.
- World-model training deliberately included one behavior action and three
  alternative commands per anchor. Expert preference was not supervised.
- Saved training source episodes include 47 unperturbed heuristic episodes;
  all 47 ended with reward +100, mean return 278.953. Saved validation/test
  heuristic episodes also succeeded (12 each). These are observed finite samples,
  not a universal expert success guarantee. Gym's callable heuristic can supply
  more labels/episodes without training another expert.
- The reconstruction-only world-model arm developed synthetic-state runaway.
  Diagnostic script/report: `experiments/diagnose_gym_transition.py`,
  `reports/gym/lunar_lander/diagnostics.json`. Do not casually add its unstable
  synthetic loss to the imitation-only baseline.

## Explicit graph for the new work

```text
Shared latent draw + terrain:
G1 -> st
G2 -> at
G3 -> st+1

Paired reconstruction / world prediction:
E_pair(st, current at, terrain) -> z -> G1/G2/G3

Action selection:
E_control(st, previous at, terrain) -> z -> G2 -> expert-like current at

Live episode:
previous at = [-1, 0]              # engines off
E_control(st, previous at) -> z -> G2 -> at
actual simulator.step(at) -> st+1
previous at = at
repeat until terminated or truncated
```

Use separate E_pair and E_control instances so the two meanings of the action
input cannot conflict. Initialize E_control as a copy of the existing E for both
learned arms; both start with the same G2 and prior checkpoint. All generators,
encoders, and discriminators keep terrain context. No reward enters an encoder.

G3 remains latent-input. World-model G3 is not used to advance the live physical
world in this round. A later planner through G3 is a distinct experiment.

## Frozen data and comparison arms

Build an expert-only finite dataset from the 47 existing **training** heuristic
episodes. Use their actual expert behavior transitions, not the three exploratory
branches. Each shuffled record contains st, previous command, expert current
command, next state, terrain. At episode start, previous command is [-1,0].
Record source episode IDs and hashes. Never reconstruct previous commands from
the shuffled four-action branch arrays. Keep the existing training-only scaler
and identical initial weights across learned arms.

Previous command is an explicit input field; training still consumes individual
records, without sequence unrolling or a trajectory loss. Rollout feedback will
use the learner's own previous command, unlike demonstration-only training.
Record this distribution shift; addressing it is the motivation for later DAgger.

| Arm | Updates |
| --- | --- |
| Heuristic expert | None; reference controller in the actual simulator |
| Existing E(previous action) -> G2 prototype | None; measures the starting behavior |
| Imitation only | Train E_control + G2 against expert current actions; freeze prior and other world-model modules |
| Joint three-generator imitation | Same control imitation loss, plus the expert-transition joint/marginal GAN and paired reconstruction objectives; train G1/G2/G3, E_pair, prior and Ds as well |

The action loss is mean squared error in the shared action scaler's coordinates,
with weight 1. Preserve the established joint/marginal averaging, contact
representation, and explicit reconstruction weights in the joint arm. Initialize
its Ds from the same existing checkpoint and keep the current recipe unless a
correctness smoke exposes a necessary change. Document the old synthetic
reconstruction path precisely; retain it for this first joint comparison rather
than simultaneously changing its detach semantics. Stop/report a nonfinite run.

The inference graph and initial E_control/G2/prior are matched; auxiliary
capacity and computation are intentionally larger in the joint arm. Report both
trainable and inference parameter counts, record draws, simulator calls, and
timing. This tests the benefit of the complete auxiliary training package, not
which individual generator or discriminator caused a difference.

Before training, freeze initial budget/configs: 2,500 updates, batch256, checkpoints
250/1,000/2,500, same MoG optimizer/LR/EMA conventions adapted to this budget.
Run correctness/timing smokes and the two learned arms sequentially on GPU1.
No seed-repeat training experiments, automatic long extensions, or hidden sweeps.

## Primary metrics: actual control outcomes

Create a fresh control evaluation protocol before running any candidate. Use
20 fixed validation episodes and 50 fixed test episodes, with disjoint reset
seeds outside the previously collected/trained worlds. Example seed ranges:
391000–391019 validation, 491000–491049 test. Verify no overlap against provenance.
These are evaluation episodes shared by all methods, not training-seed repeats.

Select checkpoints by validation successful-landing fraction, breaking ties by
mean episode return. Score test only after selection. Also report final
checkpoints separately. Same initial states/terrain/reset seeds per method;
different actions will legitimately yield different trajectories.

Primary leaderboard columns:

- Successful landing count / episode count and landing rate, with uncertainty
  appropriate to the finite episode sample.
- Mean and median episode return.
- Crash, out-of-bounds, and time-limit counts, using explicit simulator reasons.
- Episode length and main/lateral engine usage.
- Paired episode wins/losses versus imitation-only and the untrained prototype.

Define success from the installed simulator's successful terminal condition,
not merely positive reward or `done=True`. Preserve terminated vs truncated.
Do not count a time limit as a crash. Print reasons and retain per-episode
records for review. No stepping after episode end, no ground snapping, and no
cherry-picking scenes for the aggregate scoreboard.

Useful diagnostics, separate from the primary control ranking:

- Expert action error and engine-regime agreement on held-out expert records.
- On-policy action traces: previous/current commands, repeated commands,
  selected latent components, offset saturation, and visited states.
- Frozen world-model prediction/reconstruction and distribution checks to
  identify changes in G1/G3 while adapting control. Do not rewrite the old board.
- Inference latency and real-time playback speed.

Evidence for the research idea requires the joint arm to improve actual held-out
control relative to imitation-only. Better expert action MSE or nicer animations
alone is insufficient. Report ties, failures, and uncertainty candidly.

## Live simulation and deliverables

After evaluation, add a controller selector to the live viewer: expert, original
prototype, imitation-only, joint three-G. Default to the validation-selected
controller winner; retain the other modes. The simulator always supplies the
physical successor and native rendered frame. Display the selected controller,
episode return, commands, latent route where applicable, and terminal reason.
Changing controller/reset must pause and restart the episode so comparisons have
clear provenance; preserve Play/Pause/Step and deterministic reset behavior.

Keep artifacts separate, suggested paths:
`configs/gym/lunar_lander_control/`, `results/gym/lunar_lander_control/`,
`reports/gym/lunar_lander_control/`. Provide source/data/checkpoint/protocol hashes,
fresh per-run directories and flushed logs with a stable alias:

```bash
tail -F results/gym/lunar_lander_control/live.log
```

Finish with the new leaderboard, explanation of whether joint training helped,
the live viewer URL, and a recommendation based on metrics. If imitation works
on demonstrations but fails after its own mistakes, DAgger is the next targeted
comparison. Do not add RL or planning automatically during this baseline round.

This document preserves the experiment's original design. The completed results,
costs, verification, and follow-up recommendation are recorded in the readout.

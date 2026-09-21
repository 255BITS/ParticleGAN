# Lunar Lander control with three generators

This experiment asks whether joint state/action/outcome training helps a learned
controller land, compared with training the same action-selection path alone.
The simulator provides the physical successor at every step.

The first round is complete: imitation-only landed **50/50** fresh test worlds,
joint training **12/50**, and the original prototype **0/50**. The live viewer
defaults to imitation based on validation results. The joint arm remains
available; its added training package did not improve control in this round.
The same action-MSE objective has a dedicated entrypoint with no adversarial
loss path: [L2 finetune](gym-l2-finetune.md).

```text
Shared MoG latent + terrain:
G1 -> st
G2 -> at
G3 -> st+1

World-model reconstruction:
E_pair(st, current at, terrain) -> z -> G1/G2/G3

Control:
E_control(st, previous at, terrain) -> z -> G2 -> current at
simulator.step(current at) -> st+1
```

`E_control` starts as a copy of the existing paired encoder. Separate encoders
give the action input a consistent meaning in each task. The control encoder
does not receive the expert's current action when choosing an action. Initially
the previous command is `[-1, 0]` (engines off). Inference uses one latent route
from the 1,024-component MoG and the corresponding bounded offset. All branches
also receive the eleven terrain heights used by the existing world model.

The finite demonstration set contains 9,297 individual transitions from 47
existing unperturbed heuristic training episodes. Each record contains the
observed state, previous command, expert current command, successor, and terrain.
Records are shuffled; there is no trajectory unrolling or trajectory loss.
The original training-only normalization is retained. Demonstrations contain
expert previous commands; live rollouts feed back the learner's own commands.

The comparison has four controllers:

| Controller | What learns |
| --- | --- |
| Heuristic expert | Nothing; simulator's reference controller |
| Original prototype | Nothing; paired encoder reused with the previous command |
| Imitation only | E_control and G2, supervised expert action MSE |
| Joint three-generator imitation | Same imitation objective, plus G1/G2/G3, E_pair, MoG prior, and joint/marginal discriminators |

Both learned controllers start from the same validation-selected world-model
checkpoint and train for 2,500 updates with batch size 256. The MoG recipe uses
Rp logistic losses, bcap regularization, prior spread regularization, and EMA.
The joint arm preserves the existing real and synthetic reconstruction losses,
including detached synthetic targets and live synthetic encoder inputs. Its
additional parameters and computation are part of the comparison.

Checkpoints at 250, 1,000, and 2,500 updates are ranked by successful landings on
20 fixed fresh validation episodes, breaking ties by mean episode return.
Selected checkpoints and final checkpoints are then evaluated on 50 fresh test
episodes, shared across controllers. These are paired evaluation worlds, not
repeated training runs with different random seeds. Test scores do not select
the model or the default controller.

Landing success follows the installed simulator's successful terminal condition
(lander asleep). Crash, out-of-bounds, and time-limit outcomes are retained
separately, together with the raw termination flags. Every rollout stops at the
first termination or truncation. Episode returns, engine use, action traces,
latent routes, and paired outcomes accompany the landing-rate leaderboard.
World-model prediction and generation scores are secondary diagnostics on the
old frozen reference mixture; they do not select a controller.

The [experiment plan](gym-control-plan.md) records the frozen design and limits.
Training logs are flushed to a shared path:

```bash
tail -F results/gym/lunar_lander_control/live.log
```

To reproduce in fresh output directories, freeze the evaluation protocol, run
the reference controllers, then train the two arms sequentially on GPU 1:

```bash
python -u experiments/evaluate_gym_control.py --freeze --baseline
python -u experiments/train_gym_control.py --config configs/gym/lunar_lander_control/imitation.yaml
python -u experiments/train_gym_control.py --config configs/gym/lunar_lander_control/joint.yaml
```

For each arm, pass its three candidates and final checkpoint to the evaluator:

```bash
python -u experiments/evaluate_gym_control.py --arm imitation \
  --checkpoint results/gym/lunar_lander_control/imitation/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_control/imitation/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_control/imitation/checkpoint_2500.pt \
  --final results/gym/lunar_lander_control/imitation/final.pt
```

Repeat that evaluation command with `joint` in place of `imitation`. The
evaluator stores validation selection before evaluating test episodes and
writes `best.pt`, the leaderboard, and the live controller manifest. Training
refuses nonempty output directories; evaluation only reuses cached results when
their provenance matches. Existing baseline artifacts should be preserved.

The playable simulator runs with:

```bash
python -u examples/gym_lander_live.py
```

Open http://localhost:8787. The controller selector compares the expert,
original prototype, imitation-only, and joint three-generator controllers.
Switching controllers pauses and resets the episode. Play, Pause, Single Step,
and deterministic Reset operate the actual Gym simulator and its native renderer.
`G3` remains available as a world model; it does not advance the live lander.

See the [control leaderboard](../reports/gym/lunar_lander_control/README.md)
and [research readout](../reports/gym/lunar_lander_control/READOUT.md) for results,
costs, and the next experiment justified by those results.

A later fine-tune keeps this initialization and three-generator graph. The
default particle path is YuE2 paired-error RpGAN at `adv_weight=1`
([ParticleGAN fine-tune](gym-particle-finetune.md)). On the shared control
protocol that recipe scored validation 20/20, test 50/50, mean return 287.7
([PR #18](https://github.com/255BITS/ParticleGAN/pull/18)). L2 and the slider
fine-tune remain separate arms.

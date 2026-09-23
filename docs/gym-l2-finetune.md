# Lunar Lander L2 finetune

Fine-tune the pretrained three-generator controller with **standardized expert
action MSE** only. This is the imitation objective that landed 50/50 on the
published control worlds. The trainer has no discriminator step and no
reconstruction term.

```text
Start: results/gym/lunar_lander/adversarial/best.pt
E_control = copy of paired E
Train: E_control and G2
Frozen: G1, G3, paired E, MoG prior, discriminators

Loss: mean squared error of scaler.action(G2) against the expert command
Not in the loss: contact BCE, state reconstruction, prior spread, adversarial
```

`E_control(st, previous at, terrain) -> z -> G2 -> current at`. The encoder reads
the frozen MoG prior when it routes `z`. Prior weights are not optimizer steps.
Training records use the expert's previous command. Rollouts feed back the
controller's own command, starting from engines-off `[-1, 0]`. The simulator is
not called during training.

## How this differs from scratch training

| Recipe | Initialization | Objective |
| --- | --- | --- |
| L2 finetune (this path) | Adversarial world-model checkpoint; E_control copied from paired E | Standardized action MSE on E_control and G2 |
| Imitation arm in `train_gym_control.py` | Same checkpoint and same action MSE | Same objective, but that file also implements the joint GAN arm |
| Joint finetune | Same checkpoint | Action MSE plus reconstruction, prior spread, and discriminator losses |
| Direct next-state L2 | Fresh `DirectPredictor`, not this graph | Next-state MSE and contact BCE on shuffled transitions |
| Previous-action GAN | Fresh G, E, prior, and D | Action MSE plus reconstruction and joint/marginal adversarial losses |

Contact BCE is the world-model state term. It is not part of the imitation
objective, so this finetune does not add it. The direct predictor remains the
scratch comparison for next-state error; it is a different network.

The published imitation run, not a new measurement, is the reference for this
objective. Same initialization, 9,297 expert transitions, 2,500 updates, batch
256, seed 24002. From
[the control readout](../reports/gym/lunar_lander_control/READOUT.md):

| Controller | Selected update | Validation landings /20 | Test landings /50 | Test mean return |
| --- | ---: | ---: | ---: | ---: |
| Heuristic expert | — | 18 | 50 | 287.29 |
| Imitation action MSE | 2,500 | 20 | 50 | 286.76 |
| Joint finetune | 2,500 | 7 | 12 | 74.87 |
| Original prototype | 1,000 | 0 | 0 | −374.53 |

The direct supervised world model, on a different prediction task, has test
next-state MSE **0.007048** at its validation-selected checkpoint
([world-model readout](../reports/gym/lunar_lander/READOUT.md)). That number is
not a landing rate.

No checkpoint or episode file is stored in git. This environment has no GPU, so
landing counts above are the prior readout only.

## Run

Full finetune on GPU 1, after the world-model checkpoint and collected episodes
exist. The output directory must be empty. Logs are line-buffered:

```bash
python -u experiments/train_gym_l2_finetune.py \
  --config configs/gym/lunar_lander_finetune/l2.yaml
tail -F results/gym/lunar_lander_finetune/live.log
```

CPU smoke (2 updates, still needs the same checkpoint and episodes):

```bash
python -u experiments/train_gym_l2_finetune.py \
  --config configs/gym/lunar_lander_finetune/l2.yaml \
  --device cpu --steps 2 \
  --out-dir results/gym/lunar_lander_finetune/l2_smoke \
  --live-log results/gym/lunar_lander_finetune/live.log
```

Each log line is prefixed `[l2]` and reports `action_mse` and
`adversarial_updates=0`.

After a full run, score the three checkpoints on fresh resets. This writes a
new report directory and does not update the published control board:

```bash
python -u experiments/evaluate_gym_l2_finetune.py --freeze \
  --out reports/gym/lunar_lander_finetune \
  --episodes results/gym/lunar_lander/data/episodes.json
python -u experiments/evaluate_gym_l2_finetune.py \
  --out reports/gym/lunar_lander_finetune \
  --checkpoint results/gym/lunar_lander_finetune/l2/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_finetune/l2/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_finetune/l2/checkpoint_2500.pt \
  --final results/gym/lunar_lander_finetune/l2/final.pt
```

Selection is validation landing rate, then mean return. The evaluator rejects a
checkpoint whose saved supervision is not action MSE or whose summary recorded
discriminator updates.

`tests/test_gym_l2_finetune.py` trains a tiny CPU fixture, checks that G1, G3,
the paired encoder, the prior, and D stay at their initial weights, and checks
that the action MSE matches the existing imitation arm on the same batches.

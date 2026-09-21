# ParticleGAN fine-tune — cited Lunar board

**Default / recommended:** YuE2 paired-error RpGAN at `adv_weight=1`.
The numbers below are the post-merge measurement on the shared control
protocol (validation seeds 391000–391019, test seeds 491000–491049), selected
step **2500**. They are cited from
[PR #18](https://github.com/255BITS/ParticleGAN/pull/18). This checkout does
not re-roll the simulator.

| Arm | Val landings | Test landings | Test mean return |
| --- | ---: | ---: | ---: |
| **YuE2 paired-error RpGAN (`adv_weight=1`) — default** | **20/20** | **50/50** | **287.7** |
| L2 imitation (separate arm) | 20/20 | 50/50 | 286.8 |
| Slider paired-error (separate arm) | — | 45/50 | — |
| Native #16 live `(record, z)` | 1/20 | 4/50 | −66.8 |
| Collapsed pure RpGAN + `b_cap` | 0/20 | 2/50 | −112.9 |

`adv_weight=0` configures RpGAN and `b_cap` without applying them. The trainer
rejects it. L2 and slider configs are separate arms and are not edited here.

Playback is `E_control(st, previous at) -> z -> G2`. G1, G3, `E_pair`, the
prior, and the transition critic stay frozen. The controller step is
edit-normalized paired-error RpGAN plus lazy sample-point `b_cap`. Details
are in [the experiment note](../../../docs/gym-particle-finetune.md).

`diag_action_mse` during training is teacher-forced action error on the
expert's previous command. It is not a landing rate. The #16 live-pair toy
can pass that kind of gate and still miss the pad; see
[the autopsy](../../../docs/native16-autopsy.md).

## How to reproduce

GPU 1, fresh directory. Then select with the control evaluator. Training does
not pick `best.pt`.

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
tail -F results/gym/lunar_lander_particle_finetune/live.log

python -u experiments/evaluate_gym_particle_finetune.py --freeze \
  --out reports/gym/lunar_lander_yue2 \
  --episodes results/gym/lunar_lander/data/episodes.json
python -u experiments/evaluate_gym_particle_finetune.py \
  --out reports/gym/lunar_lander_yue2 \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_2500.pt \
  --final results/gym/lunar_lander_particle_finetune/particle/final.pt
```

CPU toys (not Lunar scores):

| Toy | Command | Meaning |
| --- | --- | --- |
| YuE2 gate (default picture) | `python -u examples/yue2_particle_2d.py` | PASS: paired RpGAN + `b_cap` lands; `adv_weight=0` rejected |
| Collapse repro | `python -u examples/particle_control_2d.py` | Test passes only when pure RpGAN + `b_cap` FAILs |
| Native live pair | `python -u experiments/toy_particle_native_2d.py` | Old gate PASS, Lunar FAIL |
| Honest pad | `python -u experiments/toy_native16_autopsy.py` | Teacher-forced vs on-policy |

# ParticleGAN fine-tune vs L2 imitation

The previous particle recipe deleted imitation and reconstruction L2 and
trained the full graph with four-path RpGAN plus sample-point `b_cap`. On the
shared-protocol board that run scored **0/20 validation** and **2/50 test**
landings, against L2 at 50/50 and the slider arm at 45/50. Diagnostic action
MSE grew from about 0.06 to above 1. Those figures are cited from that board.
They were not remeasured in this checkout, and they are not a score for the
paired-error objective below.

This arm keeps a nonzero adversarial controller step. `adv_weight` is 1.
Setting it to 0 configures RpGAN and `b_cap` without applying them, which is
the failure mode of the closed model-glue attempt. The CPU gate
(`python -u examples/yue2_particle_2d.py`) rejects that supervised-only arm
even when its landings pass, and it accepts paired-error RpGAN plus lazy
`b_cap`. Playback is still `E_control(st, previous at) -> z -> G2`. G1, G3,
`E_pair`, the prior, and the transition critic stay frozen. Details are in
[the experiment note](../../../docs/gym-particle-finetune.md).

## Cited control leaderboard

Rows other than the collapsed particle line are copied from the completed
[control leaderboard](../lunar_lander_control/README.md). They are not
remeasured here. The collapsed particle line is the shared-protocol result
for the previous four-path recipe. The paired-error recipe in this branch
has no rollout yet.

| Controller | Selected update | Validation landings | Test landings | Notes |
| --- | ---: | ---: | ---: | --- |
| Imitation L2 | 2,500 | 20/20 | 50/50 | Playable default |
| Slider paired-error | — | — | 45/50 | Separate arm, left intact |
| Collapsed four-path particle | 2,500 | 0/20 | 2/50 | Previous recipe; action MSE exploded |
| Paired-error particle (this branch) | — | not run | not run | `adv_weight` 1; CPU gate only |

## How to run

GPU 1, fresh directory, no seed sweep:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
tail -F results/gym/lunar_lander_particle_finetune/live.log
```

CPU gate and smoke (correctness only, not a landing result):

```bash
python -u examples/yue2_particle_2d.py
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/smoke
```

`experiments/toy_particle_native_2d.py` remains a separate CPU example of the
live `(record, z)` pair. It is not the gym controller step.

## Safe-fast arm

The first `particle_safe_fast.yaml` (#21) scored 0/20 validation and 0/50
test on Lunar, mean return about −407, against YuE2 #18 at 20/20 and 50/50.
The throttle-up revision (`safe_fast_speed_limit` 0.18) was then trained on
the same protocol: step 1000 scored 17/20 (mean return about +243) and step
2500 scored 0/20 (about −189). Between steps 1750 and 2000, diag action MSE
went from about 0.069 to 2.28, D-loss from about 0.63 to 0.012, and the
safe-fast term from about −0.46 to +4.9. Those Lunar numbers are not
remeasured here. The trainer used to copy step 2500 to `final.pt`. With
`safe_fast_weight` positive it now ships the last checkpoint that passes
the late-collapse rule (on that trace, step 1000). `particle.yaml` is
unchanged and still copies the last step. The CPU gate fails a blind export
of both the frozen trace and a live walk-off, and passes selection.
`adv_weight` stays 1. A Lunar retrain of the export has not been run.
See [the note](../../../docs/gym-safe-fast.md).

```bash
python -u examples/safe_fast_2d.py
```

## Recommendation

Keep `particle.yaml` as the paired-error default. Do not ship `adv_weight=0`
or put action MSE back in place of the GAN. Do not start a seed repeat.
The safe-fast yaml is the candidate if a later rollout wants earlier
landings. Do not score its step-2500 weights from the collapsed run; the
export rule would have kept step 1000. Score the shipped checkpoint on the
existing validation worlds before any test claim. This checkout did not run
that rollout.

# ParticleGAN fine-tune vs L2 imitation

The pure RpGAN + sample `b_cap` fine-tune deleted paired MSE/BCE and trained
the full graph. That arm is the collapsed row below. The trainer in this
checkout no longer runs that update. It trains G2 with the model-glue paired
continuation described in
[the experiment note](../../../docs/gym-particle-finetune.md). Landings for the
continuation have not been run.

Playback is still `E_control(st, previous at) -> z -> G2`. G1 and G3 stay in
the checkpoint and do not train. Details, learning-rate groups, and the removed
terms are in
[the experiment note](../../../docs/gym-particle-finetune.md).

## Cited L2 control leaderboard

These rows are copied from the completed
[control leaderboard](../lunar_lander_control/README.md) and
[readout](../lunar_lander_control/READOUT.md). They are not remeasured here.
Selection used validation landings, then mean return. Test worlds were the
50 fresh episodes in that protocol. Intervals are the reported 95% Wilson
intervals.

| Controller | Selected update | Validation landings | Test landings | Wilson 95% | Mean test return | Median | Crash / bounds / time limit |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Imitation L2 | 2,500 | 20/20 | 50/50 | 92.9%–100.0% | 286.76 | 286.80 | 0 / 0 / 0 |
| Joint L2 + GAN + reconstruction | 2,500 | 7/20 | 12/50 | 14.3%–37.4% | 74.87 | 26.48 | 37 / 1 / 0 |
| Original prototype (no fine-tune) | 1,000 | 0/20 | 0/50 | 0.0%–7.1% | -374.53 | -430.87 | 33 / 17 / 0 |
| ParticleGAN pure RpGAN + b_cap (collapsed) | 250 | 0/20 | 2/50 | — | about −113 | — | — |
| ParticleGAN model-glue continuation | — | not run | not run | — | — | — | — |

Imitation validation progress in that readout was 3/20, 9/20, 20/20 at updates
250, 1,000, and 2,500. Joint was 0/20, 0/20, 7/20. The joint arm lost 38 test
landings to imitation and won none.

Other GAN reports use different worlds or training histories. They are not
rows in the table above. For orientation only: the later joint GAN landed
34/50 on its own worlds
([GAN control readout](../lunar_lander_gan_control/README.md)); the
previous-action L2 GAN landed 7/50 and the slider-error replacement landed
6/50 on yet another paired set
([slider readout](../lunar_lander_slider_gan/READOUT.md)). None of those
numbers is a score for this fine-tune.

The collapsed row is the reported shared-protocol result for the pure
adversarial arm. It was not recomputed in this checkout, and it has no Wilson
interval here. The continuation row is empty until a rollout.

## How to run

CPU formulation gate:

```bash
python -u examples/particle_control_2d.py
```

GPU 1, fresh directory, no seed sweep:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
tail -F results/gym/lunar_lander_particle_finetune/live.log
```

CPU smoke (correctness only, not a landing result):

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/smoke
```

After a full run, score checkpoints 250, 1,000, and 2,500 on the existing
control validation worlds before any test claim. This repository's control
evaluator currently accepts only the imitation and joint checkpoint format, so
that rollout harness still has to grow a loader for
`gym_particle_finetune_v1`. Until then the landing cell stays empty.

This checkout has no GPU and no saved adversarial checkpoint, so the full
command was not run. A CPU correctness smoke did run: two updates on a
synthetic eight-record fixture, with flushed `live.log` lines, auxiliary L2
weight 0, adversarial updates 0, gradients into G2 only, and identical actions
from `final.pt` and `checkpoint_2.pt`. That smoke is not a Lunar Lander score.

## Reported collapse and the 2D check

A later shared-protocol rollout of the pure RpGAN + `b_cap` arm, with every
imitation and reconstruction weight at 0, selected step 250 and scored 0/20
validation landings and 2/50 test landings, mean return about −113. L2 on that
protocol was 50/50. Training `diag_action_mse` moved from about 0.06 to above 1.
This checkout did not rerun that rollout.

`examples/particle_control_2d.py` is the CPU check of the replacement recipe.
The pure-GAN baseline trains the stem, action head, and particle cloud. The
fix freezes the stem and cloud and trains the action head with the model-glue
anchor and functional match. The gate prints PASS or FAIL. It does not use
Lunar weights. Passing it does not show that the gym wiring lands.

## Recommendation

Keep the imitation controller as the playable default. The particle trainer now
follows the toy's passing recipe (G2 only, anchor 0.1, functional weight 1,
adversarial weight 0, head learning rate 1e-5, EMA 0.98). Do not treat
`selected.pt` as a landing champion until the existing control worlds are
scored. Do not start a seed repeat. The L2 and slider arms are unchanged.

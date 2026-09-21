# ParticleGAN fine-tune vs L2 imitation

Hypothesis: the imitation fine-tune lands because pointwise action MSE trains
`E_control` and G2 directly. The joint fine-tune kept that MSE and added
reconstruction MSE/BCE plus the RpGAN/b_cap game, and landings fell. This arm
deletes every paired MSE/BCE term, including any tiny auxiliary, and fine-tunes
the same checkpoint with only the relativistic paired game (sample-point
`b_cap`, joint and marginal critics) and the MoG table regularizer. If action
MSE was necessary, landings should stay well below imitation. If the joint
arm's drop was L2/GAN interference, a pure adversarial fine-tune could do
better. One unrun recipe cannot decide that.

Playback is still `E_control(st, previous at) -> z -> G2`. G1 and G3 stay in
the trained graph. Details, learning-rate groups, and the removed terms are in
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
| ParticleGAN fine-tune (this arm) | — | not run | not run | — | — | — | — |

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

## How to run

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
weight 0, gradient reach into both encoders and all three generators, and
identical actions from `final.pt` and `checkpoint_2.pt`. That smoke is not a
Lunar Lander score.

## Recommendation

Keep the imitation controller as the playable default. Run one 2,500-update
fine-tune on GPU 1, then evaluate landings. Do not start a seed repeat, a
slider hybrid, or a claim that this arm beats 50/50. If the rollout is far
below imitation, the next useful change is to put action MSE back on
`E_control` only and leave the ParticleGAN terms on the prior and state paths,
rather than tuning `b_cap`.

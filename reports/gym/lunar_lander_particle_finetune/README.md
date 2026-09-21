# ParticleGAN fine-tune vs L2 imitation

Post-merge landings on the shared protocol were 1/20 validation (mean return
about −84.6, selected step 2500) and 4/50 test (mean return about −66.8).
Training `diag_action_mse` stayed about 0.07–0.18. That is the proxy the 2D
gate treats as a pass. The comparison with the collapsed particle run (2/50)
and the L2 run (50/50, mean about 286.8) is written up in
[the native-16 autopsy](../../../docs/native16-autopsy.md). This directory's
trainer is unchanged.

Hypothesis: deleting paired L2 is viable when the adversary is the one
ParticleGAN already uses without reconstruction. An observation critic matches
transition marginals and leaves the conditional action free, which is the
collapse (diagnostic action MSE rising, 0/20 validation and 2/50 test landings
on the shared protocol). `examples/five_modes.py` drops reconstruction and
scores the joint pair `(x, z)` with Rp logistic loss and sample-point `b_cap`.
This arm does that. Adversarial weight is 1. L2 weights stay 0. The generator
step is what updates `E_control` and G2. G1, G3, `E_pair`, the MoG table, and
the observation critics stay frozen.

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
| ParticleGAN fine-tune (post-merge, not remeasured here) | 2,500 | 1/20 | 4/50 | — | −66.8 | — | — |

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

The post-merge score of checkpoints 250, 1,000, and 2,500 on the shared
protocol is the 1/20 and 4/50 result above. This checkout does not contain
that run's traces. The 2D pass is not a substitute for it.

This checkout has no GPU and no saved adversarial checkpoint, so the full
command was not run. A CPU correctness smoke did run: two updates on a
synthetic eight-record fixture, with flushed `live.log` lines, auxiliary L2
weight 0, adversarial weight 1, gradient from the RpGAN step into `E_control`
and G2 only, and identical actions from `final.pt` and `checkpoint_2.pt`.
That smoke is not a Lunar Lander score.

## CPU 2D gate

`python -u experiments/toy_particle_native_2d.py` (also
`tests/test_particle_native_2d.py`). Rp logistic, `b_cap` coefficient 1,
adversarial weight 1, L2 weight 0. EMA action MSE:

| Arm | EMA action MSE | Result |
| --- | ---: | --- |
| Observation critics, detached reals, all modules trained | 2.1065 | collapse (≥ 1) |
| Live `(record, z)` pair, `E_control` and G2 only | 0.1396 | pass (≤ 0.18) |

Init error was 1.9997. About 6 seconds on CPU.

## Recommendation

Keep the imitation controller as the playable default. The 2D gate's pass is
teacher-forced action MSE, and the post-merge landings show that proxy can
sit in the pass band while the craft does not land. Keep adversarial weight 1.
Do not set it to 0, and do not start a seed repeat. The next GAN change belongs
on the state-pad gate in the autopsy, not on another Lunar run, until that
gate fails a false pass and accepts a controller that actually holds the pad.

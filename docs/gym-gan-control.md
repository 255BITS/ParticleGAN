# Three-generator GAN control

This experiment restores adversarial training from the first update and ranks
only GAN-trained controllers. Both new models start from scratch:

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

observed st -> E -> z -> G1 / G2 / G3
playback: st -> E -> z -> G2 -> at -> simulator.step(at)
```

Terrain enters every G and E. One state-only encoder drives the playable policy;
G1/G3 remain trained parts of the graph. All generators, E, and the MoG prior
receive adversarial training throughout. Paired action/state prediction losses
also remain active; there is no imitation-only training stage.

The matched comparison adds marginal critics to a masked joint discriminator:

- **Joint GAN:** D observes complete labeled triplets or state/successor pairs
  with hidden action coordinates masked out.
- **Joint plus marginal GANs:** the same joint D, plus D on labeled actions and
  a shared state D for current and successor observations.

Both discriminator views see fake examples generated from prior codes and from
E(observed state). Real and fake receive the same observation mask. The mask is
fixed context for D, and is never an input to E or G. Hidden actions remain
unavailable even to preprocessing or gradient penalties.
The masking function is fixed; there is no learned mask generator in this
experiment. A shared state critic receives a current/successor role indicator.

We retain 1,010 action labels from five fixed episodes and 9,297 state/successor
pairs from all 47 expert training episodes. The default MoG recipe supplies
1,024 particles, relativistic adversarial loss, bcap, prior regularization,
optimizers, and EMA. Both runs use 2,500 updates on GPU 1.

The leaderboard also reevaluates the historical joint adversarial controller.
It used 47 labeled episodes and pretrained weights, so it is a reference rather
than part of the matched sparse-label comparison. Non-GAN results remain in
their historical reports and do not enter this leaderboard or choose its default.

The [frozen design](gym-gan-control-plan.md) records losses and evaluation rules.
Completed test results: **joint GAN 34/50 landings**, legacy GAN 23/50, and
joint plus marginals 16/50. Marginal critics improve several distribution metrics
but worsen control. The joint GAN is the validation-selected playable default.
See [the readout](../reports/gym/lunar_lander_gan_control/READOUT.md).

Follow progress:

```bash
tail -f results/gym/lunar_lander_gan_control/live.log
```

Reproduction commands (full training requires fresh output directories):

```bash
.venv/bin/python -u experiments/evaluate_gym_gan_control.py --freeze --baseline
.venv/bin/python -u experiments/train_gym_gan_control.py --config configs/gym/lunar_lander_gan_control/joint.yaml
.venv/bin/python -u experiments/train_gym_gan_control.py --config configs/gym/lunar_lander_gan_control/marginals.yaml
for arm in joint marginals; do
  run_dir="results/gym/lunar_lander_gan_control/$arm"
  .venv/bin/python -u experiments/evaluate_gym_gan_control.py --arm "$arm" \
    --checkpoint "$run_dir/checkpoint_250.pt" \
    --checkpoint "$run_dir/checkpoint_1000.pt" \
    --checkpoint "$run_dir/checkpoint_2500.pt" --final "$run_dir/final.pt"
done
.venv/bin/python experiments/plot_gym_gan_control.py
.venv/bin/python -u examples/gym_lander_live.py --seed 991000
```

Open http://localhost:8787 to play the real simulator. The newest controller
manifest selects a GAN-only validation winner by default. Earlier models remain
available with explicit GAN/non-GAN labels.

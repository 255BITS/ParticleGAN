# Checkpoint-preserving capacity scouts

User authorized stopping two-D continuation and using both GPUs through the experiment pipeline. The two-D run stopped at logged step 172100; latest and best completed FID50k was 18.3010 at 170k. Its existing checkpoints remain available. This round tests capacity growth from the original one-D checkpoint, not the two-D trajectory.

## Hypotheses

- G has insufficient capacity to exploit the discriminator's feedback. Add one residual refinement block at each of 8, 16, and 32 pixel resolutions.
- Trainable D heads underuse existing pretrained features. Add one residual refinement before pooling in each of the three feature heads, keeping the pretrained ResNet18 frozen.
- G and D capacity limit each other. Test both expansions together to measure their interaction.

G already uses GroupNorm and latent affine conditioning. New G blocks retain these choices. New residual branches start with zero final convolutions and use `x + residual`, without the existing block's division by sqrt(2), to preserve the learned function. D additions also start as identity functions. E, prior, reconstruction routing, optimizer rates, and one-D update ratio retain the parent settings.

## Four-arm protocol

| Arm | Grow G | Grow D heads |
|---|---|---|
| control | no | no |
| grow_g | yes | no |
| grow_d | no | yes |
| grow_both | yes | yes |

All restore `runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt`, SHA256 `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`, FID50k 18.9012. Same seed and complete model/EMA/Adam/RNG state; only new parameters have fresh optimizer state. Separate fixed initialization streams ensure the new G and D branches match between individual and combined arms without advancing training RNG.

Train 20k additional G updates, from 50k to 70k. Evaluate FID50k at 55/60/65/70k, with EMA G/prior and the existing CIFAR train50k Inception reference. Evaluate held-out reconstruction on 10k test images. Preserve a checkpoint at each evaluation. Lazy exact bcap every eight D updates, coefficient multiplied by eight. No seed experiments.

The unchanged arm is a contemporaneous control for this standalone trainer and evaluation schedule. Rank final 70k FID; report intermediate minima separately. Compare FID against both updates and elapsed training time. Factorial interaction is FID(both) − FID(G) − FID(D) + FID(control); one trajectory cannot establish uncertainty or a unique cause. A gain under one FID point requires particular scrutiny given the user's compute-cost preference. No automatic long-run promotion.

## Execution and validation

Standalone trainer `experiments/train_cifar_ae_growth.py` preserves all historical trainer and shared-library source certificates. Subagent implemented the trainer; primary agent handles pipeline, tests, integration, and reports.

Validation: real-parent four-arm output/input-gradient, old model/Adam/RNG preservation, new-layer learning, frozen EMA/features and idempotence tests; deterministic expanded-model 8-update versus 4+4 full-state replay; then four 16-update real-parent smokes through the pipeline before scout launch. Runtime growth audits are saved per run.

Launch: `bash experiments/cifar_ae_growth_pipeline.sh growth_scout 0,1`

Tail: `tail -F runs/cifar_particle_ae/growth_scout/PIPELINE.log`

Automatic completion report: `reports/cifar-particle-ae/growth_scout/LEADERBOARD.md`, `leaderboard.json`, and `curves.png`. Scheduler runs one job per GPU, two waves for four configurations. Completed results and recommendations are generated automatically.

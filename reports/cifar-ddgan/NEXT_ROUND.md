# Next round: preserve the best point before another architecture jump

The [attention and duration round](attention_duration/READOUT.md) is complete.
Plain U-Net: final FID50k25.397 at50k,45.73 training minutes.
Attention U-Net: final FID50k26.334 at50k,53.09 training minutes.
Attention improved the10k scout (29.327 versus historical31.555), and its best
5k-sample diagnostic was27.799 at30k, before worsening to30.792 at50k.
Plain U-Net's best diagnostic was29.379 at50k. Do not compare diagnostic
5k-sample FID numerically with final50k-sample FID as equal estimators.

Keep the plain U-Net no-argument default, exact lazy-4 bcap, cached frozen
ResNet18 conditioning features, fused Adam, batch64 and constant LR. The longer
baseline config is configs/cifar_ddgan/duration_50k/baseline.yaml. It has already
completed; use a fresh output path for a new experiment. Attention remains
optional via g_attn_resolutions:[8,16], with g_heads:4. Core ParticleGAN/DDGAN,
joint UCD, particles, Gaussian step noise and regularizer formulation are intact.

The useful next proposal is:
1. Save periodic/best checkpoints so a promising intermediate point can be
   evaluated at50k generated samples. Current checkpoint.pt holds only the last
   evaluation; attention30k weights were overwritten, although grids remain.
2. Validate attention around30k and test a gentler learning-rate tail against
   constant LR. The trajectory motivates this test, but does not prove that
   annealing helps or that the late regression is overfitting.
3. Rank final50k-sample FID, training time and trajectories. Do not promote from
   a selected diagnostic minimum or an attractive1k result alone.

These are proposals, not queued work. Both GPUs are free. All four experiments
from this round completed successfully. No seed-only repeats; maintain fresh
YAMLs, source provenance and tail-friendly logs. The user wants experiment
updates only after completion. Avoid extra runtime check-ins.

Historical logs:
tail -F results/cifar_ddgan/duration.live.log results/cifar_ddgan/attention.live.log

The earlier capacity round did not support G-width doubling or ResNet34.
FD remains experimental and is not promoted; retain the existing bcap objective.
NCSN++ optimization failed10k validation; do not automatically resume its old
restart-interrupted run. No1200-epoch training is proposed.

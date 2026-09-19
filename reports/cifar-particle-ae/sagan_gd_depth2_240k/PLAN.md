# Grow SAGAN attention in G and D from the best checkpoint

User authorized stopping the plateaued continuation and increasing attention; then explicitly requested scaling D attention too. Resume certified 200k best (FID50k 12.53445669) and continue to 240k on GPU 1. The unchanged trajectory already supplies 205k–240k controls, with 240k FID50k 13.1283. Preserve all historical trainers and checkpoints.

Add a second sequential attention block at 16x16 to G and to D's trainable pixel branch. Each adds 5,120 parameters; G becomes 936,003 parameters. Frozen pretrained D feature branch unchanged. Both new blocks have zero output projection, nonzero query/key/value, fixed unit residual coefficient and no schedule. Initial G and D functions are preserved. Projection learns first, then the other new projections receive gradients. All existing model/EMA weights, optimizer slots, and RNG streams are restored; only new parameters receive fresh Adam state.

Same prior, sigma, learning rates, objectives, E-only reconstruction and lazy bcap every 8 steps. FID50k before new training and every 5k; keep checkpoints. Validate real checkpoint identity, optimizer migration, active D double backward, 16-update full-size GPU smoke, then 8-update resume of the expanded checkpoint. These are validation runs, not benchmark or seed experiments.

Compare matched-step FID and runtime with the existing stopped control and compare best expanded FID to the parent. Generate results, leaderboard, checkpoints, findings and recommendations at completion. Joint G/D change cannot isolate which side contributes; checkpoint growth does not measure scratch-training potential. No automatic extension beyond 240k.

Tail: `tail -F runs/cifar_particle_ae/sagan_gd_depth2_240k/PIPELINE.log`

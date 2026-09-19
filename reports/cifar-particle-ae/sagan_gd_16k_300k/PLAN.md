# SAGAN unchanged continuation: 200k to 300k

User authorized continuing SAGAN. Resume the certified 200k checkpoint (FID50k 12.53445669) on GPU 1 for 100k additional steps. Preserve the parent and all 5k checkpoints. Restore models, Adam, EMA and RNG; unchanged architecture, objectives, learning rates, 16,384 particles and fixed sigma. Use the historical SAGAN trainer without source changes.

FID50k every 5k steps; reconstruction evaluation uses the existing 10k samples. Full-state preflight verifies source/config certificate and checkpoint SHA. Compare the late curve and best checkpoint against the 200k parent; generate leaderboard, findings and recommendations after completion. No automatic extension beyond 300k. Estimated about 90–100 minutes from the preceding continuation; runtime may vary.

Tail: `tail -F runs/cifar_particle_ae/sagan_gd_16k_300k/PIPELINE.log`

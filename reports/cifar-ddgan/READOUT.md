# Current CIFAR results and next priority

**Best completed FID:26.680**, U-Net32 + pretrained D at50k updates,99.34 training
minutes. The fast10k default remains31.741 in20.05 minutes. Both are final FIDs
using50k generated samples.

NCSN++128 was interrupted by a host restart after step27,200. Its last saved
checkpoint is20k, with diagnostic FID65.209 using5k samples (240.594 at10k).
There is no final score for that run. Nothing is running or queued.

[Scaling results and interrupted-run handoff](scale_50k/READOUT.md)

[Earlier moonshot leaderboard and findings](moonshot/READOUT.md)

[Next priority: profile and optimize training speed](NEXT_ROUND.md)

Keep joint UCD, learned latent particles, Gaussian step noise, T4, Rp logistic,
candidate-only bcap, unique-row VICReg and the shared train_denoising.py formulation.
No new default promotion or automatic restart is planned.

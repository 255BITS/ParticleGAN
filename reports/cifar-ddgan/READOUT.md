# Current CIFAR results and next priority

<!-- superseded-banner: generated -->
> **Superseded — historical record.** Current status: [reports/forge/technique-inventory.md](../forge/technique-inventory.md).
> Results below are preserved as recorded; statements about "current" selections, defaults,
> leaderboards, queues or GPUs describe the time of writing.

**Best completed FID: 26.680**, U-Net32 + pretrained D at 50k updates, 99.34 training
minutes. The fast 10k default remains 31.741 in 20.05 minutes. Both are final FIDs
using 50k generated samples.

NCSN++128 was interrupted by a host restart after step 27,200. Its last saved
checkpoint is 20k, with diagnostic FID 65.209 using 5k samples (240.594 at 10k).
There is no final score for that run. Nothing is running or queued.

[Scaling results and interrupted-run handoff](scale_50k/READOUT.md)

[Earlier moonshot leaderboard and findings](moonshot/READOUT.md)

[Next priority: profile and optimize training speed](NEXT_ROUND.md)

Keep joint UCD, learned latent particles, Gaussian step noise, T4, Rp logistic,
candidate-only bcap, unique-row VICReg and the shared train_denoising.py formulation.
No new default promotion or automatic restart is planned.

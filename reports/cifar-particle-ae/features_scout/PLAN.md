# Pretrained feature replacement and selective G reconstruction

User accepted the proposed next experiments after all four capacity-growth scouts completed. Use the same original one-D 50k checkpoint and both GPUs through the pipeline. Target remains generation FID50k below 13. No seed experiments or automatic long promotion.

## Hypotheses and arms

1. **control:** unchanged original architecture and objective.
2. **resnet34:** replace the frozen ResNet18 discriminator feature extractor with ImageNet ResNet34 through layer3, retaining the existing learned pixel branch, feature heads and their Adam state. Feature tensors have matching dimensions but different coordinates. This changes D scores immediately: the test measures replacement and subsequent joint adaptation, not an isolated function-preserving capacity increase. No extra D warmup or head reset.
3. **grow_g_adv:** add the same three identity-initialized G refinements as in the prior growth round, but block reconstruction gradients only on their parameters. Reconstruction still flows through the branches into old G, E and prior. New parameters receive adversarial gradients. This preserves the reconstruction forward function; it does not skip or detach the branch activations.
4. **both:** combine ResNet34 and selective G-growth routing.

The prior all-gradient G expansion ended at FID22.8981 versus control19.2033, with better reconstruction. That is historical context for selective routing, not a new contemporaneous all-gradient control. Neither experiment proves a unique cause if it succeeds or fails. In particular a failed pretrained swap could reflect head adaptation to changed feature coordinates.

## Protocol

Parent `runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt`, SHA256 `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`, FID50k18.9012. Restore full old G/E/prior, EMA, trainable D, optimizer and training RNG state; only the named interventions change. Separate forked CPU construction does not advance training RNG. Frozen replacement parameters are excluded from Adam. Existing G refinements copy into EMA and start as identities.

50k→70k G updates, one D update per G, lazy exact bcap every8 D updates with coefficient×8, unchanged learning rates and reconstruction weight. FID50k every5k on EMA prior samples against the fixed CIFAR train50k Inception reference; reconstruction diagnostics on the 10k test split; full checkpoints each evaluation. Rank final70k endpoints and report intermediate minima separately, with train/wall cost and factorial interaction. No automatic long continuation.

## Validation and execution

New standalone `experiments/train_cifar_ae_features.py` preserves historical source certificates. Tests verify selective routing changes only new-G parameter gradients, preserves the forward result and old-G/E/prior gradients, and preserves the prior-built adversarial graph. CUDA tests check real-parent backbone replacement retains G/EMA and learned D/Adam state, frozen features, unchanged RNG, and finite double backward. The combined model must also pass deterministic8vs4+4 full-state resume.

Four real-parent16-update smokes run through the pipeline before scout launch. Runtime `backbone_audit.json` records the initial D output/input-gradient discontinuity and unchanged trainable D; `growth_audit.json` records identity-preserving G insertion and double-backward compatibility.

Command: `bash experiments/cifar_ae_features_pipeline.sh features_scout 0,1`

Tail: `tail -F runs/cifar_particle_ae/features_scout/PIPELINE.log`

Automatic report: `reports/cifar-particle-ae/features_scout/LEADERBOARD.md`, with `leaderboard.json` and `curves.png`.

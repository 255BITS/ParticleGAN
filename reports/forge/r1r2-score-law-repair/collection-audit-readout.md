# Saved collection audit

**PASS** at execution `517257ff3879516ae2da64009c375ffffe9b001a`.
[Receipt](collection-audit.json) verifies all 21 certified cases and 21 actual-training
GIFs: 14 exact cached controls and seven new runs. It checks 18 ordinary terminal
checkpoints, three separate clock proofs with 78 saved snapshots, 51 optimizer
groups and 1,386 named RNG stream states against the original captured receipts.

Across all seven hosts, the new complete task-bound recipe differs from its exact
AMS m110 parent only in `loss`: `relativistic` → `non_saturating`. Actual saved
optimizer histories retain native AMSGrad, beta1 D/G+E/P = .5/.5/0, beta2 = .99,
LR = .0085 and epsilon = 1e-8. Both R1 and R2 remain active at gamma .1 every
update, with the learned prior, initialization, ordered target batches and sampling
contract preserved. The immutable public source binds native non-saturating
critic/generator losses and the joint word encoder's reversed real stream.
The scheduled baseline retains its original schedules.

All 203 historical charges remain unchanged. The seven new attempts cost
182.87967788797687 seconds, giving 210 aggregate attempts and
6405.0795450999285 paid seconds within the shared 8,000-second ceiling; cached
controls add zero paid cost. Archive stdout:
`/mnt/ml7tb/ParticleGAN-forge/r1r2-score-law-repair-20261012/audit-collection.log`.

The frozen [auditor](audit_collection.py) adds zero updates or draws and leaves
Python, NumPy and Torch CPU RNG states unchanged. Original independent grades,
metrics and GIF inputs are preserved; this receipt does not qualify or rescore
results. Initialization/data identities use captured receipts. No historical
per-update paired scores, force decomposition or sampled-row ownership is
reconstructed; EMA moments do not recover instantaneous G/D gradients.

# Saved collection audit

**PASS** at execution `3cb46b91121f42f8b171a77f17776365aebad1d8`.
The [receipt](collection-audit.json) verifies all 21 certified cases and 21
actual-training GIFs: 14 exact cached controls and seven new runs. Coverage
includes 18 ordinary terminal checkpoints, three separate clock proofs with
78 saved snapshots, 51 optimizer groups and 1,386 named RNG stream states.

Every complete task-bound recipe differs from the measured non-saturating
AMS m110 parent only in `reg_coeff`, **.1 → .01**. Both positive R1/R2 penalties
remain active every update. Actual native AMSGrad histories retain beta1
D/G+E/prior .5/.5/0, beta2 .99, constant LR .0085 and epsilon 1e-8. Frozen
source/runtime/task contracts, captured initialization and ordered target-batch
receipts, prior/sampling laws, original grades and GIF inputs all match their
certified evidence. The scheduled reference retains its original schedules.

The original weak-penalty result remains **2/6 ordinary PASS plus clock PASS**;
Gaussian, pole, Ring and words fail. All 210 historical charges and owners remain
unchanged. Seven new attempts cost **181.08289201298612 seconds**, giving
**217 aggregate attempts / 6586.162437112915 paid seconds** within the shared
8,000-second ceiling. Exact caches add zero current cost; no scientific retry
occurred. Stdout is archived at
`/mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012/audit-collection.log`.

The frozen [auditor](audit_collection.py) is unchanged from execution and adds
zero draws or updates; Python, NumPy and Torch CPU RNG states remain unchanged.
This receipt preserves stored independent grades and supplies no requalification
or rescoring. Initialization/data identities use captured receipts; no historical
score pairs, force decomposition, sampled-row ownership or instantaneous G/D
gradients are reconstructed.

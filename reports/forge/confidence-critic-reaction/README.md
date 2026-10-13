# Extra critic reaction: analytical rejection

One additional stop-gradient critic reaction failed its saved-state admission
prerequisites. This note preserves a useful negative result without presenting
an analytical proposal as an executed experiment. **New GPU attempts, trainer
steps, optimizer calls, sampling draws and budget: all zero.**

The ordinary next update already takes D0 to D1 before G. The proposed extra
reaction takes D1 to D2 and holds D2 fixed through the G derivative. Comparing
D0 with D1 would measure the ordinary update, not the proposed mechanism.

The corrected task law has 256 uniformly weighted MoG components, latent
dimension 2 and isotropic sigma 0.1. Earlier correlated sigma 0.3 surrogate panels
are retained as diagnostics and cannot establish actual task-law gradients.
The paid trials used the correct sampler; their grades remain unchanged.

Four frozen lineages at updates 1000/4000/6000 were examined using two fixed
panel pairs and 9x9/13x13 Gauss-Hermite integration per component. The tensor
proposal uses the actual zero-momentum DualNorm polar/row-normalization law
and BCAP kernel. Cached D/G panels, normalized parameter displacement and
finite-proposal effects remain distinct from a stochastic training trajectory.

| Retained fixture | W2 squared before | Ordinary D1 guidance | Extra D2 guidance | Reusable finding |
|---|---:|---:|---:|---|
| Baseline 6000/panel0 | .274910 | .408767 | .241337 | W2 improves while excessive variance expands |
| Baseline 6000/panel1 | .274910 | .401545 | .367392 | Extra guidance still worsens mean, variance and W2 |
| Positive-excess 1000/panel1 | .032538 | .026806 | .036396 | Extra reaction reverses a repairing acquisition proposal |
| Positive-excess 6000/panel1 | .005433 | .004056 | .002749 | A local terminal improvement does not satisfy the full prerequisites |

The fixed prerequisite required repairing both wrong late mean/variance
directions without spoiling acquisition on either panel and either integration
resolution. Four terminal falsifying cases and the acquisition reversal reject
this exact stop-gradient correction. All 144 central-difference derivative
checks agree in sign with autograd; the integration resolutions have no W2
direction-sign disagreement. This numerical agreement strengthens the local
analysis and supplies no trained-candidate qualification.

The [compact receipt](receipt.json) retains the original decision, numerical
checks, acquisition falsifier, zero-work counters and archive hashes. Full
functional evidence, cached data and proposed restore contract remain in the
original archive by those pins. No implementation is claimed to pass that
contract. Differentiating through a virtual reaction map would be a different,
unmeasured formulation; it was neither implemented nor admitted here.

For the executed acquisition/retention results, use the
[confidence screen](../generator-confidence-lr-bcap-direction/README.md) and
[response variants](../generator-confidence-lr-response-variants/README.md).
The practical lesson is to test restoring mean/variance directions and
acquisition regressions before spending on this particular reaction hypothesis,
while keeping analytical evidence separate from original full training gates.

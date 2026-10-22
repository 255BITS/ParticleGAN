# BISPF first-principles successors

This bounded follow-up investigates three different parts of the measured
failure: the settling signal, the restoring force, and the finite update.
The [first comparison](../three-theme-repairs-20261022/README.md) repaired
anisotropic vectors/blobs with energy distance and overlap with quartic
transport, while Gaussian stability, local tails/shape and native accuracy
remained failures. Its complete evidence is preserved under its original source.

| Perspective | Concrete mechanism | Cheap decision |
| --- | --- | --- |
| Statistical equilibrium | [Local exchangeability thermostat](settling/README.md), resolving rare support while distinguishing sampling fluctuations | **REJECTED before implementation/training.** AUROC .75 misses the fixed .8 admission requirement; non-reference passing states remain hot. Zero GPU spend. |
| Finite-step mechanics | [Same-batch secant extragradient](mechanics/README.md) of the unblended normalized G/prior field, then existing anchor direction protection | Corrected raw-field controls pass; preserve the original protected-field false certificate as a negative result. No joint-game convergence claim. |
| Information geometry and transport | [Learned Bregman c-transform game](fidelity/README.md), using critic prices and stable pre-sigmoid restoring forces | Saturation, routing, constant-law, public API and checkpoint controls pass. Finite-batch prices and shared parameter motion remain risks. |

The thermostat's failure distinguishes distributional equality from practical
approximation: two laws can both satisfy the original gates while being
statistically distinguishable. Its local support sensitivity does not fix that
problem. The mechanical prototype exposed a different error: anchor direction
protection can reflect a reversed trial force and conceal overshoot. The
corrected definition checks the raw normalized field before applying that
protection. These negative controls were retained without changing their
thresholds or coefficients.

[Reproduction and log commands](REPRODUCE.md) pin the actual execution source.

The [protocol](protocol.json) admits at most four paid arms: the original
BISPF control, the first-round energy winner, and one successor for mechanics
and fidelity. Each uses all fifteen declared tasks, protocol seed 0 and the
public deterministic initializer, with original architecture, data law and
consumed streams, initial prior/capacity, sampling, full update allowances,
evaluation cadence and numerical gates. The mechanics parent is the energy
winner; the game parent is original BISPF. Each is one global recipe, with no
per-task selection. Reservations total 112,080 seconds and at most 60 paid
attempts, with no scientific retries or seed/constant sweeps. Failed own
checkpoint prerequisites remain unexecuted BLOCKED cells.

All arms must freeze one common committed source/runtime before execution.
The game explicitly replaces both effective adversarial payoffs even though
the parent `loss=non_saturating` compatibility field remains. Its effective
payoff is `learned_bregman_c_transform_v1`; its new Boolean is part of the
declared recipe. Prior learning and SBNP .01 remain active in every admitted
arm, and no task prior width/weight or scoring law changes.

The common adapter retains scalar public step returns **after the complete
update** in hashed external JSONL. This includes earned thermal scale, each
active repair's diagnostics, and explicitly returned movement measurements.
It adds no sampling draw or model forward. After source freeze, a software-only
portability fix makes future trace paths relative to their adapter artifact
directory; the executed campaign retains its original absolute-path receipts.
The publisher accepts both versions. No trained result is rerun or rewritten. Optional optimizer diagnostics
have their own scope; a pre-policy optimizer proposal is not silently called
the final applied move. Bulk stdout, traces, tensors and checkpoints remain
outside Git at:

```text
/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022
```

The [publisher](publish.py) checks certified results against requests, full
source manifests, resolved recipe deltas, runtime, initial tensors/priors,
consumed named streams, direct Gaussian data hashes and the unique paid-cost
ledger. It keeps unexecuted prerequisites separate from scientific attempts.
Actual-training GIFs will render only retained observations, after numerical
results are available. This diagnostic supplies no ordinary qualification,
default adoption or merge. The sole primary goal leaderboard remains the
[current technique inventory](../technique-inventory.md).

# BISPF first-principles repair comparison

The two new mechanisms repair different failures, but neither is a suitable
replacement for the global trainer. The Bregman game repairs overlap and all
three failing images while losing ring acquisition and mode retention. The
finite-step correction improves Gaussian shape and native precision but also
loses those two passing controls. The earlier energy arm retains all four
controls and remains the stronger balanced research reference. No candidate
solves Gaussian stability, rare density or native accuracy.

| One global recipe | Fifteen-task result | Original failures repaired | Original controls retained |
| --- | --- | --- | --- |
| Original BISPF | 4 PASS / 11 FAIL | None | 4/4 |
| Energy reference | 6 PASS / 9 FAIL | Anisotropic, blobs | 4/4 |
| Secant extragradient | 5 PASS / 10 FAIL | Anisotropic, overlap, blobs | 2/4 |
| Bregman game | 6 PASS / 9 FAIL | Overlap, bars, blobs, intensity | 2/4 |

This is a fixed research panel, not an ordinary qualification result or a
selection of different recipes for different tasks. The [technique
inventory](../technique-inventory.md) remains the sole primary goal leaderboard;
its selected scientific rows and archived qualification are preserved. The
[first comparison](../three-theme-repairs-20261022/README.md) retains its original
source and evidence. Both fresh reference arms reproduce all fifteen of its
final metric packets and gate decisions exactly.

The [complete numerical appendix](numerical-results.md) binds every task,
unchanged gate, cost and actual-training GIF. Detailed saved-state diagnoses
are in [temporal behavior](settling/MEASURED.md), [finite-step mechanics](mechanics/MEASURED.md)
and [Bregman geometry](fidelity/MEASURED.md). GIFs illustrate actual retained
training observations; the numerical gates decide outcomes.

## What the three perspectives established

| Perspective | Registered mechanism | Measured decision |
| --- | --- | --- |
| Statistical equilibrium | [Local exchangeability thermostat](settling/README.md) | Rejected before training: AUROC .75 misses the fixed .8 screen, and all three non-reference passing states stay hot. Zero GPU spend. |
| Finite-step mechanics | [Same-batch secant extragradient](mechanics/README.md) of the unblended normalized G/prior field, then existing anchor protection | Accepted steps satisfy the local frozen-critic certificate, but Gaussian stability and native accuracy still fail. Ring and mode retention regress. Reject this exact global revision as a replacement. |
| Information geometry and transport | [Learned Bregman c-transform game](fidelity/README.md) with critic prices and stable sigmoid pullback | Sustained image and overlap gains, but missing rare mass, local underdispersion, worse native precision and two control regressions prevent replacement. |

The cheap thermostat study separates equality from useful approximation. Two
laws can satisfy the practical gates while remaining statistically
distinguishable. A sensitive two-sample test is therefore insufficient as an
equilibrium thermostat. Its null/support controls and failed admission screen
remain recorded; it was not silently replaced by another thermostat.

The mechanics study separates a local integration certificate from equilibrium
of the coupled game. Its corrected field check holds the discriminator fixed;
the discriminator still takes its ordinary update. An explicit bilinear
negative control shows that this construction does not damp a joint rotating
game. A rejected protected-field prototype also remains visible: direction
protection could reflect a reversed force and conceal overshoot. The admitted
method checks the raw normalized field first, with the same fixed coefficients.

The game study separates a small transport-like minibatch cost from target-law
fidelity. Finite neural prices, nearest-row routing and globally normalized cost
do not certify local component shape or a converged population transport solve.
Its changed pricing, pairing and sigmoid pullback form one registered mechanism;
the image gains do not identify which component caused them.

## Shared failure modes and practical implications

**Settling is still a coupled dynamical problem.** Mechanics improves Gaussian
stationary passing checks from the energy reference's 12/72 to 40/72 and shifted
hold checks from 0/24 to 12/24. Its first observed post-shift pass arrives after
42 updates, yet it fails the original deadline/retention rule. Its endpoint
passes all three output bounds; that does not replace the failed temporal gate.
The game also improves residual shape after fitting panel mean/width, while
centering and sustained retention fail. All 28,800 observed Gaussian/overlap
returns across the four arms have thermal scale exactly 1, so attenuation does
not explain those cells. All 37,000 game updates are also fully hot. Mechanics
does cool on some other tasks, including grid, rotated and mode hold; its
remaining failures cannot all be attributed to an inactive thermostat.

**Occupancy and local geometry need separate evidence.** Game native holdout
precision is .07912/.06664/.11210 on grid/rotated/staggered, versus original
.30497/.22301/.33617. Some nearest-cell mass measures improve at the same time.
These are large continuous regressions on already-failing tasks. Conditional
shape statistics remain null when any component lacks two in-radius samples;
nulls do not establish point collapse or certify covariance. Mechanics improves
native precision substantially. Grid and rotated still fail all four accuracy
bounds; staggered passes centering but fails mass, trace bias and radial shape.

**Removing tails does not guarantee persistent component shape.** The game's
unequal-width covariance error falls from 9.68690 to .48856 and all final
component >4-sigma tail counts become zero. Its minimum eigenvalue ratio is
.14738 below the unchanged .15 bound, and its terminal pass suffix is zero.
Its 2% rare component is absent. Mechanics instead retains rare samples but
fails their local minimum eigenvalue bound. These are distinct residual errors.

**Passing controls expose global tradeoffs.** Both new methods lose ring16 and
mode hold. Game mode hold ends with only 3/8 modes and HQ .31543; mechanics ends
with all eight modes and passing HQ but has only three terminal passing checks
where five are required. Ring fails full component covariance for both. A total
pass count would hide these differences, so neither method is adopted globally
or combined into a taskwise winner mixture.

Keep the existing selected recipe and the energy reference's evidence. Before
another paid panel, the most useful next controls are a coupled-player replay
that tests relative critic/G/prior speeds and a local-scale/price-dynamics check
that distinguishes cell occupancy from within-cell fidelity. In particular,
shrinking G/prior moves while leaving critic moves ordinary changes relative
kinetics despite equal update clocks. This is a testable hypothesis, not a
causal conclusion from these runs. Price spread, canonical output force and
parameter motion are different measurements. Do not automatically stack the
repairs, tune per task, relax gates or launch seed-only comparisons.

## Protocol, provenance and verification

The frozen [protocol](protocol.json) contains four arms, each across all fifteen
tasks with protocol seed 0 and the public deterministic initializer. Task models,
data laws and consumed batches, initial priors/capacity, sampling, update budgets
and evaluation cadence remain fixed. Each candidate has one global trainer
configuration. Mechanics derives from the energy reference; the game derives
from original BISPF. Prior learning and SBNP .01 remain active throughout.
The game's effective payoff replaces both adversarial terms even though its
parent `loss=non_saturating` compatibility field remains; its declared effective
payoff is `learned_bregman_c_transform_v1`.

Scientific execution source is `a03373da3421303b89c99a39898b091e22910134`,
manifest `668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`
(1,292 files). Reporting/receipt portability fixes after that freeze do not
change trained evidence. The original campaign retains absolute-path trace
receipts; future adapters use paths relative to their artifact directory.

The [publisher](publish.py) verifies original requests, all source files,
resolved recipe deltas, runtime, initialized models/priors, named consumed
streams, direct Gaussian data hashes and the unique paid-cost ledger. Its
ownership audit verifies 110 global non-task fields across all fifteen tasks
per arm while retaining seven legitimate task-owned recipe fields. Smoke
continuations resume each arm's own terminal 1,000-update checkpoint after its
acquisition gate passes; that terminal state can itself fail output bounds.
No earlier favorable state substitutes for it. All 60 paid attempts completed,
consuming 17,213.254 seconds against 112,080 reserved, with no scientific retries
or seed/constant sweeps. The numerical appendix retains per-arm certified costs;
these shared-device times are not an isolated throughput comparison.

The common adapter records scalar public returns after the complete update,
without extra draws or model forwards. Canonical adversarial output/prelink
forces exclude prior regularization and are not realized parameter moves.
Bulk stdout, traces, JUnit, tensors and checkpoints remain outside Git at:

```text
/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022
```

[Reproduction and tail commands](REPRODUCE.md) pin source and artifact identities.
[Software verification](software-verification.json) records focused tests, full
CI and the separate local Python 3.14 limitations with unchanged-parent controls.
Independent saved-only reductions reproduce the temporal and game reports;
no diagnostic reduction trains or samples. All failures and negative controls
remain evidence. This study grants no qualification, default adoption or merge.

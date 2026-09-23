# Network-floor grid diagnostic

The isolated candidate `a3be165` uses the affine/square 100-mode host and
global G/D learning-rate floor 0.005, while the particle prior keeps recipe
floor 0.05. It also changes the shared gradient-penalty κ from 1.25 to 1.0.
Both changes are present in this run, so comparison with the archived
κ1.25/floor0.05 control cannot attribute an effect to either field alone.

| Grid100 live result | Archived H1600 control | κ1.0 / network floor 0.005 |
|---|---:|---:|
| First sustained five-check accuracy pass | step 7000 | step 7000 |
| Final 20k HQ | 0.9868 | 0.98265 |
| Final 20k mass TV | 0.0435 | 0.0447 |
| Final 20k center RMS / Gaussian σ | 0.1502 | 0.1441 |
| Final 20k covariance-trace bias | −0.0349 | −0.0387 |
| Final 20k radial KS | 0.0130 | 0.0163 |
| 100k holdout HQ | 0.98636 | 0.98156 |
| 100k holdout mass TV | 0.03565 | 0.03704 |
| 100k holdout center RMS / Gaussian σ | 0.13912 | 0.12381 |
| 100k holdout covariance-trace bias | −0.03706 | −0.03852 |
| 100k holdout radial KS | 0.01214 | 0.01455 |

The new run reaches all 100 modes by step 750 and keeps 20k mode mass TV at
0.0447. Live width and center remain unstable: covariance-trace bias swings
from −0.273 at step 750 to +0.455 at 1500, then stays +0.275 at step 5000.
The first live accuracy pass is step 6000; every terminal check from 6000
through 7000 and the independent 100k holdout pass. The original coverage and
accuracy gates both report PASS after relocation.

The complete 52-file RAM evidence tree was copied byte-for-byte to
[`artifacts/toy100-accuracy/network-floor-native/grid100-run`](../../artifacts/toy100-accuracy/network-floor-native/grid100-run).
Every copied file matched its RAM original by SHA-256, and the archived
23-file source bundle passed offline verification from the isolated floor
worktree. The RAM evidence remains intact. Rotated100 and staggered100 are
held pending a shared transfer recipe that passes all nine bottleneck hosts;
the floor-0.005 transfer candidate was 8/9 because `img_bars4` missed one
required sustained checkpoint. This grid-only diagnostic is not common-22
evidence.

## Floor 0.01 follow-up

The predeclared floor-0.01 variant changes only the name and G/D floor from
the floor-0.005 configuration. It passed the nine frozen transfer bottlenecks,
but the full 19-host replay was 18/19: `residual_student` failed. Its matching
grid100 native run passed both individual gates. Live 20k final HQ was 0.98285,
mass TV 0.0447, center RMS 0.1291σ, covariance-trace bias −0.0378, and radial
KS 0.01663. The independent 100k holdout passed with HQ 0.98168, mass TV
0.03704, center RMS 0.11213σ, covariance bias −0.03767, and radial KS
0.01438. Live accuracy first passed at step 6000 and passed each of the five
terminal checks.

Because transfer failed, the all-three CLI was intentionally interrupted
after grid100 completed. `rotated100` was interrupted during step-0 evaluation
and `staggered100` was never started; both are explicitly excluded from
numerical success/failure counts in `interruption.json`. The complete RAM tree
was copied to
[`artifacts/toy100-accuracy/network-floor010-native/all3-run`](../../artifacts/toy100-accuracy/network-floor010-native/all3-run),
all 59 files matched by SHA-256, the 23-file source archive verified, and both
grid100 gates passed again from the isolated floor worktree after relocation.
The RAM original remains intact. This variant is also not common-22 evidence.

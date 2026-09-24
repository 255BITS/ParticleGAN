# Sampled-anchor output stability and internal state

This is a read-only assessment of the completed v2 warm and hold gates. It
does not run training, change a candidate, or promote a diagnostic. The
guarded post-native and pre-start variants each passed all 200 warm checks
and all 1,200 subsequent live checks through update 2,400: eight modes and
HQ 1 at every check. Their source-bound controls and the precise native
evaluation-counter reconciliation passed. Cold acquisition and continuation
from an independently acquired state remain separate gates owned by the
experiment driver.

## What changed internally

Both variants use the same sampled-group objective and target solver. The
pre-start variant resets G and the prior to their pre-G parameters before
the joint output fit; native GAN proposals remain separately evaluated.

| Diagnostic | Post-native fit | Pre-start fit |
|---|---:|---:|
| G parameter norm, update 1,200 → 2,400 | 15.7475 → 19.4083 | 13.2565 → 13.2337 |
| Prior norm, update 1,200 → 2,400 | 3.6795 → 3.8010 | 3.5746 → 3.5934 |
| D parameter norm, update 1,200 → 2,400 | 12.9413 → 13.2134 | 12.9431 → 13.2276 |
| Median first-GN largest singular value, first → last 50 active updates | 13.0687 → 86.2697 | 12.7261 → 12.7582 |
| Median first-GN condition, same windows | 8.9755 → 66.7130 | 8.7791 → 9.0640 |
| Median native-proposal anchor loss, last 50 updates | .0429175 | .0030667 |
| Median pre-G anchor loss, last 50 updates | .00244306 | .00244305 |

All 1,400 active fits converged and were selected in each hold run. All
recorded model and Adam tensors were finite. These are finite-horizon
observations, not a proof of bounded parameters. In particular, the
pre-start G Adam median denominator fell from .0007291 at update 1,200 to
.0003549 at 2,400: its corresponding coordinate metric increased. The
improved internal trace is not explained by a global decrease in the Adam
coordinate step scale.

Norm growth alone does not identify a nullspace mechanism. The separate
same-target diagnostic at update 2,401 provides local attribution. Relative
to the pre-step 24-output Jacobian, the native parameter displacement had
norm .0218796 and nullspace component .0208217. Fitting from that proposal
left a .0208407 nullspace component. Starting the same target fit from the
pre-G point reduced this component to 1.1054e-6, with both fits converged.
This establishes a local carried component; it is not a global nonlinear
nearest-parameter theorem or a proof that this component caused every
earlier instability. The exact input result hash and selected fields are
retained in `parameter-motion-extract.json`.

## Why pre-start removes the affine carried component

For a fixed full-row-rank output Jacobian J, native displacement v and
desired output displacement d, an exact minimum-norm correction from the
native point ends at

`v + J†(d − Jv) = J†d + (I − J†J)v`.

Fitting from the pre-G point instead produces `J†d`. It removes the
native component in the kernel of this **fixed linearization**. The
scientific test exercises a nonzero kernel component and verifies exact
output agreement, as well as zero movement for a zero target displacement.

## Output invariance does not imply parameter boundedness

The [conditional invariant-region note](anchor-invariant-region.md)
controls generated support locations under correctly matched, bounded
sample-centroid errors and the stated fit/selection rules. The observed
1,400-step traces had inferred K=8 throughout, all MM targets covered those
groups, and maximum centroid error .0992058. Substituting the recorded fit
tolerances gives an output radius bound .099358 and a conditional
population HQ lower bound .999309. Known ring centers enter only this
retrospective calculation. They are absent from the controller.
This is a conditional real-arithmetic interpretation of computed float64
losses, not a formal bound on every floating-point objective-rounding error.

These observed centroid errors are not bounds on all future Gaussian
batches. The calculation neither guarantees every future finite-sample HQ
measurement nor bounds G, D, prior parameters, Adam moments, or output
Jacobians. The guarded nonconvergence rule preserves the prior output
region by resting when the fit fails; it does not by itself establish
acquisition or a uniform bound on future fitting cost.

Even exact minimum-norm motion from the current parameters can accumulate
internal motion in a nonlinear chart. Consider

`F(x,y,z) = (x, u=y+xz)`, with `ker DF = span(0,−x,1)`.

The minimum-Euclidean-norm parameter velocity realizing `(dx,du)` is

`dy = (du − z dx)/(1+x²)`,

`dz = x(du − z dx)/(1+x²)`.

It solves the output differential and is orthogonal to the kernel, so it
is exactly the pseudoinverse lift. Rearranging gives

`d[z sqrt(1+x²)] = x/sqrt(1+x²) du`.

Lift the closed, bounded output rectangle
`(0,0) → (a,0) → (a,b) → (0,b) → (0,0)`.
The final output equals the initial output, but z increases by
`ab/sqrt(1+a²)`. Repeated oriented loops make z unbounded. For a=b=1,
each loop adds `1/sqrt(2)`. The tests verify the pseudoinverse velocity,
orthogonality, exact rectangle endpoints, and repeated fiber displacement.
This is a counterexample for the continuous minimum-norm lift, not a claim
that the finite-step neural host follows that example. It rules out
deducing a global parameter bound from output confinement and locally
minimum-motion GN alone. An exactly unchanged target permits rest; varying
sample centroids can trace loops.

## A condition that avoids this obstruction

If outputs have a fixed affine representation `Y=HW`, with H full row rank,
fixing a reference readout gives the path-independent section

`W(Y)=W_ref + H†(Y − HW_ref)`.

Bounded output targets then imply bounded readout displacement:
`||W(Y)−W_ref|| ≤ ||H†|| ||Y−HW_ref||`.
The kernel component stays fixed. With fixed H, successive exact
minimum-norm increments telescope to this same expression. A test verifies
path independence and return to the reference around a closed target loop.
More general nonlinear parameterizations would need an appropriate bounded,
path-independent section; one is not established for this neural host.

The copied-state feature diagnostic found full row rank for the native
12×97 hidden-feature-plus-bias matrix at every inspected state:

| State | Relative-cutoff rank | Condition | `||H†||₂` |
|---|---:|---:|---:|
| Cold update 1, pre-step | 12 | 132.2 | 30.56 |
| Cold update 100, accepted | 12 | 173.4 | 10.75 |
| Original passing update 1,324, pre-step | 12 | 560.9 | 48.16 |
| Guarded post-native fit, update 2,400 | 12 | 976.8 | 11.60 |
| Pre-start fit, update 2,400 | 12 | 610.1 | 53.07 |

Rank uses the same declared relative cutoff 1e-6 for all states. Native
float32 features are evaluated on copied models; SVD is float64. These
numbers establish representational feasibility for **fixed existing codes
and features**. They do not establish that training a frozen readout is a
good acquisition method: conditioning can amplify target changes, freezing
features/codes changes optimization and expressivity, and D remains
uncontrolled by this construction. No such controller is implemented here.

Finally, the successful output correction uses an explicit sampled-group
coverage objective and solver. Both recorded holds selected that fit on
every active update. This is evidence for that combined algorithm on the
fixed toy, not evidence that the original GAN objective alone has been
made indefinitely stable.

## Reproducibility

`anchor_drift_assessment.py` verifies archived sources, candidate and
disabled snapshot file hashes, full saved-state hashes, finite Adam
moments, and the recorded output/fit data. It never invokes a trainer.
`anchor_feature_chart_diagnosis.py` also verifies the existing cold and
passing snapshots and asserts that all input states and caller RNG are
unchanged. Seven focused scientific/integrity tests pass.

The evidence directory is
[`continuous-evidence/anchor-drift-assessment`](continuous-evidence/anchor-drift-assessment).
It contains both v2 assessments, the feature chart receipt, the one-state
motion extract, source/input hashes, and test output. The earlier v1
assessment is superseded; its incomplete parity epoch is not used for a
gate claim.

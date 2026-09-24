# Saved1530 convex readout and profiled-value diagnostic

One generator move reduces a numerically bounded profiled critic value and
improves paired noisy HQ from `.968994` to `.997803`, retaining all eight
modes. Scale `1/4` is accepted after scales `1` and `1/2` fail the value test.
This is one saved-state result, not a native training candidate or a hold /
acquisition qualification.

The result supports better critic response at this point. Refitting the
linear readout changes both the original-style non-saturating field and the
new value field from outward motion on the problem particles to inward
motion. It does **not** establish that changing G's objective alone repairs
the failure.

## Fixed problem and exact controls

The input is update 1530 from the exact original PR84 capture-v2, whose full
file hash is `37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
Sixteen frozen native-sized D banks give 2048 paired real observations,
particle indices and output-noise draws. Input noise is zero and output
noise is `.029`. The original first D gradient matches the captured Adam
first moment exactly; the actual next G bank is exact. A separate one-bank
control reproduces all G/D/prior parameters and both optimizer dictionaries
at the original accepted G boundary bit-for-bit (native HQ `.918457`).

The diagnostic then copies the native **accepted D** to float64 and freezes
its nonlinear features. Only the 96-dimensional final linear readout is
optimized. Its scalar bias stays fixed: it cancels from paired logits and
does not enter an input-gradient penalty. No new regularizer is introduced.
All fake-input paths through `b_cap` remain in the G graph.

The finite objective is

`L(w;theta) = mean softplus(w·[phi(fake_theta)-phi(real)])`
`             + coeff * mean_all relu(sqrt(||J_phi(x) w||²+1e-12)-kappa)²`,

where `mean_all` averages the 2048 real and 2048 fake points. Thus it is the
native sharp Rp D objective and native one-sided slope cap, restricted to a
fixed readout class. It is convex in `w`. The G value is

`V(theta) = log(2) - inf_w L(w;theta)`.

This is a profiled **paired finite-bank** value. It is not the exact
population distribution value or a complete real–fake Cartesian-product
expectation. The population analogue is convex in the generated law and
has a minimum at a matching law when constants belong to the critic class;
that observation does not turn this finite pairing, neural parameterization
or restricted features into a global distribution-learning theorem.

Cached and functional float64 losses agree exactly; the readout-gradient
maximum difference is `2.99e-15`. The saved post-Adam G/prior diagonal is
fixed throughout this diagnostic. No Adam state or moment is advanced by
the candidate. All original input tensors and caller RNG remain unchanged.

## Numerical global lower bounds

Let `S=mean_all J_phi(x)^T J_phi(x)`. For the native cap,

`relu(sqrt(||Jw||²+eps)-kappa)² >= .5||Jw||²-kappa²`.

If an evaluated critic loss gives upper bound `U` on the minimum, every
minimizer lies in the ellipsoid
`||w*||_S <= R = sqrt(2*(U/coeff+kappa²))`.
The constant critic gives `U<=log2` here. Convexity then supplies, at any
evaluated `(w,L,g)`, the lower bound

`L* >= max(0, L-g·w-R*sqrt(g^T S^-1 g))`.

This uses the original cap's coercive sublevel set, not an added L2 penalty.
At the base point, `S` has numerical rank 96, minimum eigenvalue`3.67e-6`,
maximum eigenvalue`4.629`, and a whitening identity error`1.72e-11`. Whitening
is an invertible solver coordinate change. A numerically deficient `S`
receives no such certificate. These are bounds calculated in float64,
not interval-arithmetic certificates.

Each point receives one L-BFGS attempt, at most 100 iterations and 200 closure
calls, with no restart, tolerance or coefficient search. The declared target
gap is `1e-7`. A final read-only gradient evaluation is counted separately.

## Retained V1 stop and the authorized V2 continuation

V1 fits the base readout from loss `.662731` to`.6162760631` in 111 closures /
100 iterations. Its lower bound is `.6162488645`, leaving a gap`2.72e-5`.
It therefore returns `UNRESOLVED_BASE_OPTIMALITY` and makes no G move under
its strict convergence gate. That result and exact source remain archived.

V2 keeps the same solver, budgets, gradient, metric and tolerances. Its base
fit must reproduce every V1 receipt field exactly. The explicitly authorized
change is to test disjoint value bounds even when the inner fits have not
reached the strict tolerance. If

`lower(L*_trial) > upper(L*_base) + c*alpha*sum(P_G*g_G²)`,

then the actual finite-bank profiled value decreases by at least that bound
difference. This inference does not require either optimizer to return an
exact minimizer. Both `NOT_CERTIFIED` statuses and their numerical gaps stay
visible. Here `c=1e-4` is fixed and the only proposed direction is
`Delta_G=-P_G*partial_G[-L(w_fit;theta)]`. At most nine fixed halvings are
allowed; there is no gain scan or target-quality acceptance.

| Point | D-loss lower bound | Evaluated upper bound | Remaining gap | Closures |
| --- | ---: | ---: | ---: | ---: |
| Base | .6162488645 | .6162760631 | 2.72e-5 |111 |
| Scale1 | .4110400939 | .4110415018 | 1.41e-6 |117 |
| Scale1/2 | .5937275997 | .5940225370 | 2.95e-4 |108 |
| Scale1/4 | .6288462471 | .6291821392 | 3.36e-4 |108 |

The accepted lower-bound improvement is `.012570184`, versus the required
`.0000040484`. There are 444 cached readout-gradient closures and four final
gradient evaluations in V2. Cache construction, native parity controls and
G derivative checks are additional work; this count is not a full-compute
equivalence to 444 native GAN updates.

## Field, derivative and quality attribution

The G partial derivative includes the fake-coordinate cap derivative. Two
central differences, at `1e-7` and`5e-8` of the fixed proposed displacement,
agree within `1.82e-9` relative error. Both comparisons have zero LeakyReLU
and cap-activation switches. Achieved displacement errors are below `3e-8`.
This checks the frozen-readout scalar derivative, not a total derivative
through the numerical optimizer or a Hessian of the full game.

All three diagnostic fields below use the same D banks; they are not three
training arms. Radial measurements refer to target centers **only after**
the update direction is fixed and never enter any decision.

| Field | Metric output RMS | Mean radial work | Inward particles | Work on particles 8 / 10 |
| --- | ---: | ---: | ---: | ---: |
| Original readout, non-saturating stencil | .04812 | +.000375 |3/12 |+.00720 /+.00347 |
| Fitted readout, non-saturating stencil | .41611 | −.04129 |11/12 |−.10857 /−.09598 |
| Fitted readout, negative full sharp D loss | .45941 | −.04167 |11/12 |−.10488 /−.09408 |

The refitted critic is the main observed sign change. The proper value and
its refit test provide a separate acceptance rule that rejects the two
oversized proposals. The accepted nonlinear move has clean RMS`.11662`,
mean radial work`−.01055`, and 11/12 particles move inward. Paired original
4096-draw noisy HQ changes`.968994 -> .997803`, with eight modes throughout.
Neither small value nor value descent is a general HQ-invariance proof.

## Next bounded test and native-implementation limits

The next authorized tests are the same one-move assay at independent warm
state 1325 and cold state 472. Cold 472 must report objective and allocation
response; it is not required to acquire all eight modes in one move.

A native implementation would need explicit moment ownership. One possible
sequence is a single D Adam update of its features, a base convex readout
fit, one G Adam proposal from the full sharp value field, and bound-verified
refits along that fixed proposal. The accepted trial readout materializes;
a rejected G move restores its parameters while retaining the one moment
update. That is **not yet implemented or validated**. The current diagnostic
uses a captured fixed metric and float64 copies. New-moment and native-rounding
controls must precede a native branch.

If D's nonlinear features change between updates, the optimized critic
class and hence the profiled value change too. Per-step descent then is not
one global Lyapunov proof. Permanently freezing the features supplies a
fixed class but requires its own cold acquisition/expressivity evidence.
The restricted feature class can also miss discrepancies. These limits
remain even after a successful local numerical value certificate.

## Evidence and tests

The [manifest](continuous-evidence/convex-profiled-value1530/manifest.json)
binds both versions' raw results, logs, generated V2 runner, every declared
source file, and tensor payloads. Payloads preserve every original 1530
capture phase, frozen banks, metric, features, initial/fitted readouts, full
G proposal and accepted G/prior state. V2 additionally stores the selected
trial readout. The original full capture can be rebuilt with the previously
archived stationary capture-v2 procedure; its hash is mandatory at input.

Six focused tests cover convexity, exact bias gauge, known global optimum,
exact zero-field rest, rank-deficient certificate rejection, disjoint bounds
with intentionally nonconverged inner points, and the guarded V2 transform.
No training or controller test is claimed from these mathematical controls.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_convex_profiled_value1530.py \
  --states FULL_CAPTURE_V2/selected-states.pt --output NEW_V1_OUTPUT

# Run with the same environment. V2 verifies exact replay of the V1 base fit.
python -u reports/toy100/pr84_convex_profiled_value1530_v2.py \
  --base-result NEW_V1_OUTPUT/result.json \
  --states FULL_CAPTURE_V2/selected-states.pt --output NEW_V2_OUTPUT
```

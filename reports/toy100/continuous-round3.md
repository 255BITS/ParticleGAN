# PR81/PR82 review and alternating fixed-target dynamics

No replacement has qualified. PR #82 identifies a real update-order confound:
the older ExtraAdam-derived joint-field adapters changed the host's D-then-G
update into a simultaneous update. Independently restoring alternation clears
trajectory acquisition, but neither PR #82 nor the new controllers below
passes all required acquisition and stability gates.

The task remains initial learning and sustained live quality on a **fixed
target**, without LR decay. Target-distribution shifts are separate. Rest is
allowed; no minimum movement is imposed. All work runs locally on the pinned
Python3.12.13 / PyTorch2.13 CPU environment, without waiting for GitHub CI.
No seed sweep, architecture change, budget extension or threshold relaxation
is used to turn a failed acquisition run into a pass.

## Independent review of the two other agents' PRs

- [PR81](https://github.com/255BITS/ParticleGAN/pull/81), head
  `e00de90c7d8075adb6f7ad2985931daa9816f667`: its target-error guard uses
  exact ring centers or trajectory identity targets. It is a diagnostic
  oracle, unsuitable as a general optimizer controller. Its reported closest
  cold result, MSE .02110, also fails the unchanged .02 sustained gate.
- [PR82](https://github.com/255BITS/ParticleGAN/pull/82), head
  `c1515197fdec44fc95c9f8d991e5a775421ef6be`: the imported alternating
  adapter passes exact full-host parity when its bounds are disabled. The
  parity comparison includes full optimizer parameters/moments, untimed
  results, global RNG and isolated noise RNG on both1200-update ring and
 400-update trajectory.

| Independent control | Cold trajectory | Cold ring |
| --- | --- | --- |
| Untouched alternating constant Adam | PASS .0034514, suffix15 | FAIL7 modes/HQ .36938 |
| PR82 with bounds disabled | Exact ordinary-host parity | Exact ordinary-host parity |
| Simultaneous Adam control | FAIL .254434,0/24 | Not run |
| PR82 G bound .25 / D bound3 | PASS .00094266, suffix18 | FAIL6 modes/HQ .89209,0/24 |

The PR82 bounded candidate passes the warm fork200/200, minimum HQ .94043.
Its ring outcome differs numerically from the other machine's archive, which
ends8/HQ1 but passes only3/24 checks. The archived config and adapter bytes
match; **both environments reject the method**. No seed was changed.

The update-order finding applies to the earlier simultaneous recorder family;
direct Adam-step wrappers such as the functional metric already alternated.
Simultaneous game methods are intentional algorithms, not invalid experiments,
but their failures cannot be attributed solely to their stabilization rule.
See the [independent audit](pr82-independent-audit.md) for raw hashes,
reproduction commands and the precise limits of the curvature interpretation.

## New configurations, stopped at their first failed gate

Warm stability checks every update1001–1200 after one matched scheduled prefix.
Cold acquisition removes decay from the beginning. Trajectory has400 updates,
MSE <= .02 and at least five consecutive passing observations. Ring has1200
updates and requires all eight modes/HQ >= .9 at every original terminal check.

| Configuration | Warm | Cold trajectory | Cold ring / first failure |
| --- | --- | --- | --- |
| Joint G+prior functional metric | 200/200 | FAIL .037413,0/24 | Not run |
| Output-coordinate RMSProp + joint Jacobian pullback | 200/200 | FAIL .041949,0/24 | Not run |
| Alternating G+prior trapezoid correction | FAIL196/200 | Not run | Warm minimum HQ .85107 |
| Network-only trapezoid correction | FAIL154/200 | Not run | Warm minimum HQ .48584 |
| Positive own-secant G response, D bound2 | 200/200 | FAIL despite final .001159: suffix4/5 | Not run |
| D Armijo, unchanged G .25, hard error on exhausted search | 200/200 | Execution stops at update66 | Bounded search encounters cap discontinuity |
| D Armijo with verified rejected-proposal rest, unchanged G .25 | 200/200 | PASS .0010046, suffix18 | FAIL5 modes/HQ .823,0/24 |
| Verified D and G Armijo steps | FAIL193/200 | Not run | Seven failures1130–1136; minimum HQ .82837 |
| D Armijo rest + positive own-secant G response | 200/200 | FAIL .035842,0/24 | Not run |

The hard-error execution and its declared rejected-proposal repair are both
retained; the former is not a completed acquisition result. A separate zero-update cold
launch with an uppercase candidate label was a harness setup error; changing
that label is not an additional candidate or an acquisition result.

All scheduled identity forks have exact uninterrupted-control state parity.
All matching ordinary constant-rate warm controls reproduce6/200. Nominal
rates remain G/D .00425 and prior .0085; proposal replays do not advance Adam
moments. The output-coordinate method adds its own explicitly recorded second
moments. No failed candidate advanced to longer hold or production common22.

The [machine-readable ledger](continuous-round3-results.json) derives these
eight completed candidate verdicts from archived results and keeps execution
errors separate. Local integration has **200 passing tests**:184 in the
[integrated suite](continuous-evidence/round3/integrated-tests.log), followed
by16 disjoint tests for subsequently integrated controllers in the
[additional log](continuous-evidence/round3/added-controller-tests.log).

Detailed mechanisms, tests and hashed evidence:

- [Joint geometry and output RMSProp](joint-functional-dynamics.md)
- [Alternating trapezoid corrections](alternating-heun-report.md)
- [Positive own-secant response](alternating-positive-secant-report.md)
- [Verified player steps and cap discontinuity](alternating-linesearch-report.md)
- [Verified D with positive own-secant G](alternating-armijo-secant-report.md)

## What the new isolation establishes

PR82's one-shot scalar shrink does not verify the scaled point. In an exact
replay of its G.25/D3 ring run, the effective accepted G curvature
`alpha * rho_at_accepted_point` exceeds `.25 * 1.01` on1150/1200 updates.
Its median is .81443 and maximum3.23971. The multiplication by accepted alpha
matters: raw rho in the original Adam metric does not itself shrink with step
length even for a quadratic. Independently,72 accepted G steps increase G's
own same-batch loss and66 accepted D steps increase D's. This motivates
checking actual candidate points rather than treating an endpoint secant as
a smoothness guarantee.

The bounded D line search exposed a second issue at trajectory update66.
Exact zero displacement reproduces total loss .4079834521 and every D gradient
bit-for-bit. At1/8192 of the proposal, the raw GAN loss improves but the cap
term jumps from .0119967461 to .0121780634; at1/65536 it returns to the
original branch. This is consistent with the discontinuous input-gradient
penalty across a LeakyReLU activation boundary, not changed training noise.
The repaired rule rejects an unresolved proposal only after verifying the
zero point; the next outer update again tries the full rate. It fixes execution
and retains trajectory acquisition, but does not solve the ring.

Finally, **own-loss descent is insufficient for live quality**. All seven
warm failures of the two-player Armijo controller satisfy both players'
sufficient-decrease inequalities with strictly positive slack even without
rounding tolerance. Its final checkpoint and all five sparse terminal checks
pass. Only the dense stability filter reveals the intervening seven-update
failure. The counterexample concerns alternating own losses at the opponents
used in each phase, not a claim that a joint game potential decreased.

## Remaining work and promotion order

The remaining issue is acquiring the complete ring within its unchanged
budget while keeping live quality through continued fixed-rate training.
Several methods can preserve a learned state; several can acquire trajectory;
none above does all required work. Removing the separate shift requirement
does not change these results.

The next bounded experiment reuses the earlier full-J implicit response on
an explicitly alternating field, so the generator field includes D's response
to generator movement. Its result will be recorded separately; it is not a
winner claim. The earlier full-J method passed warm200 but used simultaneous
fields and failed trajectory.

A survivor must still pass cold ring, the other cheap acquisition hosts,
an uninterrupted >=2400-update ring hold with noise horizon fixed at1200,
then older19, all three strict native100-mode cases, fresh production common22
and longer continuation. All scratch adapters remain ineligible for the
production gate until a real trainer implementation is audited.

For a future transplant, `GANTrainer.step` covers the ten vector/image hosts
and three native hosts. Seven additional legacy loops need their own bindings.
The cheap next custom host is `residual_student`. A production policy needs
explicit loss replay closures, noise/buffer restoration, once-per-update
moments and disabled-policy parity; global Adam interception alone does not
supply the required replay contract.

# Alternating line-search follow-up: accepted points and cap discontinuities

**No fixed-target solution.** Two mechanism-driven arms were tested at constant
nominal G/D learning rates .00425 and prior rate .0085. The discriminator-only
repair passes warm stability and cold trajectory, then fails the original cold
ring. Adding verified generator loss decrease fails the dense warm filter.
Neither reached the longer hold test. Distribution shifts were not tested and
are separate from this fixed-target question.

The [derived summary](continuous-evidence/alternating-linesearch-round3/summary.json)
and [hash manifest](continuous-evidence/alternating-linesearch-round3/manifest.json)
bind raw results, declarations, exact source snapshots and execution failures.
No seed search, horizon-dependent rule, new zero-centered penalty or relaxed
gate was used. All candidates remain `shared_gate_eligible=False`.

## Isolate the accepted-point problem first

This work starts at `d7d6cdf` and imports the four PR82 adapter/test files from
`266fed1` (equivalent local commit `da73749`). D takes its accepted step before
G's first gradient is evaluated. That preserves the host's alternating game.
Earlier simultaneous recorder adapters change this evaluation point; direct
Adam-step wrappers that already alternate are unaffected by that finding.

The PR82 G-curvature .25 / D-curvature 3 arm reproduces locally at final
6 modes/HQ .89209, with zero passing observations. The archived PR82 result
has a different ring trace, so it is not substituted for this local control.
[`alternating_linesearch_diagnosis.py`](alternating_linesearch_diagnosis.py)
adds two read-only, same-batch evaluations at the accepted points. All 24
production measurements and all 1,200 clean support traces match an
uninstrumented run exactly. Every extra evaluation checks original RNG
progression; model and optimizer updates are unchanged.

PR82 measures curvature at the full proposal, then shrinks it once without
checking the accepted point. For original metric `P`, accepted displacement
`delta`, and accepted gradient change `dg`, define

```text
rho_accepted = sqrt(sum(P * dg²) / sum(delta² / P))
effective accepted curvature = proposal_factor * rho_accepted
```

The proposal factor is essential: `rho_accepted` alone is independent of the
step fraction on a quadratic. Counts below use the **effective** value and a
1% comparison allowance.

| Diagnostic over 1,200 ring updates | D, bound 3 | G, bound .25 |
| --- | ---: | ---: |
| Accepted steps that increase their own current-batch loss | 66 | 72 |
| Effective accepted curvature above 1.01 times the original bound | 2 | 1,150 |
| Negative directional secant curvature | 3 | 193 |

G's median effective accepted curvature is .81443 and its maximum is 3.23971.
The original norm estimate therefore both misses some accepted-point
curvature and shrinks directions with useful negative directional curvature.
These measurements motivated the two bounded arms; no threshold scan followed.

## Arm 1: verify D, preserve the original G rule

[`alternating_linesearch_scratch.py`](alternating_linesearch_scratch.py) changes
only D's safeguard. Compute the ordinary Adam proposal once and require

```text
L_D(D0 + alpha * delta_D, G0)
    <= L_D(D0, G0) + .1 * alpha * grad_D(L_D) dot delta_D
```

The full current loss includes the existing cap. Trials reuse the actual
training minibatch and noise. Alpha starts at 1 on every outer update and is
halved at most twelve times. On a scalar quadratic, this sufficient-decrease
fraction leaves a positive-curvature margin of 1.8. G retains PR82's .25
norm-curvature rule exactly and responds to the D point actually accepted.

The first bounded execution stopped at trajectory update 66. This was an
execution limitation, not a quality verdict. The loss did not return smoothly
to its base level within the trial budget. A read-only replay reproduces the
first 65 records exactly and isolates the cause:

| D trial alpha | Raw GAN loss | Existing cap contribution | Total loss |
| --- | ---: | ---: | ---: |
| 0, exact base replay | .3959867060 | .0119967461 | .4079834521 |
| 1/8192 | .3959863186 | .0121780634 | .4081643820 |
| 1/65536 | .3959866762 | .0119966865 | .4079833627 |

At alpha 0, every D gradient also matches bit-for-bit. The small nonzero trial
improves raw GAN loss while crossing a discontinuity in the input-gradient
cap through the piecewise-linear critic. This is not an RNG mismatch. The
diagnostic is preserved in
[`alternating_linesearch_failure_diagnosis.py`](alternating_linesearch_failure_diagnosis.py).

One repair was predeclared: if the bounded search is exhausted, replay alpha
0, require exact base loss and gradient identity, and reject that proposal.
D's moments still advance once; G responds to unchanged D. The next outer
update again starts at alpha 1, so this is responsive rest with no time-based
attenuation. The retry count, coefficient and rounding tolerance are unchanged.
A separate zero-update setup error from an uppercase candidate label is also
archived, with no training or quality verdict attributed to it.

The repaired warm run matches the earlier warm run's final complete state
exactly: D needs no backtracking in either passing-state fork. Cold trajectory
needs 74 limited D proposals, including four zero rejections, and passes.
The ring has no zero rejections, but still fails acquisition and briefly reaches
zero modes at update 900. Safe D own-loss descent alone does not solve it.

## Arm 2: verify G at the same positive-curvature margin

[`alternating_two_player_linesearch.py`](alternating_two_player_linesearch.py)
retains the repaired D rule and replaces G's one-shot norm shrink with actual
same-batch Armijo decrease. Its fixed fraction is 7/8: on a scalar quadratic,
`2 * (1 - 7/8) = .25`, matching the previous positive-curvature margin.
Negative directional curvature can retain a full improving step. The same
bounded retry and verified-zero rejection policy applies. No parameter sweep
or target-coordinate access is introduced.

| Arm | Warm dense 200 | Original cold trajectory | Original cold ring | Hold >= 2,400 |
| --- | --- | --- | --- | --- |
| D verified, original G rule | PASS, min HQ .94043 | PASS, MSE .0010046; sustained 18/24, first 117 | FAIL, 0/24; final 5 modes/HQ .82300 | Not run |
| D and G verified | FAIL, 193/200; min 7 modes/HQ .82837 | Not run | Not run | Not run |

The second arm fails at updates 1130–1136, although all five original sparse
tail checkpoints pass. Both players satisfy sufficient decrease at **every**
failed update, with positive slack even without rounding tolerance:

| Update | D actual / required decrease | G actual / required decrease | G alpha |
| --- | ---: | ---: | ---: |
| 1130 | .0308427 / .00386878 | .00489759 / .00290830 | 1 |
| 1131 | .00592184 / .000665387 | .00296503 / .00253120 | 1 |
| 1132 | .0164339 / .00202143 | .000427425 / .000406084 | 1 |
| 1133 | .0279775 / .00315185 | .000607014 / .000574149 | 1 |
| 1134 | .00924242 / .00109776 | .000377178 / .000354114 | .5 |
| 1135 | .00943828 / .00116014 | .000745058 / .000643695 | 1 |
| 1136 | .00740439 / .000790204 | .00341260 / .00214631 | 1 |

D alpha is 1 in all seven rows. The smallest G slack is .0000213412, versus
rounding tolerance .000000476837. This isolates a substantive remaining
limitation: alternating own-loss descent does not imply live-quality stability
for this game. D is compared with G fixed at that update's base; G is compared
using the accepted D. The dense filter catches failures hidden by sparse checkpoints.

## Accounting, validation and reproduction

Each outer update advances each player's moments once, including rejected
proposals. Replays restore the original global, host and isolated noise RNG
states and immutable host buffers. G is evaluated after accepted D. All replay
callbacks and actual moment counters are recorded; nominal rates are checked
at every callback. Noise burn-in remains tied to 1,200 updates. The D-only
trajectory uses 1,410 field evaluations per player for 400 moment updates;
its ring uses 4,050 for 1,200. No longer run follows the failed ring.

Fourteen tests pass: five imported PR82 tests and nine new tests covering exact
alternating Adam/state/RNG parity, preservation of the original G rule,
analytic D-before-G backtracking, zero-field rest, rejection at nonsmooth
boundaries, the scalar G curvature margin, useful negative curvature and
actual host query/moment accounting. Independent read-only review checked
the D phase ordering and same-sample loss evaluation.

After local isolation, primary research was checked:
[ICML 2025 stochastic Armijo analysis](https://proceedings.mlr.press/v267/vaswani25a.html)
studies adaptation to local smoothness under stated assumptions;
[the November 2025 nonconvex extension](https://arxiv.org/abs/2511.20207)
analyzes adaptive stochastic steps under additional conditions. Neither
certifies this Adam-preconditioned, alternating neural game with a nonsmooth
input-gradient penalty. The experiments make only measured local claims.

Use the Python/one-thread CPU/AVX2 environment from the main handoff, fresh
paths, and the controllers' enforced filter order:

```bash
python reports/toy100/alternating_linesearch_probe.py --phase warm --output /tmp/d-warm
python reports/toy100/alternating_linesearch_probe.py --phase cold \
  --previous /tmp/d-warm/run/summary.json --output /tmp/d-cold
python reports/toy100/alternating_two_player_probe.py --phase warm --output /tmp/dg-warm
python -m pytest -q tests/test_alternating_curvature_scratch.py \
  tests/test_alternating_linesearch_scratch.py tests/test_alternating_two_player_linesearch.py
```

The current second-arm result deliberately blocks cold acquisition; the current
first-arm ring result blocks `--phase hold`. Neither is a fixed-target LR-decay
replacement, regardless of endpoint recovery or a passing trajectory task.

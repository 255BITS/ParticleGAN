# Alternating same-sample G trapezoid correction

Neither of the two declared scopes passed the cheap passing-state stability
filter. No cold trajectory, cold ring, extended hold, or shift was run. This
is a negative scratch experiment, not a production update or evidence that
all alternating predictor-corrector methods fail.

PR #82 identified a concrete confound: earlier simultaneous game adapters
altered the host's D-then-G order. This experiment retains that order and
the PR #82 D own-curvature bound at 2. D first makes its ordinary Adam step,
which is bounded using a same-sample replay. G/prior then make their ordinary
Adam proposal against the realized D. A second same-sample replay evaluates
the G field at the proposal. In Adam's current diagonal metric `P`, the final
proposal is the explicit trapezoid correction

```text
delta = -P g_base
delta_final = delta - 0.5 P (g_proposal - g_base).
```

This changes the **vector direction**, unlike PR #82's scalar G bound.
The replay restores all training RNG streams and does not update Adam moments.
The two scopes were fixed before comparing results: `joint` corrects the
G network and learned prior; `network` corrects only the network, leaving the
prior's ordinary Adam proposal. The latter was motivated by prior attribution
that late quality loss mainly follows network motion. Both use constant
nominal rates G/D `.00425`, prior `.0085`; no time clock, target center,
quality metric, or zero-centered pull controls an update.

| Warm variant | Passing checks 1001–1200 | Minimum HQ | First failed update | Final |
| --- | ---: | ---: | ---: | --- |
| Scheduled identity | 200/200 | .99634 | — | 8 modes, HQ1 |
| Constant Adam observer | 6/200 | .00537 | 1001 | 6 modes, HQ .50464 |
| Joint trapezoid | 196/200 | .85107 | 1076 | 8 modes, HQ1 |
| Network-only trapezoid | 154/200 | .48584 | 1014 | 8 modes, HQ .99902 |

The joint correction fails at 1076–1078 and 1160. Its correction averages
`.615` times the ordinary proposal norm and points against the proposal
(mean cosine `−.72`). At 1076–1078 its norm is about the proposal norm with
cosine near `−.99`, nearly canceling or reversing the ordinary update. The
failure at 1160 has smaller, almost orthogonal correction. Thus a stable
final checkpoint cannot certify the intervening training path, and the
vector correction has no guaranteed quality protection. Restricting the
correction to the network worsened the warm result, so there was no reason
to spend the next cold-host budgets on either tested configuration.

The identity child has exact full-state parity with the uninterrupted
scheduled control in both runs. The constant child reproduces the known
6/200 failure. Each candidate's 200-step continuation has 600 recorded
gradient calls per player (three same-sample phases) but only 200 new Adam
moment updates per player, for final moment step 1200. All actual rates in
the continuation are the declared constants. The D bound is inactive on
both warm continuations; the observed distinction is the G correction.
Four focused tests verify exact zero-correction parity with PR #82's
alternating adapter, the analytic G vector correction and D-then-G order,
one moment update per player, network/prior scope, and zero-field rest.
Together with five inherited PR #82 tests, `9 passed` in the matching
Python 3.12/PyTorch 2.13 CPU environment.

The [manifest](continuous-evidence/alternating-heun/manifest.json) hashes
both compact raw archives. Each contains the source copies and source hashes
at execution, effective warm evidence, per-update checks, full-state parity,
actual rate/call/moment receipts, and a tailable run log. The joint archive
preserves the initial source before the network scope was added; this is why
its adapter hash differs. To repeat with a new output directory:

```bash
python -u reports/toy100/alternating_heun_probe.py --phase warm --scope joint --output NEW_PATH
python -u reports/toy100/alternating_heun_probe.py --phase warm --scope network --output NEW_PATH
```

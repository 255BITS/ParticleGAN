# PR84 review and the leading partial candidate

**Latest update:** the original PR84 adapter also fails a conditional longer
warm continuation:112/120 checks through2400, first failure1390, worst7 modes/
HQ .78857, despite finishing8/HQ .99805. Its first200 passing updates reproduce
exactly. The [same-dataset stability review](stationary-stability-status.md)
supersedes the short-warm stability assessment below; the adapter is retained
as a research reference only. This is an additional diagnostic of the same
configuration, not a seventh round4 configuration.

**No no-decay replacement qualifies yet.** PR84's original smoothed-critic
update is the leading local research candidate among the reviewed PR81/82/84
arms: it passes the dense warm fork and cold trajectory, then ends the cold
ring with seven modes and HQ1.0. All eight modes are required. This is a
candidate to investigate further, not a proposed production replacement.

The scope remains initial acquisition and continued live quality on a fixed
target without LR decay. Distribution shifts are separate. All tests run
locally with the pinned CPU environment; no CI wait, seed sweep, threshold
relaxation or acquisition-budget extension.

| Method | Local warm | Cold trajectory | Cold ring / first failed gate |
| --- | --- | --- | --- |
| PR82 G.25 / D3 | 200/200; min HQ .94043 | PASS .00094266, suffix18 | FAIL6 modes/HQ .89209 |
| PR84 original G-only stencil, `40ecb67` | 200/200; min HQ .97021 | PASS .00094266, suffix18 | FAIL7 modes/HQ1.0;0/24 |
| PR84 head plus rest gate, `468ad26` | 200/200; min HQ .99414 | PASS .00094266, suffix18 | FAIL7 modes/HQ .99438;0/24 |
| PR84 rest gate, single-convolution repair | 200/200; min HQ .91602 | PASS .00094266, suffix18 | FAIL6 modes/HQ .96094;0/24 |
| Same frozen stencil for D and G | 200/200; min HQ .92261 | PASS .00094266, suffix18 | FAIL7 modes/HQ .78125;0/24 |
| Sampled-real coverage projection | FAIL197/200; min HQ .84619 | Not run | Stopped at dense warm |
| Bidirectional Chamfer projection | 200/200; min HQ .92188 | PASS .00094266, suffix18 | FAIL7 modes/HQ .87012;3/24 |

PR81's oracle target-error guard remains diagnostic and its reported cold
trajectory fails. The preceding [nine-candidate round](continuous-round3.md)
is closed: the final alternating-field implicit arm passes warm200/200 but
fails cold trajectory at .252398. It did not advance to ring or hold.

## What PR84 adds, and what reproduced

PR84 averages the critic score over a five-point spatial stencil during G's
update. D's own objective remains sharp. Width is at most .15, chosen from
critic sharpness; G/D curvature bounds remain .25/3. Nominal rates remain
G/D .00425 and prior .0085. These are responsive proposal corrections, not
an elapsed-time decay schedule.

The original report records warm196/200 and a cold-ring pass. Locally, exact
source `40ecb67` gives warm200/200 but only seven cold-ring modes at every
terminal check. Both evidence sets reject the full candidate. The reported
smoothed runs have no committed raw archive in PR84 from which to explain
the difference. We retain exact source versions and separate reported and
reproduced results. See the [independent audit](pr84-independent-audit.md).

The later rest gate contains a concrete implementation error: its manual
five-point slope estimate calls an already five-point-smoothed critic. It
therefore thresholds a twice-smoothed field, whereas G uses a once-smoothed
field. The original smoothed-only arm has no such gate. A byte-minimal
[implementation repair](pr84-rest-repair-report.md) keeps the existing .2
threshold, stencil width and curvature bounds frozen. It passes warm200 and
trajectory, then fails ring at6 modes/HQ .96094. No further tuning followed.

Both PR84 arms currently apply smoothing only to the 2D ring critic;
trajectory exercises their unchanged bounded alternating path. The best
partial candidate now has a [small research adapter](pr84-smoothed-candidate.md)
without the inactive oracle, time-boost or rest-controller branches. Exact
full-state parity against both original cold hosts passes, including optimizer
moments, EMA, RNG, metrics and every curvature/width record. The
[selection manifest](continuous-selected-candidate.json) freezes the source,
settings, failed gate and remaining qualification work for other agents.

## New consistency test

The [shared-stencil experiment](consistent-stencil-report.md) gives both
players the same operator, frozen through every replay. It passes warm and
trajectory but loses ring acquisition. This rules out that particular repair
under the unchanged gates; it does not prove a general smoothing theorem
wrong. The isolated code and five analytic/parity/accounting tests are kept
with hashed source and raw evidence.

## Current diagnosis and next filter

A [read-only final-state replay](pr84-field-diagnosis.md) of `40ecb67` reproduces every untimed cold
observation. It finds that the missing mode has the highest critic score,
yet the local gradients at its nearest generated particles point away from
it. The actual generator proposals follow those gradients; network motion
dominates prior motion. Along the path toward the missing mode, the critic
score first dips and then rises. This indicates a local gradient barrier,
rather than simply a step that is too large or a Jacobian reversing the
available direction. Known mode centers are used only to diagnose this
failure, never as a candidate update input.

The next cheap mechanism test should recover a useful acquisition direction
from sampled training data or critic queries across that barrier, while
allowing exact rest when its signal is zero. Test this on the captured state
before another warm/cold run. A positive diagnostic is not an acquisition
pass; the original warm200, trajectory400, ring1200, other cheap hosts and
uninterrupted >=2400 fixed-target hold still apply. Hold must preserve the
original1200-step noise horizon. Production common22 follows only a reviewed
trainer implementation and exact disabled-policy parity.

A [recent nonlocal transport rule](nonlocal-signal-filter.md) was first tested
on tiny fixed clouds. Its exact-matching field rests, but the distinct
eight-mode proxy loses quality, so it did not advance to GAN training. One
sampled-data [coverage projection](coverage-projection-report.md) recovers
the eighth mode in a clean final-state check, but fails three of200 dense
warm updates. It stops before cold acquisition. Every accepted correction
decreases its sampled coverage objective; that alone does not guarantee
quality of every generated particle. An [exact replay](coverage-failure-diagnosis.md)
isolates empty cells and one-sample outlier targets as the warm failure causes.

A [bidirectional Chamfer follow-up](chamfer-projection-report.md) repairs all
three saved warm states, then passes a fresh warm200 and cold trajectory.
It fails cold ring training: it first passes at update150 but finishes at
7 modes/HQ .87012, with only3/24 passing checks. Cold nonlinear target errors
are much larger than warm errors; some ideal targets also miss modes.
Both failures are retained for the next isolated test. No hold or other host
was run after rejection, and the original PR84 partial candidate stays selected.

[Machine-readable results](continuous-round4-results.json) bind the selected
candidate, six completed local configurations and evidence hashes. The
[integrated local suite](continuous-evidence/round4/integrated-coverage-tests.log) passes
230 tests in29.43 seconds, including all previous controller checks and
the new stencil, repair, extraction and coverage checks in one invocation.
The bidirectional helper and adapter add 11 passing tests in a separate
2.30-second invocation. The [latest report](chamfer-projection-report.md)
records the remaining issue and exact reproduction commands.

# Stress and dynamics: trained solvability witnesses

All six practical ranking stresses now have an actual GAN training witness under
their original numerical thresholds. The witnesses use different configurations.
**No shared configuration passed all six**, and the previously seen alternating
critic cadence remains unsolved in this search. This is development evidence,
not a new held-out evaluation or a production-default recommendation.

| Task | Trained witness | Original → actual outer steps | Actual D / G updates | Final passing suffix | Covariance error | Minimum eigenvalue ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fast critic | Prior LR multiplier 30 | 1,200 → 1,200 | 1,200 / 1,200 | 6/24 | .4542 | .2585 |
| Slow critic | G updates every second outer step; longer run | 1,200 → 6,000 | 6,000 / 3,000 | 11/24 | .2384 | .6403 |
| Small batch | Three times the update budget | 1,200 → 3,600 | 3,600 / 3,600 | 11/24 | .2828 | .2684 |
| Larger critic | Three times the update budget | 1,200 → 3,600 | 3,600 / 3,600 | 8/24 | .3803 | .3877 |
| Long horizon | Three times the task's update budget | 2,400 → 7,200 | 7,200 / 7,200 | 6/24 | .4580 | .1627 |
| R1+R2 | Stock-inspired larger setup | 1,200 → 3,600 | 3,600 / 3,600 | 16/24 | .1158 | .7558 |

The fast-critic witness changes only the prior LR multiplier from 10 to 30. Its
architecture, data, batch size and update budget remain unchanged. The slow-critic
witness retains the original discriminator LR multiplier **.75**; it gives the
critic more updates relative to G. Its live HQ is 100%, mass TV .13916 and
normalized sliced distance .04482. It first passes at outer step 3,500, confirms
the five-check suffix at 4,500, and remains passing through 6,000. EMA also passes
but does not contribute to the live verdict.

The larger R1+R2 setup uses 1,024 particles, width 96 with three hidden layers,
three Fourier bands, generator LR .0006 and prior regularization weight 1. The
task's R1+R2 arm and coefficient .1 are preserved. Cap-only overrides apply only
to cap tasks. This resource change is reported separately from tuning at the
original architecture and budget.

## What failed and what it suggests

The best shared card was simply three times the original update budget: it
passed four of the six ranking stresses and failed the seen cadence. The
stock-inspired card passed two ranking stresses. Prior LR 30 passed only the
fast-critic stress. Beta2 .99 and cap coefficient 10 passed none of the seven
complete cross-checks. The [full matrix](MATRIX.md) includes every failed and
untested cell; the [per-attempt table](attempts.md) retains all numerical outcomes.

The initial failures often had good global distance and high HQ but incorrect
component covariance. For example, the original fast-critic run had component
covariance errors 4.04, 13.01 and 7.92 in three components. Prior LR 30 reduced
the mean error to .4542 and kept all component errors below .971. More capacity
was not an automatic fix: the stock-inspired card reached about 99.6–99.7% HQ
on the small-batch and larger-critic tasks but failed the minimum-eigenvalue
bound. Keeping that width check exposed real narrowing along an axis.

Slow-critic followups also show the importance of update balance. More particles,
larger batches and 6,000 updates alone still failed (covariance error 8.34), as
did prior LR 3 at either 1,200 or 3,600 updates. Updating G every second outer
step at 3,600 achieved a final pass but only two passing observations. The
6,000-step version retained a passing suffix of eleven. These results support
an optimization-timescale explanation; they do not isolate a universal cause
or prove that one default works across tasks.

## Protocol and retained evidence

There were **66 actual episodes**: a 24-card fast-task screen, six additional
prior-30 cross-checks, 31 further finalist cross-checks, and five targeted
slow-critic attempts. The final planned sixth targeted attempt was not run once
a sustained witness was found. Four additional fast-task refinement ideas were
written down but never executed; they are explicitly separate from actual rows.

All episodes use seed 0, the original five numerical bounds, 24 fixed live
observations and a final passing suffix of at least five. EMA, action traces,
actual update counts and runtime remain separate in the raw results. No metric
thresholds or task tiers were changed. The original D-LR, small-batch,
capacity/horizon and R1+R2 perturbations are preserved when applying shared
cards. The targeted density control explicitly changes batch size, particle
count and budget and is marked as a resource change.

Target-distribution samples at finite evaluation size pass all seven scoring
cards, providing a scoring positive control. That control alone is not a GAN
witness; the trained results above provide the optimization evidence for the
six ranking tasks. The covariance/eigenvalue gates were inherited unchanged
from the corrected v2 suite.

Machine-readable [episodes.json](episodes.json.gz) includes each candidate,
original/effective specs, the complete result, and independently recomputed
final/EMA/sustained verdicts. [results.json](results.json.gz) retains the original
serial records. Exact source files, experiment scripts, phase cards and logs
are archived alongside them. Source checkout: `afe615264221eda47c5d1b7fd2cf4082552e30e9`.
Recorded episode wall time totals **888.82 seconds** on a shared CPU host.

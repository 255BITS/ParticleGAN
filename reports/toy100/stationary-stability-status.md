# Same-dataset stability status, September 24

**Critic refinement now passes the conditional same-dataset stability tests;
full acquisition and longer stability remain pending.** Earlier PR84 methods
fail a longer continuation from the same passing state. Their short warm
passes and excellent final checkpoints conceal failures in between. The
fixed dataset never changes.

**Subsequent exact diagnosis and new attempt:** the [fifth round](continuous-round5.md)
checks every update through2400. Original PR84 fails89 of1,200 later checks,
while a new G opponent-prediction rule fails22/1,200 despite finishing at
8 modes/HQ1.0. The scheduled control passes1200/1200. Original training state
and all shared archived observations remain exact; denser checks expose more
failures. Prediction stops before cold tests. A saved-state diagnostic finds
coherent outward G gradients across all16 held-out batches at1325, without a
sudden change in Adam's denominator. See the linked report for the complete
ledger, direction tests and exact evidence. The ten-step results below retain
the original diagnostic's provenance.

**Current candidate:** bounded refinement of the same penalized critic loss
passes all44 saved-state checks and warm200/200, minimum HQ .99707. Its
constant-rate dense hold through2400 passes1200/1200, minimum HQ .939453.
Cold trajectory passes at MSE .000910; the original cold ring aborts on a nonfinite inner-fit trial
at update472. No full ring quality verdict is available. An [independent exact audit](pr84-critic-refinement-failure-audit.md)
isolates float32 strong-Wolfe interpolation overflow. A [bounded finite-trial repair](pr84-critic-refinement-finite-recovery.md)
recovers the exact best finite critic, leaves44 passing saved-state checks bitwise unchanged,
and completes twenty saved-state numerical updates; these remain at three modes and do not
pass acquisition. Its full cold rerun reproduces trajectory and is testing ring acquisition.
This costs roughly53 extra
1024-pair D gradients per update; fits remain nonconverged. Cold acquisition
is not yet complete. [Independent warm/hold audit and evidence](pr84-critic-refinement-independent-audit.md),
[strict saved-state filter](pr84-critic-refinement-filter.md). The
[capped population toy](capped-critic-tracking-toy.md) supports a critic-tracking
mechanism without implying that distribution mismatch alone is impossible.

## New continuation result

The candidate is the unchanged original PR84 G-only stencil on alternating
curvature-bounded Adam. The scheduled prefix ends at update1000; the candidate
then runs1400 further updates at nominal G/D rates .00425 and prior .0085.
The noise horizon remains1200. Models, Adam moments, EMA and random streams
continue without reset. Checks require all8 modes and HQ >= .9, using live
weights. Updates1001–1200 are checked individually, then every10 through2400.

| Exact shared-state branch | First200 checks | Later120 checks | Lowest later HQ / modes | Final HQ / modes |
| --- | ---: | ---: | --- | --- |
| Scheduled control | 200/200 | 120/120 | .99829 / 8 | .99951 / 8 |
| Ordinary constant Adam | 6/200 | 6/120 | .10181 / 4 | .76782 / 8 |
| **PR84 original stencil** | **200/200** | **112/120 FAIL** | **.78857 / 7** | **.99805 / 8** |

The PR84 failed checks are1390,1540,1550,1560,1570,1910,2060,2150.
At1390 it retains8 modes but HQ falls to .87866. At1540 it has7 modes and
HQ .78857; seven-mode checks persist through1560. The final25 sampled checks
pass. A final-score or final-suffix gate alone would therefore hide the loss.
First *observed* failure means the first ten-step sample; intervening updates
after1200 were not scored. This is a finite conditional stability diagnostic,
not an equilibrium proof, a cold acquisition pass, or a claim about infinity.

The identity child matches a separately uninterrupted scheduled run's full
final state and measurements. The candidate inherits the exact archived
warm-state hash and reproduces every archived first1200 diagnostic and all200
clean-adapter update records. All4200 post-checkpoint gradient-evaluation calls
per player use the declared constant rates. There are1400 actual candidate
moment updates, plus1000 prefix updates; same-sample replay is verified2800
times. The candidate episode takes42.72seconds on one CPU thread; controls run
on independent cores. No CI, seed sweep, new rule or parameter tuning is involved.

[Raw episodes, source snapshots, logs and hashes](continuous-evidence/pr84-stationary-hold/manifest.json)
retain the result. The initial postprocessing check compared complete records
against original PR84's older schema, which includes inactive oracle diagnostic
fields removed by the audited extraction. Training and identity checks had
completed successfully. A corrected read-only validator compares every record
to the previously audited clean reference, and verifies that reference's warm
and final state hashes and diagnostics against original PR84. Both versions
and the initial error are archived. Training was not rerun or altered.

## What the other evidence says

The earlier ordinary constant .001 experiment is another delayed failure:
warm200/200, then119/120 longer checks, failing at1950 with HQ .7903 while
finishing near .999. See [extended controls](continuous-probe.md).
The current Chamfer follow-up passes warm200 but repeatedly gains and loses
cold ring quality; only3/24 checks pass. Its objective decreases do not protect
the complete game update. See [completed result](chamfer-projection-report.md).

PR82 has advanced to `4285309d8e8ac034fb8f7e378716d804e0e9b180`.
Its author reports v16 warm196/200 with a cold ring pass; v17 and v19 pass
warm200 but fail cold ring. The v20 hard slope gate loses a mode at1175 when
slope crosses .2 and undamped proposals resume. Its raw archive has199/200
warm checks, final8 modes/HQ1, constant applied rates, and no long hold.
These are author-machine results, not new local reproductions. Our exact
PR84 source replay differs: warm200/200 and cold7 modes. Neither evidence set
qualifies. [Exact-head PR82 report](https://github.com/255BITS/ParticleGAN/blob/4285309d8e8ac034fb8f7e378716d804e0e9b180/reports/toy100/alternating-curvature-report.md).

PR60's production head remains `983d037`; PR81 and PR84 heads remain
`e00de90` and `468ad26`. None has supplied a qualifying replacement. The existing
230 integrated tests plus11 separate helper checks establish implementation
checks, not sustained training stability; that distinction remains essential.

## Research priority

Same-dataset continuation remains the first research priority. The exact replay
now finds both a discrete destructive proposal at1325 and accumulated drift
before mode loss at1533. All16 held-out G batches support a coherent wrong
direction at1325. Another minibatch-average or scalar-damping grid is therefore
not the indicated next test. Opponent prediction changes some directions but
still fails the longer hold; the next mechanism must repair the surviving
feedback failure in short saved-state continuations before another full run.

After a new candidate's warm200 pass, extend that same branch through at least
2400, checking every update, before spending on additional acquisition hosts. A warm pass still cannot
qualify a frozen model: retain cold acquisition without decay from step1,
followed by uninterrupted continuation of its own acquired state. Further
continuation beyond2400 is required for survivors. At beta2=.999, the old
second-moment contribution is multiplied by .999^200 ~= .819 after200 updates;
the short warm test need not reflect settled optimizer dynamics.

Rest when there is no useful signal is allowed. Constant nominal rates do not
require nonzero parameter movement, and state-dependent correction is allowed.
A learner must remain able to respond to error on the same target. A separate
small model-state perturbation with a frozen control can test that capability
without changing the dataset; it has not been run for this candidate. Entire
distribution shifts remain outside the required scope.

The original PR84 adapter remains a reproducible failed reference. The
[current research candidate](continuous-selected-candidate.json) adds empirical
critic refinement and passes the conditional hold; it is not yet a qualified
solution. No production code or benchmark threshold changed.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_stationary_hold_probe.py \
  --output NEW_OUTPUT > /tmp/pr84-stationary-hold.log 2>&1
tail -f /tmp/pr84-stationary-hold.log
```

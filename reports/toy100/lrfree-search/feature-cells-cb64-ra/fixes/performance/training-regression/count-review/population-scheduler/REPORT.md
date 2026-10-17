# Population-continuity stationarity candidate

**CPU PASS: 13 focused contracts. GPU and strict quality pending.**

`pkg-POPULATION` is a full, separate RA4-derived package. Only `continuous.py`
and `training.py` change. Its count law, geometry, copy law, RNG API, noise,
losses, sampling, update order, and averaging/serving formulas remain RA4.

## Rule

The sequential table tester records the rows contributing at least two finite
pair observations at the negative scale used by its stationary decision.
Excluded rows do not count. A whole-population descent requires participation
from at least N-floor(.05*N) rows. Insufficient coverage is inconclusive and
uses the existing longer-scale search; the gradient test levels are unchanged.
Participation is not a collection of individual row stationarity verdicts.

Every rebase clears moved rows from that mask, whether a row was copied or born
from a novel anchor. Repeated replacement of the same row consumes continuity
once. Once more than5% are outside the mask, the accepted population verdict is
revoked. Only its immediately preceding descent is undone once. Its partial
window, block scale b, intrinsic tau, and untouched-row evidence are preserved.
Whole-group restart also drops old lineage pair evidence.

This is a scheduling response to a changed population, not a gradient DRIFT
test. A fresh covered negative decision can accept a new descent and renew the
mask. Positive/mixed/frozen decisions retain the original direction/scale law.

## Rate effect

| Before accepted descent | After descent | After continuity expiry |
|---:|---:|---:|
| 1 | .5 | 1 |
| .5 | .25 | .5 |

The prior LR uses its existing base*s formula on the next update. G network
scales are unchanged. D retains its existing floor at .75*prior_scale and payoff
damping, so a table release can also raise that existing critic floor. The EMA
weight retains s/(serve_average*b), and the unchanged serving predicate sees
last_decisive=0 after expiry. Output sigma retains its original state formula;
no noise or averaging policy is overridden.

A table-only release can increase prior motion. Passing strict learned toy
(.90 precision, all25 modes, .10 TV) and native grid stability is unestablished.
This candidate should be assessed prospectively with the matched fixed protocol.

## State and API integration

Trainer checkpoint schema becomes5. The sequential tester persists
population_schema=1, its exact policy/Q, one N-element boolean mask, the active
flag, the pre-descent undo scale, counters, and a compact last diagnostic. Old
trainer4 and old tester states are rejected. Mask shape/type, active/stationary
consistency, coverage, undo scale and policy are validated before mutation.
Same-law serialization/continuation and nonaliasing are checked.

The compositor copies the proposed `continuous.py` and `training.py` unchanged.
Exact AST splices are the `SequentialSettleTest` class,
`StationarityLR.__init__` mask initialization, and the two trainer checkpoint
methods `_state_dict`/`_load_state_dict`. Every other continuous/trainer AST is
identical to RA4. The trainer reaction hook already calls
`tester.rebase(group['params'], birth_death.moved_rows)` and resets row evidence.
**All ordinary copy, isolation copy, and novel-birth children must be included
in that moved_rows vector.** No new call signature is required.

The standalone package keeps RA4 backend4. Root's combined novel/paired law uses
its separately versioned backend and settings; trainer5 is the scheduler law.
The frozen lineage test suite hardcodes trainer4; rerunning it under trainer5
requires a separately declared private copy of that schema expectation, with
all original checker/receipt bytes preserved.

## Verification

`cpu-attempt3.json` contains13 contracts: complete AST scope, fresh coverage,
Q boundary/duplicate spending, preserved window, decisive2b coverage/exclusion,
existing broad hold,10 original direction/scale parity cases, one-descent undo
and renewal, serialization/continuation,8 atomic malformed/old-state rejections,
whole-group restart, served trainer5 roundtrip/old-state rejection, and the
actual trainer hook handling ordinary and novel row sets. Saved RA4toy1250
model/prior tensors construct the mechanical full-trainer case. No gradients,
optimizer steps, new seed, CUDA context or global RNG advance occurred.

Attempt1 was a test-helper identifier syntax error; its script/log are retained.
Attempt2 used the host's old Python interpreter, which cannot import the frozen
RA4 package's union type annotations. Attempt3 uses the canonical Python3.12
environment and passes. Production source was unchanged during all attempts.

The earlier saved dynamics receipt remains frozen in the adjacent
`post-ra4-quality` directory. Its deterministic W-decrease evidence proves at
least1016 of1024 rows replaced after750 while RA4's stationary stamp remains744.
The high dimensional row gate limitation99<384 remains a distinct issue.

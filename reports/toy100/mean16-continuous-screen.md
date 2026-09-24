# Averaging both fields repairs one failure, but not continued stability

The actual Adam update with sixteen native-sized banks per field repairs the
captured update1325: eight modes, HQ .997314, with exactly one advance of each
retained optimizer and unchanged learning rates. Continuing from the resulting
candidate states still fails20 of44 saved-window checks. This rejects the
finite-bank mean update before warm or long training.

| Consecutive window | Passing checks | Minimum HQ |
| --- | ---: | ---: |
|1324–1335|12/12|.994141|
|1380–1395|11/16|.845947|
|1530–1545|1/16|.821533|
|Total|24/44|.821533|

The original control also passes24/44, with a different distribution of failures.
The candidate fails1386–1390 and1531–1545; the last branch reaches seven modes.
At the first later failure,1386, HQ falls .911865→.863281 with clean-output RMS
motion .02364 and G bound factor .824. At1531, HQ falls .921875→.876953 with RMS
.02504 and G factor .425. Every D factor is one; maximum measured D ratio is
.916, so changing its bound from three to two would not affect these updates.

The [frozen evidence](continuous-evidence/round8-mean16-screen/) includes every
source epoch, declarations, receipts, failure states and logs. Original replay,
native RNG/noise clocks, rates, and final live optimizer counts pass in all
three windows. Each candidate update evaluates sixteen D fields at each of two
D points and sixteen G fields at each of two G points. The diagnostic adapter
also computes and discards three native fields per player:35 fields/player,
70 total, while retaining one optimizer advance/player. It applies the bounded
update from the same pre-step models and full Adam states, including newly
updated moments. The native critic-advantage receipt describes the discarded
field and is explicitly not attributed to the replacement.

Two earlier epochs are retained. The independent-step screen passes1324 and1325
but fails1326 from the original already-degraded pre-state. That is a failed
instant-repair assay, not evidence that the candidate destabilizes its own
trajectory. The consecutive screen resolves this distinction. Its first source
epoch encounters a scratch-constructor/host optimizer wrapper conflict before
candidate quality can be measured. The second isolates scratch constructors;
the first error is not counted as a training failure.

The [variance diagnostic](pr84-finite-bank-variance-geometry.md) finds that most
sample variation at1325 enters through D. Reducing this variation matters, but
does not by itself make the mean game field restoring. Its calibrated local
secants also fail monotonicity and uniform-contraction checks in a fixed saved
metric. Neither these finite banks nor these local checks establish a universal
impossibility result for the nonlinear capped game.

The next game diagnostic profiles the original sharp penalized D objective over
a frozen-feature convex readout, then differentiates that same value, including
the fake cap term. In parallel, the proper emitted-distribution route tests
smoothed forward KL after local MMD loses a mode while improving even its true
population objective. These are distinct declared experiments, not larger
batch or gain sweeps.

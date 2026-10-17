# Saved dynamics and population stationarity

CPU mechanical review: **PASS, 8 focused contracts**. Learned quality remains
**FAIL**. This review made zero optimizer updates, started no CUDA context,
consumed no random numbers, and left all frozen source/checkpoint hashes intact.

## Measured traces

| Step | RA4 emitted P / modes / TV | RA4 clean P / modes | E22 emitted P / modes / TV | RA4 cumulative copies |
|---:|---|---|---|---:|
| 1250 | .876465 / 22 / .188335 | .947266 / 23 | .617798 / 25 / .382202 | 7097 |
| 1500 | .614014 / 19 / .416982 | .663086 / 19 | .654785 / 25 / .345215 | 8305 |
| 1750 | .616821 / 19 / .407544 | .673828 / 18 | .688477 / 25 / .311523 | 9713 |
| 2000 | .758057 / 21 / .263540 | .820312 / 21 | .715454 / 25 / .284546 | 10701 |

All ten checkpoints have identical held-real FIFO tensors, cursors, and data
positions between RA4 and E22. The output sigma stays .029. At 2000 both clean
precision values are .820312, but E22 retains all25 modes and has only1873 total
copies. Precision loss1250→1500 is already present in the clean particles, so
output noise alone cannot explain it. These are saved evaluation records.

## Reproduced accounting defect

RA4 accepts table STATIONARY at744, with s=.5 and b=16. Every later copy rebases
the corresponding gradient evidence, but leaves the whole-table stationary
stamp active. At1512 an inconclusive decision increases b to32; averaging then
slows from1/128 to1/256, while the stationary stamp still dates to744. The saved
1500 window's oldest three pair vectors have only21,25,27 finite rows out of1024.

There is a deterministic lower bound on replaced rows: row-history W only grows
between resets, so W decreases prove at least one birth/copy. Between750 and1000,
398 distinct rows are proved replaced. The union reaches807 at1250,941 at1500,
and1016 at2000. The old whole-population continuity claim therefore cannot stand,
even if the original decision had included every row. Rebase on a saved CPU
tester clone reproduces last_decisive=-1, s=.5 and unchanged b/tau after those
398 rows are rebased.

This is an evidence/scheduling defect. It does not establish that removing it
alone will solve the quality failure. Fast training parameters are restored at
the start of every update; a serving-only change cannot repair training drift.

## Row gate limit

RowEvidence uses window50, so its effective sample size is bounded by99. The
128 dimensional learned prior requires384. Every saved RA4 and E22 learned
checkpoint has zero mature rows and zero row flags. valid=True after an update
means the calculation ran; it does not mean any row was testable. Holds, excluded
row votes, and hot-row full-rate corrections are consequently inactive.

Increasing the window alone cannot supply the missing observations:128 uniform
draws among1024 rows touch a row about235 times over2000 updates, before copying
resets. A gradient projection would be a separate statistical mechanism.

The native prior is2 dimensional and this particular dimensionality failure
does not apply. The retained native snapshot is progress evidence through5750,
not a final native verdict. Its table accepted STATIONARY at3832; the earlier
open table rate and motion should not be described as a final failure.

## One bounded prototype

[population_certificate.py](population_certificate.py) wraps the existing
sequential table tester. It proposes one population-continuity scheduling law:

1. At a negative decision, record rows with at least two finite pair observations
   at the same scale used by the original decision, excluding nonvoters. Accept
   a whole-population descent only when uncovered rows are at most floor(Q*N).
   Insufficient coverage follows the existing inconclusive longer-scale search.
2. Rebase clears copied rows from that participation mask. Replacing a row twice
   spends continuity once. When surviving coverage falls below1-Q, invalidate
   the old whole-population stamp and undo its most recent table descent once.
3. Preserve the partial window, b, tau, and untouched-row evidence. Persist one
   N-element boolean mask and small scalars; reject the old checkpoint law.
   Noise, output sampling, EMA formulas, and statistical test levels are unchanged.

The rate undo is a scheduling response to a changed population. It is **not** a
positive-gradient DRIFT verdict and carries no new statistical claim. On the
best-case saved bootstrap it changes s=.5→1 once; that bootstrap exists only
for the mechanical proof and is expressly rejected by production-style loading.

Exact Q-boundary/duplicate spending, window preservation, coverage rejection,
fresh renewal, same-law checkpoint continuation, and atomic old/bad-state
rejection pass. Source, input hashes and CPU RNG remain unchanged.

## Recommendation and limits

Use the mask/invalidation accounting in the next candidate if whole-population
stationarity is used by its scheduler. A table-only rate undo can increase prior
motion, so this prototype is an explicitly unqualified ablation, not a candidate
claimed likely to pass .90 precision/all25 modes/.10 TV. Couple its decision
with the separate birth-supply and paired EMA geometry findings before choosing
the single matched quality run. No benchmark labels enter this prototype.

Participation coverage does not certify every row individually stationary. The
underlying sequential and count procedures retain their existing limits; count
tests have no repeated adaptive guarantee. No historical CUDA action replay,
new training trajectory, rescoring, seed, horizon, or gate change occurred.

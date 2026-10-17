# RA6 final saved diagnosis

The authoritative root CUDA toy result fails the strict gates: emitted P=.516968, coverage23/25 and TV=.483032. CPU read-only diagnosis reproduces saved FAST clean counts exactly. All frozen source/checkpoint hashes, checkpoint tensors and global RNG remain unchanged; no CUDA, new seeds, proposals, emissions or updates. This report does not choose a serving override.

| Final2000 | Clean fast P / modes / TV | Clean EMA P / modes / TV | Saved emitted P / modes / TV |
| --- | --- | --- | --- |
| RA6 | .585938 /24 /.414063 | .915039 /23 /.134258 | .516968 /23 /.483032 |
| RA4 | .487305 /18 /.514688 | .820313 /21 /.236836 | .758057 /21 /.263540 |
| E22 | .818359 /25 /.181680 | .820313 /25 /.179688 | .715454 /25 /.284546 |

## Current support and births

FAST misses clean mode3 by one row:10 supported rows versus11 required. Its annotated reference cells have real target40, vacancy28, four inside eligible parents and initial copy capacity4. All25 reference modes have inside eligible parents. The current CPU refit has389 flags,580 p>Q rows,406 inside eligible rows and total initial physical unique-copy capacity upper bound215. Among600 raw-supported FAST rows,58 fail p>Q and142 more fail inside, leaving400; six further learned eligible inside rows are raw-unsupported. EMA clean misses modes6/11. Learned support and oracle support describe different quantities.

Births did not stop after1072. Saved checkpoints show evaluations125/156/187/218/250 and cumulative births459/553/671/786/893 at1000/1250/1500/1750/2000. Final acceptance is893/940 attempted target cells. The latest reaction occurs at2000, with2 mass copies+14 local copies+4 new-latent births+31 global copies=51 ordinary actions, no isolation. All four latest newborns are raw-supported and current CPU p>Q/inside for both models at checkpoint age0. No absent-parent-only explanation or production acceptance/accounting violation is reproduced.

The age0 observed-reference tail at1000 is a separate finding: one birth closely follows an even real point outside the oracle's .09 radius. It does not explain the broad1000-to2000 motion or justify introducing oracle labels into production.

## Generator and table coadaptation

| Generator / latent table | Fast P / modes | EMA P / modes |
| --- | --- | --- |
| G1000 / z1000 | .781250 /25 | .989258 /25 |
| G2000 / z1000 | .291016 /12 | .378906 /14 |
| G1000 / z2000 | .286133 /11 | .501953 /16 |
| G2000 / z2000 | .585938 /24 | .915039 /23 |

With saved z1000 fixed, later FAST G retains raw support for234/800 previously supported coordinates;230 retain the mode. EMA retains388/1013, all388 in the same mode. Table changes also strongly affect outputs under fixed G1000. Actual G2000/z2000 is much better than either crossed pair, showing substantial coadaptation; these nonlinear comparisons do not provide an additive causal decomposition. Offline old newborn-coordinate fitness under a fixed old CPU head is recorded separately. One hundred25 intervening reactions prevent same-incarnation survival claims.

## Controller and timing regimes

FAST remains served because the retained prior decisive result is drift. Its latest scheduling decision remains904 inconclusive, s=1 and current b=64. The525-participant coverage rejection belonged to tested b=32 at904; b64 followed that verdict. Population remains inactive with two coverage rejections and no expiries. The independent participant-clock owner audits intrinsic-time progress; this report does not claim that the clock stopped.

The data controller is already closed by saved500, with updates continuing through2000; this is not a newly observed closure after1072. G's stationarity scale goes .03125 at1000 to .015625 at final; D's tester scale goes .0009766 to .0002441. These tester scales alone are not effective optimizer LRs: training also applies its prior/D floor and payoff damping.

Measured latest reaction times drop6.841/3.258/.288/.233/.292 seconds at1000/1250/1500/1750/2000 while metric rank remains8, accepted births still use mostly one linearization and work remains about2.01M distance cells,23.19M projection products and4290 count terms. The final1000 updates consume262.813 training seconds. Reaction cessation, dimension skip and a wholesale loss of solver work are ruled out by saved counters. Sampled host times do not identify the cause of the speed change; external load, clock, synchronization and kernels are not reconstructed.

The strongest next diagnostic is therefore general generator/table step coupling or learned-feature motion trust under the actual optimizer APIs, with production decisions independent of oracle labels. That prospective work belongs in a fresh directory after this proof is frozen. Noise, horizons, data, seed, serving rule and all current quality gates remain unchanged.

Evidence: `receipt.json`, `motion-comparison.json`, `regime-summary.json` and archived metric prefixes. CPU learned partitions are current saved refits, not historical GPU replay. Parent capacities are initial-state upper bounds, not reserved postplanning ledgers. Exact birth lifetimes and unlogged per-reaction actions are unavailable.

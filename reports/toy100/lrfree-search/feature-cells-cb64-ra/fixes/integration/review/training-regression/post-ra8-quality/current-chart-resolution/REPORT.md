# Fixed current-chart resolution:64 versus128

**The finer chart resolves grid100 aliasing but overfits the smaller toy
reference.** An unconditional128-cell replacement is unsupported by these
fixed states. No production change or quality run was made.

| Current CPU chart | Toy64 | Toy128 | Grid64 | Grid128 |
| --- | ---: | ---: | ---: | ---: |
| Even fitted rows |512|512|10000|10000|
| Actual groups |25|26|30|100|
| Observed-real cell purity |1.000|1.000|.6623|1.000|
| Observed-real group purity |1.000|1.000|.3199|1.000|
| EMA p>.05 rows |1023|1023|19742|19790|
| EMA inside eligible rows |991|796|19682|19783|
| Paired same-group rows |1018|1006|20000|19999|
| Joint serving rows |985|781|19682|19783|
| Required joint rows |973|973|19000|19000|
| Current geometry eligibility |yes|no|yes|yes|

These are new current FIFO/current-D CPU refits. They do not reconstruct the
unstored historical GPU basis/centers. The saved GPU stamps and grid35-group
diagnostics remain unchanged. Both resolutions use exactly matched even-only
mean/scale/basis and the same starting/post-fit private RNG state. The chart
source and all49 inputs were frozen at08:29:48.057UTC before computation;
the sole attempt completed at08:30:16.534UTC and exited successfully.

## Geometry and calibration

At64 cells, grid reference cells merge up to3 known modes and topology groups
merge up to22. At128, every fitted cell/group is pure under downstream
observed-real annotations, with100 distinct dominant groups. Raw critic head
separation was already strong in the independent saved diagnostic; this
points to finite chart resolution as a concrete source of coarse aliasing.
Labels were applied after fitting and every chart decision, with no benchmark
labels passed to fitting, support, topology, tests, targets or eligibility.

Toy cells and groups are already pure at64. At128, one true support splits
across two groups, while the mean even/odd cell occupancy falls from8 to4.
Odd empty cells rise3->14 and empty refined categories26->76. Real odd rows
outside the fixed even-fit region rise.16797->.39258; EMA outside rises
.03223->.22266. Its strict p>.05 supply stays1023, so the serving loss is
mainly the fitted inside partition, with12 additional paired group
disagreements. The unchanged95% even-fit ordinal is sensitive to sparse
per-cell fit geometry. This is not a reason to loosen Q or the gate.

Grid has10000 odd calibration rows: mean cell occupancy156.25->78.125 and
real odd outside fraction stays about.0521->.0522. EMA outside improves
.01590->.01085 and its paired joint fraction remains above.95.

## Parent and count accessibility

Every observed mode has at least one eligible p>.05 and inside parent in
FAST and EMA under both resolutions, for both tasks. Grid128 creates9 fine
target cells with no eligible parent despite mode-level support; their EMA
vacancies total265. Toy EMA target cells without a p>.05 parent increase
8->36, with corresponding inaccessible cell vacancies52->123 (inside
vacancies52->156). These fine-cell holes are not genuinely absent modes.

The actual common family increases194->386 hypotheses; Q/m changes
.000257732->.000129534. No separate count budget or threshold was used.
The smaller toy bins yield no local inside discoveries at either resolution.
On grid FAST, original-cell formula discoveries change from0 excess/0
deficits to2/18, and refined discoveries from1/17 to3/27. Grid EMA changes
from0/0 to1/18 coarse and1/22 to2/28 refined. Some fine spatial discrepancies
are hidden by the coarser chart despite its smaller group TV.

Before shared reservations, quota overlap and group caps, the grid128 gross
accessible birth upper bounds are119 coarse/118 local inside for FAST and
103/103 for EMA, versus0/0 at64. These bounds cannot be added or called an
action plan. The existing1000-row joint ordinary budget and ledger remain
unchanged. Counts here use clean correlated finite rows, so conditional-law
values are descriptive, not valid emitted-iid inference, equivalence or
prospective training/quality evidence. All global comparisons see clean
inside excess; their non-rejection is not used for serving or claimed here.

## Work and scope

Doubling K doubles distance-cell work: toy1,183,744->2,375,680 and
grid23,044,096->46,096,384. Projection products are identical. Count terms
are toy7794->8798 and grid123908->123804: finer pooled margins can shorten
enumeration despite more hypotheses. Retained chart bytes increase about
13.1% on toy and.86% on grid. Peak distance blocks stay256 rows by at most
128 cells. Measured chart/query CPU times were.033/.039 seconds for toy and
.528/.605 seconds for grid, on one thread; these are not GPU throughput
measurements. Existing K-squared topology is bounded by128 cells; no
N-squared pass was introduced.

All source/input hashes, loaded trainer tensors and global/trainer RNG states
remain exact. Only private cloned fit RNGs draw the existing projection;
there are no new seeds, trainer/model construction, stochastic emissions,
optimizer/training updates, CUDA contexts, holdout fits or production writes.
This supports testing a reference-size/rank regularization policy separately.
It does not qualify128 generally or explain/certify the remaining covariance
and holdout gates. Evidence is `SOURCE-FROZEN.json`, `attempt1/result.json`,
retained log and final receipt/seal.

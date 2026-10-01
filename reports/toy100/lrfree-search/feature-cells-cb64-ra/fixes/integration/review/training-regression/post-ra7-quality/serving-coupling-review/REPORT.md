# Current paired average: learned geometry and serving coupling

CPU, read only, saved RA7 checkpoints. There are no new emissions, seeds,
optimizer calls, training updates, oracle queries or production edits. Source,
input, tensor and global RNG identities are unchanged. One current D/FIFO
chart is shared by each checkpoint's two clean model/prior pairs; this is not
historical GPU geometry replay.

| Update | Same real group /1024 | EMA eligible and same group /1024 | FAST eligible inside | EMA eligible inside | FAST/EMA group TV vs odd real | FAST/EMA outside fraction |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 500 | 944 | 504 | 243 | 509 | .12012 / .12012 | .76270 / .50293 |
| 1000 | 955 | 649 | 376 | 653 | .08594 / .07910 | .63281 / .36230 |
| 2000 | 1018 | 985 | 545 | 991 | .08105 / .07910 | .46777 / .03223 |

All three real-only centre graphs have 25 groups. A fixed joint empirical
criterion requiring at least 973 rows with EMA p>Q, EMA inside and equal
FAST/EMA group naturally vetoes 500/1000 and passes final. It certifies only
the stated finite-population geometry check. It does not certify temporal
stationarity, emitted quality, group correctness or all latent perturbations.

Final support contingency is 535 both eligible, 456 EMA only, 10 FAST only
and 23 neither. Fine-cell agreement is only 594/1024 and refined-category
agreement 316/1024: averaging changes within-group geometry substantially.
On the fixed 128-row subset, actual EMA has 123 eligible rows, live G with
EMA prior 104, EMA G with live prior 78, and actual live pair 71. Serving
the matched model/prior pair is material; the data do not justify swapping
only G or only the table.

## Count mismatch remains

EMA is concentrated relative to real reference width. Final odd-real outside
fraction is .16797. The existing formula reports EMA inside excess with
p=1.58e-19, whereas FAST has the larger opposite outside excess. EMA's
refined-category TV improves .37988 to .27930, but coarse-cell TV worsens
.20898 to .21680 and one coarse deficit is discovered. These are descriptive
formula results on dependent noiseless table rows. The K+2K+2 family remains
unchanged; two model comparisons are not jointly calibrated by separate Q
budgets.

A rule requiring no EMA mismatch discovery would veto this final state.
The proposed small control therefore concerns current anti-blur geometry,
with the population stationarity law still controlling its own LR descent.
It must not be presented as positive distribution equivalence. Even under a
fixed-chart iid odd-reference idealization, a simple valid all-subsets
Hoeffding union bound gives group TV uncertainty .14088 at Q=.05 for 512
heldout rows and 25 groups, before accounting for adaptive learned-head
dependence. Final EMA empirical TV plus that radius is .21998. This bound
does not establish a tight equivalence claim. Actual serving adds existing
latent/output noise, which is not evaluated here.

## Existing coupling

Serving currently requires only the table's last accepted negative verdict.
G has a separate parameter-displacement cosine test; sigma has another.
Population rejection is valid for row-history descent but can block averaging
despite current matched support-group agreement. This is a difference in the
object being checked, not a reproduced participant-clock implementation bug.

RA7 G scales .25/.125/.0625 at 500/1000/2000. Its final applied LR equals
RA6 because adaptive compensation offsets the smaller base. All G last
decisive verdicts are negative, yet old fixed-coordinate probes show large
longer-interval functional motion. Parameter directional stationarity does
not imply pointwise output stability.

Both EMA G and EMA prior use the same table-derived weight. Its reciprocal
is 64/128/256 updates at these checkpoints; the corresponding G intrinsic
window is 16 in all three, while the sigma intrinsic window is 64/128/256.
EMA does not average learned sigma. The paired-average advantage is an
observed coadaptation result; the report does not change the averaging clock.

## Small prospective control and state constraints

Root selected one empirical anti-blur gate: finite current pairs, valid
nonduplicated chart, and at least N-floor(QN) rows satisfying EMA p>Q,
EMA inside and equal current real-only group. Keep the G/prior/D updates,
population descent law, actions, count family, budgets and serving noise.
Use the legacy gate for the existing small-reference path.

Evaluate after all reaction actions with current-D FAST cache and freshly
computed EMA features. A reused chart must not mix D epochs. A bounded lease
through the next known real-FIFO turnover is an explicit empirical expiry,
not proof that every intermediate G/D update preserves the predicate. Store
typed eligibility/counts/chart serial/decision step/expiry as semantic state;
loading an ephemeral chart cache must not silently discard the decision.
Checkpoint loads need complete validation and old-law atomic rejection.

The extra EMA forward is bounded by N rows in existing chunks; assigning
features costs O(NK) with rank fixed, plus the existing bounded K graph.
There is no all-row Jacobian or N-by-N graph. A strict every-update current
geometry claim would require more computation or a separate motion bound,
which this proposal does not establish. Future quality and canonical gates
remain root-owned prospective tests.

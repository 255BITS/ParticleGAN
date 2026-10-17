# Fixed even-reference support count proposal

This is a separate private proposal based on frozen count recovery v4. RA2, its validation queue, and the v4 proposal remain frozen. All local execution uses CPU with CUDA hidden, one CPU thread, deterministic Torch operations, and the existing saved inputs. No new seed, optimizer update, score sweep, Q change or acceptance gate is permitted.

## Partition and count law

Keep the existing even-fit mean, scale, PCA basis, centers, spherical cell scales and support score. Before reading odd reference rows, compute the existing score of each even training row. The fixed global boundary is the ceil((1-Q) times M) order statistic, Q=.05 and M the even-reference count. Scores equal to the boundary are inside. The fitted category ID is `2*nearest_cell + (score > boundary)`; both regions of all K cells are retained, including empty regions. This boundary is an in-sample fitted geometric definition and is not a conformal threshold.

Test odd reference category counts against emitted fake category counts with the existing two-sided exact conditional Hypergeometric law. This replaces the original K-cell count family. All 2K tests use Q/(2K) Bonferroni, including empty bins, with no extra original-cell significance decisions. Under the fixed-partition iid null, even fitting is conditioned upon, odd real and emitted fake category counts are independent categorical draws, and each conditional test is super-uniform given its pooled category total. A union bound controls at least one discovery at Q. Repeated adaptive training snapshots have no cumulative guarantee; learned-head dependence in a real trainer is not an independent iid qualification.

Use the original odd-calibrated support p-values solely for the unchanged BH flags and parent eligibility p>Q. Degenerate metrics disable count discoveries. Tied and constant scores follow the deterministic inside rule. No fake rows, labels, model quality score, odd null quantile or odd-reference support p-value may fit the boundary.

## Actions and shared ledger

An ordinary donor must be flagged by the unchanged support law, outside the fitted boundary, and in an outside category with certified excess. An ordinary parent must be inside, unflagged, p>Q under the original support law, and in an inside category with certified deficit. Each parent is used once in the combined ordinary/isolation action; children are distinct and cannot also be parents. Ordinary moves remain at most floor(Q times N).

Full even+odd reference cell targets reserve every currently unflagged row. The coarse real-center topology is unchanged. Ordinary births may fill only actual cell vacancies and actual group vacancies, and supported rows supply no ordinary deaths. Ordinary flagged deaths do not subtract from the supported ledger; isolation subtracts only unflagged ordinary children before adding planned ordinary parents. The original isolation guard, own-cell parent policy, group vacancy caps and remaining-hole assignment apply after ordinary planning. Legacy isolation may redistribute births among cells within a group. With a small flagged set, combined moves cannot exceed the original flagged count; with broad flags, isolation rejects and ordinary retains its fixed budget.

This replacement law does not transport a purely supported categorical imbalance with no flagged outside donors. It is one bounded support recovery proposal, and that limitation must be reported rather than hidden by a joint uncorrected original count family.

## Predeclared CPU checks

1. Compare new versus v4 fitted geometry, original support flags/p-values and fitting RNG using existing fixed inputs. Mutating odd rows must not change geometry or count boundary; changing fake rows must not change the frozen boundary.
2. Enumerate a small exact pooled categorical null over every possible real allocation. Check p-value super-uniformity and family rejection probability at Q/(2K), including zero bins. Check deterministic ties, all-identical features, empty fitted cells, minimum inputs and invalid count inputs.
3. Preserve query cache/refresh behavior, package/checkpoint policy distinction, finite metrics and bounded 2K count storage/work.
4. Use saved toy1000/2000 snapshots exactly once to compare 2K discoveries, eligible supply and realized ordinary actions against frozen v4. Keep their existing CPU projections and planning RNG bytes unchanged. Report response and limitations without a quality verdict.
5. Exercise one small-flag ordinary+isolation plan with existing reference/input values and inspect per-cell/group supported counts, donor/parent category certificates, combined budget, uniqueness, no eligible parent and rare-group preservation.

The independent review is CPU-only. Root owns any subsequent fixed-input CUDA execution and canonical matched toy/MNIST/replay tests after source/evidence freeze.

# Independent count and lineage review

The one even-fit support count proposal and the bounded lineage proposal pass
their reviewed CPU contracts. Two concrete integration defects were found and
fixed by their owners before this receipt: flagged ordinary children were
subtracted twice from the isolation supported ledger, and the first graph copy
used an incompatible Torch `argsort` overload. The first-copy failure log is
retained. No remaining blocker was found in these source contracts.

The count proposal freezes the existing even-fit score's global empirical
95th-percentile order statistic before reading odd references. Ties are inside.
Category `2*nearest_cell + outside` is consequently fixed by even references;
odd scores and fake scores cannot fit the boundary. A saved-feature mutation
check verifies this, while v4 geometry, original support flags/p-values and
fitting RNG remain bit identical. Odd references remain the original conformal
support calibration and the heldout categorical count sample.

The proposal replaces the original K-cell significance family with exactly
2K exact conditional Hypergeometric tests, including empty regions. It applies
Q/(2K) to all tests. Conditioning on a fixed partition and independent equal-law
categorical draws gives each pooled-total conditional test a valid marginal
p-value; the union bound requires no independence between tests. The independent
29-allocation rational reference with pooled counts `[6,6,4,0]` gives family
rejection probability `28/2145 = .0130536131` at Q/4, with maximum p-value error
2.34e-15. This finite check substantiates the implementation, while the proof
supplies the conditional law. No cumulative adaptive-training guarantee follows.

Ordinary deaths require all three conditions: unchanged BH flag, outside
membership and the relevant outside excess certificate. Ordinary parents
require inside membership, unchanged p>Q, no flag and the inside deficit
certificate. The ledger records row categories, category allocations and both
certificates. Full even+odd cell targets reserve all unflagged rows; only
unflagged ordinary children are subtracted before ordinary births are added.
The actual source uses no supported ordinary deaths. Ordinary allocation is
bounded by cell and group vacancies and floor(Q*N); unchanged isolation uses
the remaining shared group ledger and distinct unused parents. The owners'
final seven-contract receipt shows 7 ordinary plus 38 isolation moves under
51 flags, with correct kept counts, group caps and the protected rare group
intact. With broad flags, isolation rejects and the ordinary budget remains.
The rare fixture proves its planned ledger protection; it does not resolve the
original support score's rare false-positive failures.

The count owner's saved CPU reconstructions report v4/new ordinary moves of
4/25 at1000 and 4/32 at2000, zero supported deaths and 25/32 distinct eligible
parents. Those are conditional response diagnostics on the saved projection,
not learned quality results. The global even-fit boundary accepts .8711/.8223
of odd real rows in those states; its empirical .95 fit is not a conformal
coverage promise. The replacement cannot transport a purely supported mass
imbalance without flagged outside donors. Adding the old K tests alongside it
would require multiplicity correction for the actual joint family.

The lineage graph is symmetric, bounded by min(rank,64,N-1), indexed by current
row incarnations and shared semantically by live and EMA priors. Overwriting a
row removes old reciprocal links; newer parent copies evict older links
symmetrically. Empty batches are inert. Independent fixtures verify these
contracts and malformed-state rejection. The empty graph gives bit identical
historical displacement and the latent derivative remains the indexed identity.
A specified missing-sort-neighbor case gives radius5 to .005 after the known
copy link is included, within the candidates+degree bound.

The graph is serialized; coordinate caches are discarded. Same-law graph
restore is exact, and old kernel/invalid graph checkpoints reject before graph
mutation. The feature backend schema changes3 to4; GANTrainer schema already
was4. Settings distinguish the kernel and count laws. The lineage owner's
11-contract receipt separately verifies live/EMA copies, RNG equivalence,
bounded work independent of population size and semantic replay.

Independent receipts are `count-review.json` and `lineage-review.json`.
Reviewed production hashes are count `a2be31d6…de1dd9`, lineage feature
`23c70cd5…ba80622` and lineage trainer `7edd9cb5…13b1c58`; full hashes are in
the receipts. This review performs no optimizer update, additional seed or
CUDA context. The frozen saved-state diagnosis and all original acceptance
outcomes retain their prior meaning. Root's complete-source initialization,
GPU execution and learned toy/MNIST/replay acceptance remain separate checks.

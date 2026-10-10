# Independent group-count proposal review

PASS for source scope, integer reduction algebra and retained CPU receipts. No numerical suite was repeated.

The only existing-method change is FeatureCellSnapshot._group_counts; replacing it restores the entire RA6 feature module byte for byte. All other28 modules match RA6. For production one-dimensional int32/int64 counts on the snapshot device, the membership matrix selects exactly the same cells as each original Boolean slice. Both sums promote int32 to int64. Int64 overflow uses associative modular addition, so reduction order preserves the integer result. All other dtypes and shapes retain the literal original fallback.

Real topology guarantees1<=G<=K; temporary work is bounded by K*K (4096 entries for K64). The method adds no cache, checkpoint state, random calls or count-law changes. Floating sums are unchanged.

The retained224 scalar/fallback/error cases and two complete saved reactions pass. The saved reactions retain47 copies plus4 births, all plans/actions/certificates/ledgers/work/RNG and live/averaged coordinates, moments, history, evidence and graph. Fixed planner profiles retain identical outputs and SVD work while nonzero calls fall355→55 and347→47. This is CPU operation-count evidence; CUDA timing and strict trajectory quality remain unqualified.

This independent receipt pins current production/proof/input hashes. A later owner READY receipt and root-owned CUDA qualification remain separate. RA6's running frozen package is untouched.

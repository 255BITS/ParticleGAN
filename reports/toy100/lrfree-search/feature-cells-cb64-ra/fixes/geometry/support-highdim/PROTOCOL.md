# High-dimensional support diagnosis

CPU only. This new directory is the only editable source area. Existing
geometry READY sources, support lane sources, shared RA2 and original studies
are read only. The original highdim fixture uses seed90229 at
N1024/2048/4096/8192; no new seeds, critic training or gate changes.

The fixture's captured critic features are128D. Planted unsupported rows have
less total orthogonal residual norm than real rows, because fake nuisance
coordinates are narrower. The original projected/radial score and full diagonal
score have zero recall at the smaller sizes. Thus residual norm alone does not
isolate the missing semantic direction.

## Predeclared causal tests

1. Compare original rank8 centroid support with bounded actual-real anchors:
   original64 representatives and global farthest-first up to64*4=256 even
   real rows. Use full standardized captured features, not raw samples.
2. To distinguish missing information from nuisance dilution, fit real-only
   between-cell versus within-cell variation. Exact full-feature covariance
   is a diagnostic reference, never a proposed wide-head implementation.
3. A bounded dictionary spans at most64 existing cell-mean directions. Fit
   within-cell covariance only in this dictionary; whiten and retain rank8
   between-cell directions. Setup is linear in real-row count and retains
   no full feature covariance or row-by-row distance matrix. Query degree and
   projected width stay bounded. This test is not an oracle classifier.

All score parameters/anchors use even reference rows only. Odd rows supply
held-out score calibration. Use the original upper-tail finite-sample p-value,
BH Q=.05, duplicate/reference validity guards and support gates. Oracle labels
are read only after fitting/scoring. Preserve source/input hashes and report
all four required sizes, numerical conditioning and bounded work. A failed
variant is reported as a failure, not promoted by adjusting thresholds.

The original baseline audit and the already executed CUDA geometry contract
are reviewed separately. Their validity does not establish corrected-model
training quality or full support-family qualification.

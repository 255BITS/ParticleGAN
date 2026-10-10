# Fixed current-chart resolution probe:64 versus128

Use only the sealed RA8 final toy2000 and canonical grid100 final7000 states.
For each, compare exactly the existing64 cells with one predefined128-cell
alternative. Keep metric rank8, chunk256, Q=.05, even-fit95% order boundary,
odd support calibration, MST topology and common3K+2 count family unchanged.
No other resolution, rank, level, gate, seed or checkpoint is considered.

Build current real/current-D features from the saved raw training FIFO and
current clean FAST/paired-EMA anchors. Reuse the frozen functional toy forward
and native discriminator head equations by AST extraction. No model/trainer
constructor, emitted-sample draw, evaluation metric, training or optimizer
step is used. In particular no holdout cloud is used for fit or queries.

Both charts start a private CPU fit generator from the same saved CPU RNG
bytes, with no new seed and no global/trainer RNG restoration or consumption.
Only the existing fit's randomized projection uses these private clones.
Verify their mean/scale/basis and post-fit RNG state agree exactly: increasing
K changes centers/partition, without changing the fitted projection. These
are deterministic current CPU refits, not historical GPU charts.

For each chart report real calibration occupancy, boundary/inside fractions,
clean support/flags/eligible parent supply, target cell/group accessibility,
coarse/refined/global count summaries using the actual common correction,
and the unchanged paired-average geometry result. Clean finite-table count
tests are descriptive here; clean correlated rows do not satisfy emitted-iid
sampling assumptions. No nondiscovery is called equivalence and no action
plan or future-quality result is inferred.

Only after each fit, annotate its cell/group memberships with the fixed
benchmark nearest-mode labels on the observed real references and existing
clean anchors. Report weighted cell/group purity, merged modes and eligible
parent mode counts. These labels never enter projection, clustering, support,
topology, tests, targets, eligibility or production decisions. This diagnoses
partition aliasing; it is not an oracle production repair.

The fixed measurements include2 charts per state and FAST/EMA queries each.
Their common multiplicities are194 and386, so finer local counts face both
smaller calibration bins and a stricter actual Bonferroni cutoff. Report
distance/projection/count-enumeration work, peak block dimensions and retained
chart bytes. No N-squared reference or table distance pass is added; existing
K-squared real-topology work is bounded by128 cells.

Freeze helper/protocol/source/input hashes before computation. All package,
source and checkpoint hashes, loaded trainer tensors and global CPU RNG must
remain exact afterward. CUDA must remain uninitialized. Retain any failed
helper attempt separately. No package/config/validation/frozen source edits.

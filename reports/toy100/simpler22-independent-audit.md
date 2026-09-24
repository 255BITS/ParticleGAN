# Independent copied-bundle audit: PASS 22/22

The new `reports/toy100/simpler22` bundle passes the complete common gate after
copying all 190 files to a fresh temporary directory and denying access to the
original evidence. The copied files matched their originals byte for byte.
No main-worktree file was edited and no model was retrained.

The stricter checker was frozen in a detached worktree at
`fec297421648eb3cbd2d571e551163f1a0421555`, including the seeded CPU initialization
verification introduced by `f3aaf6b`. This repeats and supersedes the earlier
audit at `ddfb1b3`. The exact `benchmarks/toy_suite.py` SHA256 is
`c34e72e2f8f52cf3251c1efa50f8b931ebdff1a35fb8316cfcec2744ade21e89`.
During the audit, a Python audit hook
denied access to `/dev/shm`, any `artifacts` directory, the main worktree, the
original research worktree and the uncopied bundle in the checker worktree.
Subprocess launches were also denied. Four deliberate forbidden reads confirmed
the guard; regrading and rendering made **zero forbidden dependency attempts**.

The exact configuration changes from the old `shared_candidate.json` are:

| Field | Old | New |
| --- | --- | --- |
| name | kappa_fine_k1176 | roundcap_k1p0_c1p0_p0p0_b0p999 |
| reg_kappa | 1.176 | 1 |
| reg_coeff | 6 | 1 |
| prior_reg | .05 | 0 |

No other config fields changed. The new config SHA256 is
`4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`.

All 19 transfer cases pass their frozen live gates. All three native problems
pass coverage, every required terminal fidelity check and the independent
100,000-draw holdout. All global recipe fields, noise settings and model policy
agree across the native and transfer records. The common gate confirms noise
application on all 19 transfer hosts and complete V2 public-package source scope.
The 23 shared executable-source files match across native and transfer runs,
including all 12 public-package files. Every native source archive has exactly
its 23 declared members and matching hashes. The transfer archive has exactly
118 declared members; its additional `benchmarks/toy100/models.py` bytes are
stored as the separately hash-bound `noise_source.py`, as required by the format.

The stricter checker independently recreates the initial native tensors on the
declared CPU stream at seed 1234. All three runs match the recorded 20,000 × 2
prior tensor hash, extrema and identity affine generator weights/bias. The
matching prior SHA256 is
`ece1639f2519f30c1a2cc0b39047e2e1347d2de2e4ea54d06b5e22cbef69c750`.
Three additional in-memory negative controls changed only the recorded prior
hash. Each was rejected specifically with `seeded affine initialization differs`.
No saved raw file was altered for these controls and no training was performed.

All four GIFs were regenerated **byte for byte** from the included live snapshot
arrays and live-event metrics: grid100, rotated100, staggered100 and the combined
animation. Each has 14 frames, the declared fixed axes, 500 ms ordinary frames
and a 2,500 ms final frame. Every sidecar GIF, snapshot and event-file hash
matches. The historical absolute snapshot paths are provenance labels; the
auditor resolved the copied problem/filename and verified its declared hash.

Only the copied aggregate `compatibility.json` changed during regrading because
it records absolute artifact paths. All raw evidence stayed unchanged. The
initial retention inventory also differs only for this regenerated aggregate.

[Machine-readable audit](simpler22-independent-audit.json) includes each case,
GIF/source hashes, the config diff, guarded reads and raw-source counts. The
guard script, copied bundle, regenerated GIFs, full log and the exact loaded
checker-module sources are retained at
`artifacts/toy100-constraints/simpler22-independent-audit-fec2974` in the
persistent-noise research worktree. The original `ddfb1b3` audit remains retained
separately. To rerun the archived script, first copy that
directory to fresh temporary storage because the script intentionally denies
all `artifacts` paths.

This is a fixed-seed, fixed-budget evidence audit. It does not establish
constant-rate learning, indefinite stability, unseen-seed performance or
adaptation to a shifted distribution. Seeded initialization is now independently
replayed by the checker. The Python guard is not an OS sandbox
against malicious native extensions. The historical public-default 19-case
control is absent from this bundle and contributes no candidate passes.

No concrete bundle errors were found.

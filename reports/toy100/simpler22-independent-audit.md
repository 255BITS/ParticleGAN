# Independent copied-bundle audit: PASS 22/22

The new `reports/toy100/simpler22` bundle passes the complete common gate after
copying all 190 files to a fresh temporary directory and denying access to the
original evidence. The copied files matched their originals byte for byte.
No main-worktree file was edited and no model was retrained.

The checker was frozen in a detached worktree at
`ddfb1b37d4dc06c31c0213f4f20a8cba5ed9ea53`. During the audit, a Python audit hook
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
guard script, copied bundle, regenerated GIFs and full log are retained at
`artifacts/toy100-constraints/simpler22-independent-audit-ddfb1b3` in the
constant-game research worktree. To rerun the archived script, first copy that
directory to fresh temporary storage because the script intentionally denies
all `artifacts` paths.

This is a fixed-seed, fixed-budget evidence audit. It does not establish
constant-rate learning, indefinite stability, unseen-seed performance or
adaptation to a shifted distribution. It uses the checker before the forthcoming
stricter seeded-initialization check. The Python guard is not an OS sandbox
against malicious native extensions. The historical public-default 19-case
control is absent from this bundle and contributes no candidate passes.

No concrete bundle errors were found.

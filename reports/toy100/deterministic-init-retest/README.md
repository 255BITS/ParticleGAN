# Leaderboard retest with develop initialization

<!-- current-closeout -->
**Resumed retest complete: no qualified default.** The 22 research cases previously
marked `UNTESTED` all ran with the merged initialization and failed the quick screen;
SN3 failed its uninterrupted long hold. [Results and evidence](resumed-new-init-results.md)
and the [97-row research leaderboard](research-leaderboard.md) include the new scores.
The [earlier closeout](closeout.md) preserves the historical pause and its limits.
<!-- /current-closeout -->

Every new measurement in this directory must use develop's actual deterministic
initialization (`batch_feature_zero`) for fresh networks and recipe-created
learnable priors. Sampling remains stochastic on the existing declared streams.
The user requested this baseline change; merely merging code while restoring an
old random initial model does not satisfy it.

Develop pin: `c720645ecae6b648e9fc6034e9d6b48ccff06ed3` (PR194).
Research merge: `c714f59d` (60 CPU tests pass). API candidate merge:
`25751c0864dd8259b00c5804f600cd41cce6e4cf` (246 CPU tests pass, one CUDA test
skipped in the authoritative CPU run). Both include the same develop initializer.
Earlier measurements remain under `../continuous-api-search/` with their original
sources and initialization. Neither PR is being merged into develop here.

**RP12 is the first confirmed old-failure to new-pass change:** its exact old
tiny-screen score was0/24; the new initialization reaches all eight modes at300
and retains19/19 passing observations through1200 (final HQ99.83%). Its prior
image intensity failure also becomes a pass: first325, then12/12 passing checks
through600, compared with only4 passing checks under the old initialization.
Its broader image follow-up then fails bars4 at 0/24, so this version is rejected
for default selection. The improvement remains a useful research result.
The first new attempt stopped on
a nested-diagnostic logging error; the reviewed serializer correction changes
no learner, sampling, initialization or scoring behavior, and the error is retained.

All49 declared quick-screen configurations (including the three public controls
and the separate historical RP1 eager diagnostic) have completed. The four
passing screens are RP12, RP15, RP14 and DV12. RP14 also changes from an old
image failure0/24 to a new image pass9/24, with one early quality departure;
RP15 passes the image gate5/24 but has no completed old comparison. These three
image improvements and DV12's vector failure are in [follow-up results](followup-results.json).
All twelve declared image checks are now complete: RP12, RP14 and RP15 each pass
intensity2, blobs4 and stripes2, but fail bars4 at 0/24. Thus all four quick-screen
survivors have a measured failure on another task. Further precision ring/vector/
long qualification is explicitly NOT_RUN after these failures; the reviewed
source preparations and completed CPU proofs remain available.
See [the image summary](image-runtime-review/complete-image-batch-audit.md).

DV12 was the first new-initialization screen survivor: it reaches all eight
modes at update650 and passes all12 observations from there through update1200.
Its exact adaptive-rate package then failed the unequal-mass target that previously
rejected it: zero of24 passing observations, covariance error .8793 above the .85
limit, and minimum mass ratio .2319 below .25. See the [follow-up audit](dv12-vector-runtime-audit.json).
It is not a release winner. No matching old tiny-screen measurement exists for
DV12, so its new tiny-screen pass is not labeled an old-fail-to-new-pass change.

Late arrivals receive separate retention measurements: unchanged checkpoint
continuations show DV2 passing25/25 observations since arrival at1200, and DV3
passing26/26 since1150, through2400. DV1 and DV4 each have one quality departure
and later recover. [Full continuation evidence](port-source/late-retention-terminal-audit/table.md)
keeps the original1200-update scores intact while recording arrival and stability.
Their own 4,600-update ring follow-ups now complete the comparison: neither
reaches full quality on the original target before the change at 2,400. DV2 then
reaches the new target after 380 updates and passes 183/183 later checks; DV3
after 500 updates and 171/171. Both autonomously reopen at update 2,407. This is
successful later adaptation with unverified initial acquisition, not a release
win or a recovery-deadline failure. [Old/new ring comparison](dv23-ring-runtime-review/audit.md).

The three public controls have completed the new-initialization screen.
K3P and default KA2 both finish with 6/8 modes; constant-rate KA2 finishes with
4/8. All three score 0/24 passing observations. These are measured coverage
failures, not initialization or harness errors. See [the new leaderboard](leaderboard.md)
and [the full source inventory](../continuous-api-search/fixed-init-retest-inventory/README.md).
The 46 experimental API configurations and historical research entries are tracked
in [the retest queue](retest-queue.md); no default has been selected.
Original research mechanisms have their own [new-init scores](research-leaderboard.md),
separate from public API measurements. Original research KA2 also fails this
strict screen, ending at7/8 modes. Its old host's looser diagnostic is preserved
but does not override the common eight-mode gate.

The API leaderboard runs candidates through the public `GANTrainer` path. The
research leaderboard runs each original custom research learner. Both use the
same 1,200-update, 24-observation coverage screen and merged initialization;
only the API runs establish behavior through the public trainer.

The quick hard screen is the existing small-particle `mode_hold`: 12 particles,
latent dimension4, batch128, 1200 updates, 24 observations and the unchanged
final-five-check rule. It was a useful discriminator under the old initialization;
its new difficulty and results will be measured. No old pass transfers.

Before each run, verify the exact candidate source plus initializer source, the
public factory route, deterministic initial parameters and derived buffers, and
unchanged sample/data stream semantics. Preserve each existing candidate's
algorithm, declared rate/noise policy and eligibility classification. This is a
retest of existing candidates, not a coefficient or random-seed search.

The inventory must include historical research leaderboard entries, exact public
API variants/revisions, released-formulation references, and the previously
measured constant-rate public KA2 control. Entries without an old measurement on
this exact screen receive an explicit missing-comparison label. Missing source
must be reported, never replaced with a guessed implementation.

For every previously failing configuration that passes the new screen, investigate
its other recorded weaknesses under the same new initialization: image/vector
quality, acquisition and retention, repeated target changes, complete API binding,
and checkpoint/budget independence as applicable. Preserve new regressions too.
A screen pass does not establish continuous-learning eligibility or a release
winner. Horizon-dependent configurations keep that separate limitation.

Three external Astra/max sessions may run independent frozen-candidate batches,
with one GPU worker each. Freeze and review the common harness first; finish and
archive old-init work before launching the new batches. Keep all samples random
according to their recorded streams and all scored observations unchanged.

The user requested a finite closeout: finish this retest and the required
follow-ups, update the PR and its body with the final evidence, then stop the
search. Do not start another mechanism search or merge either PR.

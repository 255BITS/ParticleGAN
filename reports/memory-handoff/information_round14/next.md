# After round14

User requested diagnosis followed by scouts; both completed. Read assessment.md,
diagnosis.md, scout_information.md and diagnostics.md. Five2k scouts on both
GPUs, no failures, none qualified for extension. All full cold/warm256/1024
passes remain0/128. All diagnostics done, queue sealed/empty, nothing running.
No next experiment selected/queued. Included in the user-requested commit with
round13 and the winner audit. Push was not requested.

## User's next direction after compaction

Investigate whether sequential training needs special treatment. Start from the
actual training loop and first principles before selecting another architecture
or sweep. The previous persistent/fast-state proposal is only one possibility.

Questions to distinguish: state carry/reset and detach policies; temporal credit
assignment (real-prefix BPTT versus one generated-write gradient); changing memory
representations as D's writer learns and G must track it; D/G update timescales;
and the joint particle/memory distribution under repeated fixed-z feedback.
Check previous experiments before proposing frozen-writer or recurrence controls.
Seek config-driven, cross-domain tests within the existing no-full-generated-
rollout and no-MSE-training constraints. No experiments chosen or launched yet.

Keep saved round12 match_shuffle25:5k nominal reference,2k stronger late-quality
control. New best future_full10 minQ .009970 vs .010901 saved2k; radial32 slightly
better .909 vs .945, lateQ worse .008820 vs .010869. Other new minQ: mixed25
.008107, mixed10 .007403, mixed10_detachwrite .006626, clean10 .006507.

Diagnosis: held-out2048/512/1024 probe histories, linear/nonlinear/M+z controls.
Saved5k speed R² .954 at handoff -> .436 after32writes -> approximately0 at128;
real-history128 .946. Early representation shift also matters. This is loss of
readily decodable information, not proof of information-theoretic erasure.

New future-ranking loss improves clean information at strongest weight (radius/
speed R² .685/.965 vs2k .612/.942) but generated32 drops to .112/.190 and128
remains chance. All five new scouts lose accessible information by128. Clean-only
has best new generated32 decoding but worst Q; useful decoding does not imply
correct G use. Connected generated-context gradient helps early over detached,
but no consistent32 advantage and no128 preservation. Shared-head h12 ranking
improves modestly; strongest weight also reduces immediate rank accuracy.

Implementation: optional future_rank_* config fields. Zero-initialized bias-free
horizon projection before first activation of SAME point head; h0 runtime path
exactly unchanged. D-only future ranking offsets4/12, clean/mixed contexts,
one independent detached G proposal/write, average horizons/contexts and normalize
by1+weight. Existing immediate mismatch .25 and G losses unchanged. Control
detaches ONLY explored future context, preserving clean-future/original writer
gradients. Both donor/anchor pools filter out unavailable future targets.

77 tests, two4-step GPU smokes, exact four-update h0 architecture equivalence and
resume, no-MSE/gradient/causality checks. All training sources/panels match. Cost
16.1min queue wall,26.8 GPU-min training, ~35s total information probes including
references. Existing process/local/future ranking diagnostics also complete.

Earlier possible direction: preserve process content through repeated writes,
perhaps learned separation of persistent process content and fast observation
state with an explicit local adversarial preservation incentive. Earlier fixed
slow/fast and G-GRU scouts failed; do not repeat them without a new interaction.
The user's broader sequential-training question above now guides the next pass.

Constraints: no MSE GAN training, full generated training rollouts, geometry
labels/cursor, clipping/EMA/B-cap overrides or seed sweeps. Fixed z, D-owned M,
runtime expert-free; metrics after completion. Stable log unchanged:
`tail -F runs/memory_path/core_round1/train.log`.

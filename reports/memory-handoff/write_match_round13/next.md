# After round13

Read assessment.md and diagnostics.md. Five2k scouts complete, no failures,
no extension qualified. Nothing running/queued. No next experiment selected.
Included in the subsequent user-requested round14 commit. For the latest handoff,
read ../information_round14/next.md; the next focus is sequential-training needs.

Keep round12 match_shuffle25:5k nominal reference and2k late-quality control.
All round13 full cold/warm passes remain0/128. Best new explored_mild has
minimum warmQ .009046 versus .010901 saved2k; radial slightly better, Q worse.
Head-only control is worse still, but removes clean AND explored mismatch writer
gradients, so it cannot isolate generated-write gradient benefit.

The new diagnostics show substantial initial speed response that fades over
32–128 generated writes. Saved5k median speed response .859 -> .192 -> .00066
in32-point windows beginning after0/32/128 writes; ideal1. Internal information
erasure versus retained-but-unused information is still unresolved. Suggested
next step is evaluation-only frozen-memory process decodability across depth,
with proper held-out histories and probe controls; not selected or implemented.

New context fields default to old clean behavior. One detached G proposal and
one generated replacement write in an independent D mismatch branch. M remains
D-owned; G objective unchanged. Mixed contexts equally average losses, not memory.
75 focused tests, two GPU smokes,39-tensor bitwise legacy equivalence. All training
sources/panels match across scouts. Cost~11.8min wall/~19.7GPU-min training.
Stable tail unchanged: runs/memory_path/core_round1/train.log.

Constraints persist: no seed experiments, MSE training objectives, full generated
training rollouts, circle cursor/labels, clipping/EMA/B-cap overrides. Fixed z,
expert-free runtime; completed-only metric decisions. User favors cross-domain
mechanisms. Code/config support is backward compatible. Reports preserve failed
ideas rather than removing their configuration support.

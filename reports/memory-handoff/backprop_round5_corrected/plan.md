# Round 5 corrected: G gradients through one feedback write

Four fresh2k scouts change only feedback_backprop from their completed matched
counterparts. Their names append_bp_v2 to the corresponding source configs.
D always treats the generated proposal as detached; D trains its writer. During
G training, D parameters are frozen and G's gradient passes through both G calls
and the intervening write. Both real and fake candidate scores depend on the same
generated M and therefore participate in the paired API G loss gradient.

| Scout | Matched detached control |
|---|---|
| feedback_p25_bp_v2 | feedback_round3/feedback_p25 |
| feedback_p50_mature_bp_v2 | feedback_round3/feedback_p50_mature |
| recent4_delta_fb25_bp_v2 | recent_round4/recent4_delta_fb25 |
| recent4_delta_fb50_mature_bp_v2 | recent_round4/recent4_delta_fb50_mature |

No full generated training rollout, path loss, private G memory, or new auxiliary
loss. Same fixed particles, data stream, architecture per matched pair,2k steps,
10k schedule, and API B-cap defaults. No clipping/EMA or seed sweeps. The targets
remain recovery targets; retaining this gradient does not fix target ambiguity
by definition. It is a distinct bounded feedback experiment.

A fifth job, recent4_delta_5k_v2, exactly resumes the completed2k recent4_delta
checkpoint for3000 additional updates. Its60.2% late-window circle rate motivated
this diagnostic extension, but its full cold/warm pass rates were0. No burn-in
will be dropped and this model is not declared a winner.

60 focused tests passed, including exact resume, bounded calls, writer ownership,
causal timing, and cancellation of common context-only score offsets in the paired
G loss. A corrected full-batch GPU smoke passed. The reporter additionally rejects
invalidated queues/baselines. Current queue:
`runs/memory_path/backprop_round5_corrected`. One central tail continues unchanged.

## Invalidated first attempt

The initial backprop_round5 implementation kept the real score under no_grad,
which omitted its dependence on generated M. Two jobs completed before the error
was caught; their results are excluded entirely. Two active jobs were deliberately
terminated and one pending job cancelled. All5 intended comparisons are restarted
here with distinct_v2 names. The5k extension restarts from the original unaffected
2k checkpoint, not the interrupted job. Earlier detached-feedback and architecture
rounds are unaffected because their G contexts have no G-gradient dependence.

New checkpoints record feedback_gradient_version=2; resuming an old backprop
checkpoint is rejected. Legacy detached checkpoints remain resumable.

# Round 6: noiseless diagnostic controls

Two fresh 2k configs change only noise from .03 to 0 relative to completed
matched models:

| Scout | Matched noisy model |
|---|---|
| clean_gru | handoff_dense4 |
| clean_recent4_delta | recent4_delta |

These retain the same distributions of circle center, radius, direction, speed,
and initial phase. Both training and observed real evaluation prefixes are now
noise-free; cold evaluation has no observed prefix in either condition. Treat
this as an easier data-setting diagnostic, not an apples-to-apples victory on
the original noisy warm-continuation task. There is no generated feedback write
in either model and no extra loss. Same 2k updates / 10k schedule, fixed particles,
D-owned memory, API B-cap, no clipping, no EMA, no seed sweep. Long rollouts are
post-training evaluation only.

Queue `runs/memory_path/clean_round6`, both GPUs and the same central train.log.
The question is whether removing observation noise materially changes autonomous
stability; exact deterministic next-point modeling is easier than reproducing a
noisy conditional distribution with a fixed particle.

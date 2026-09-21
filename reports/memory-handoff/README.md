**Start here:** [circle-toy collaborator guide](../../docs/circle-toy.md).
Latest completed round: [separate G state round18](state_round18/assessment.md).
The round12 shuffled-mismatch reference remains ahead; autonomous circle tracing
is unsolved. The round10/11 overview below is historical.

# Local memory GAN scouts: no full-rollout training

Latest: [Local adversarial recovery](recovery_round10/assessment.md).
Seventeen2k scouts and one exact5k continuation completed without failures. Proposal repair with mixed judging
and a25% local-pair GAN has the best continuous orbit progress; complete-circle
passes remain0. Its exact5k extension regressed; a matched no-adapter control was
worse. Keep the2k checkpoint. [Follow-up results](recovery_round10/followup/assessment.md).
All training and diagnostics are finished.
[Continuous metrics and full leaderboard](recovery_round10/leaderboard.md) now
separate radial/signed-motion quality, partial good arcs, and full success.
[Matched history probes](recovery_round10/baseline_probes.md) test whether radius,
speed, and direction actually survive autonomous continuation.

Previous: [Adversarial exploration](exploration_round9/assessment.md),
[clock motion milestone](clock_round7/assessment.md), and
[older rollout video](dynamics_round8/slow16_r10_rollout.mp4).
Current handoff: [NEXT.md](../memory-path/NEXT.md).
The sections below describe the original round, before local feedback/pair losses.

The user requested a round without cold loss to avoid generated-trajectory training
cost. This round removes **both** cold and warm trajectory losses. There is no
trajectory critic in the new trainer. Each G call produces independent next-point
predictions conditioned on real-history memory. Longer autoregressive rollouts are
used only in the final evaluation, after training finishes.

## Common formulation

1. Sample 128 real 64-point episodes and one particle per episode.
2. Sample four target positions per episode, reusing its particle for all four.
   Each target has probability .125 of being position zero; otherwise its prefix
   length is uniform from 1 through the configured maximum (default 63).
3. D encodes real history once. Gather memory snapshots strictly before each target.
4. G makes 512 independent single-point predictions in one batched call.
5. D compares each fake point with the real target using **the same cached M**.
   Scoring real/fake points never writes to memory. D alone updates writer weights.
6. For the G update, rebuild M once with updated D weights. Freeze D's weights and
   train G and the particles using the same point GAN objective. No fake is written
   back into memory during training, and G has no private recurrent state.

The memory encoder remains sequential over real observations; that cost has not
vanished. Caching eliminates repeated encoding for real/fake/penalty calls. The
reader is a batched feedforward network, without generated-state BPTT. Tests compare
cached and recomputed writer gradients with an active exact B-cap penalty.

All new scouts use 2k updates, the common 10k schedule, default public API B-cap,
no clipping, no EMA, and no geometry/MSE losses. Noise changes only the conditioning
history or snapshot; real next-point targets remain unchanged. Zero-prefix M stays
exactly zero even in the memory-jitter scout. Random streams for episode data,
particles, target positions, input corruption and state jitter are separate, so
changing memory size does not change the target-position stream. No seed sweeps.

## Configs

All variants build on `handoff_dense4`; only the stated field changes.

| Run suffix | Change | Question |
|---|---|---|
| dense4 | four sampled target points per real episode | Does denser local supervision help? |
| ctx8 | max real prefix 8 | Does cheaper, shorter context suffice? |
| ctx16 | max real prefix 16 | Intermediate history/cost tradeoff |
| input03 | extra input-history noise, std .03 | Robustness to small feedback errors |
| input10 | extra input-history noise, std .10 | Stronger conditioning corruption |
| state05 | memory snapshot jitter, std .05 | Robustness around encoded memory states |
| m8 | GRU memory size 8 | Smaller dynamical state |
| m64 | GRU memory size 64 | More encoding capacity |
| interaction | add learned memory/point embedding inner product to D score | Encourage D to use the relationship between M and candidate |
| start25 | zero-prefix fraction .25 | More training of autonomous initialization |

The historical `handoff_only` result is included from the completed prior round;
it is not retrained. It used one point per episode. Dense scouts have four times
as many point examples per optimizer update, so this is not an equal-supervision
comparison. Memory size/head variants also change capacity. Report seconds/update,
actual parameters, update count and point examples alongside quality. The first
smoke includes startup overhead and is not a steady-state speed claim.

## Evaluation and selection

Retain exactly the cold256/1024 and real-prefix8/32 continuation panel from the
core round, including startup, direction, stopping, original-orbit fidelity,
continuous errors, memory statistics and zero/shuffle interventions. Evaluation
never receives real points after its declared cutoff. Metrics only; no images.

Rank original-orbit preservation and cold autonomous stability separately, checking
both directions and coverage. Good one-step prediction is insufficient evidence
of useful runtime behavior. Only promote after completed metrics warrant it.
No extensions were prequeued. After all scouts completed, two diagnostic 5k
continuations were selected from metrics; see the assessment linked below.

## Queue and logs

Ten configs in `experiments/configs/memory_handoff/`; trainer
`experiments/memory_handoff_scout.py`. The append-only queue is
`runs/memory_path/handoff_round1`. Both GPUs drain it. Reporting runs only after
completed jobs and before completion notification.

**The previous tail command continues to work:**

```bash
tail -F runs/memory_path/core_round1/train.log
```

The new queue's `train.log` points to that same central log; old output is retained.
Its lifecycle records and completion-only `notifications.log` are separate under
`runs/memory_path/handoff_round1`.

[Completed leaderboard](round1/leaderboard.md) and [numeric results](round1/results.json).

Initial ten scouts are complete: [assessment](round1/assessment.md).
Two selected 5k checks: [combined leaderboard](long/leaderboard.md).

All twelve jobs are now complete, without failures. Neither 5k check solved the
rollout metrics. [Final assessment and recommendation](long/assessment.md).

# ParticleGAN 0.8.0 baseline

The current package defaults **pass stationary retention but fail timely target-shift
recovery** on the eight-mode ring. These are new measurements from clean source
`e21e77281d3270046b8de751c03d9da9ec81da88`, with one declared seed (0), using
`get_recipe()` and `GANTrainer`. No hyperparameters were tuned during the runs.

| Measurement | Passing checks | Minimum live HQ | Verdict |
|---|---:|---:|---|
| Hold confirmation, updates 1201–1400 | 200/200 | 98.68% | PASS |
| Disjoint hold, 1401–2600 | 1200/1200 | 98.83% | PASS |
| Extension, 2601–2900 | 300/300 | 98.95% | PASS |
| Remaining hold-run budget, 2901–7500 | 4600/4600 | 98.97% | PASS, descriptive continuation |
| Shift-run stationary checks, 1000:50:1200 | 5/5 | 98.14% | PASS |
| Shift-run prehold, 1210:10:2400 | 120/120 | 98.32% | PASS |
| Shift recovery deadline, 2800:10:3600 | **0/81** | **42.26%** | **FAIL** |
| Frozen control at the shifted target | 0/81 | 0.00% | Expected negative control |

HQ is the fraction of 4,096 generated samples within 0.21 (three target standard
deviations) of a mode center. Each live check requires HQ >= 90% and all eight
modes represented. Metrics use a fixed evaluation stream isolated from training;
these are longitudinal checks, not independent trials or a robustness estimate.
EMA outputs are diagnostic and do not decide the verdict.

The hold run completed all 7,500 updates in 147.18 seconds. Its first passing
observation was at 700; all 6,300 dense observations from 1201 through 7500 passed.
Final live and EMA HQ were both 99.54%, with eight modes and effective mode count
7.99. The shift run completed 3,600 updates in 63.12 seconds. Both used RTX A6000
GPUs shared with unrelated training, so these times are not isolated throughput
benchmarks. The 54 runner/K3P correctness tests passed again before completion.

## What recovery did

The shift translates every center by (1,0) immediately after update 2400. The
frozen control is restored from that exact learner checkpoint and never updated.
Its HQ is zero throughout the deadline window. The live model moves toward the
new target, but no post-shift observation reaches the quality threshold.

| Update | Live modes | Live HQ |
|---|---:|---:|
| 2400, before shift | 8 | 99.34% |
| 2410 | 0 | 0.00% |
| 2500 | 4 | 10.13% |
| 2600 | 7 | 17.07% |
| 2800, deadline begins | 8 | 42.26% |
| 3000 | 8 | 76.05% |
| 3200 | 8 | 84.11% |
| 3400 | 8 | 87.55% |
| 3600 | 8 | 89.09% |

All 81 deadline observations represent eight modes. The measured failure in that
window is insufficient concentration near the new centers, rather than missing
modes. Final effective mode count is 7.83; final EMA HQ is 85.74%. Ending close to
90% does not excuse the earlier deadline failures or establish eventual recovery.

## Implications for reducing hyperparameters

This baseline establishes a retention reference and exposes an adaptation target.
At the shift, generator and critic rates have already reached 0.0000425 (1% of
their initial rate), and the penalty blend is fully caps plus gradient anchoring
(`s=0`). The prior rate drops from about 0.00796 at update 2400 to 0.000425 at 3600.
Slow adaptation is consistent with these restrictions, but the baseline cannot
separate the effects of learning rate, anchoring, damping, or their interaction.

1. **First compare `reg_anchor_weight=0`, keeping everything else fixed.** This
   removes the temporal penalty and its decay choice from the active mechanism.
   Run the same hold/shift pair at seed 0. Measure how much retention and recovery
   actually depend on the anchor; do not assume removing it helps.
2. **Separate penalty blending from the learning-rate schedule before claiming
   annealing is unnecessary.** Constant LR also keeps `s=1`, disabling the late
   caps/anchor formulation. That experiment changes two mechanisms at once.
3. **Keep these full windows as acceptance gates.** A high final HQ, a quiet
   optimizer, or mode count alone is insufficient. Avoid introducing surprise
   thresholds/reset timers until a simpler comparison justifies them.

These are recommendations only; no follow-up variants or seed repeats were run.
The two protocols override only `total_steps` (7500/3600) in the default recipe.
That also changes the prior and noise schedules, so their initial trajectories
are not a matched stationary-versus-shift causal pair. The shift's own checkpoint
and frozen control provide its matched comparison.

Historical 22/22 and recovery 28/81 results used different host settings and
initialization. They cannot establish a regression or improvement here. Full toy
and native qualification, repeated shifts, and horizon independence remain
unmeasured for this fresh baseline.

## Evidence and reproduction

- [Parameters and commands](README.md), [resolved defaults](parameters.json).
- [Computed window summaries and trajectory](summary.json).
- [Hold result](evidence/hold/result.json), [shift result](evidence/shift/result.json).
- Per-run declarations, source snapshots and lossless compressed metrics are in
  [evidence/hold](evidence/hold/) and [evidence/shift](evidence/shift/).
- [Artifact hashes](artifacts.json) include the full learner/data-stream
  checkpoints retained under `runs/public-default-baseline/` in the experiment
  worktree. Checkpoints are local; the compact evidence above is committed.

Tail the uncompressed logs locally with
`tail -F runs/public-default-baseline/{hold,shift}/metrics.jsonl`.

# Class scale 6: better consistency, weaker distribution fit

The new class-6/context-2 run scores **0.11996 joint SW1**, fifth of ten. It does
not beat the **0.10027** marginal-critic leader. Against class-8/context-2, SW1
worsens **12.6%**, residual improves **15.0%**, and coverage improves **12.1%**
(2.58 percentage points). The hoped-for improvement in both classes did not occur.

Only `g_class_scale` changes from 8 to 6, plus the output path. MoG1024, fixed
sigma, bcap, one concat joint critic, 28,000 updates, batch256 and seed24002 stay
fixed. G/D parameters remain 65,286/135,169. Training took 242.8 seconds; wall
times across runs are not controlled throughput comparisons. No seed repeats.

```text
G1 -> st
G2 -> at
G3 -> st+1
D_joint(st, at, st+1)
```

The three independent Gs receive the same noisy MoG draw, class indicators scaled
by 6, and geometry/time inputs scaled by 2. There is no physics loss or forced
transition identity.

| Rank | Setup | Joint SW1 ↓ | Residual ↓ | Coverage ↑ |
|---:|---|---:|---:|---:|
| 1 | Class 8/context 1, joint + marginal Ds | **0.10027** | 0.02393 | 0.158 |
| 2 | Class 8/context 1, joint D | 0.10563 | 0.02177 | 0.195 |
| 3 | Class 8/context 2, joint D | 0.10655 | 0.01909 | 0.212 |
| 4 | Class 8/context 4, joint D | 0.10692 | **0.01550** | 0.194 |
| 5 | **Class 6/context 2, joint D** | 0.11996 | 0.01622 | **0.238** |

Reference SW1 floor is 0.03788; reference coverage is about 0.954. These geometries
are a reused development benchmark. Single-seed differences do not establish
robust rankings, especially the small gaps between the class-8 joint-only runs.

![Distribution, consistency and coverage tradeoffs](conditioning_tradeoffs.png)

## What changed

| Diagnostic | Class 8/context 2 | Class 6/context 2 |
|---|---:|---:|
| Class 0 SW1 | 0.13030 | 0.12895 |
| Class 1 SW1 | 0.08280 | 0.11096 |
| Class 0 upper frequency (target .8) | 0.90088 | 0.89209 |
| Class 1 upper frequency (target .3) | 0.29004 | 0.34521 |
| Interpolation SW1 | 0.08701 | 0.10223 |
| Extrapolation SW1 | 0.16518 | 0.17314 |
| Interpolation residual | 0.01677 | 0.01421 |
| Extrapolation residual | 0.02606 | 0.02224 |
| Interpolation coverage | 0.27793 | 0.31374 |
| Extrapolation coverage | 0.01563 | 0.01133 |

Class 0 barely improves while class 1 deteriorates. Higher coverage comes from
interpolation; extrapolation coverage actually falls to 1.1%. State/action/next
SW1 all worsen, reaching 0.10391/0.13733/0.10608. This is a consistency/coverage
tradeoff, not the desired class-balance fix.

Train SW1/residual/coverage is 0.09117/0.00546/0.613, versus held-out
0.11996/0.01622/0.238. Held-out residual is about 44% of mean reference displacement.
Shuffling generated blocks raises SW1 to 0.18220 and residual to 0.17914, showing
useful coordination still exists.

The new [geometry/class audit](geometry_class_audit.json) separates each of four
geometries and both classes, averaging the existing five per-context SW1 and
coverage measurements. Upper frequencies use the saved midpoint samples (512 per
geometry/class); residual uses all five ticks. It introduces no new reference
samples or ranking metric.

Class 6 overproduces class-0 upper states across all four geometries (0.879–0.904).
The existing marginal-critic leader also overproduces them (0.869–0.891).
Its extrapolated class-1 upper frequency is 0.297, close to target 0.3, while its
coverage is almost zero. Thus correct route frequency alone cannot explain or
repair the geometry generalization failure. Side counts do not measure support
validity, and the current aggregate audit does not identify the cause.

## Recommendation

Retain class-8/context-1 with marginal critics as the score leader. Stop input-scale
tuning for now. Next, run a **paired-latent diagnostic on frozen checkpoints**:
reuse exactly the same noisy z across classes and train/test geometries, then
measure upper/lower assignments, cross-branch agreement, residual, and spatial
error within each route away from the endpoints. Compare the leader with
class-8/context-2. This can distinguish changes in route assignment from errors
in where each branch places that route, before choosing another training change.
No next training run is configured or launched. Reserve untouched geometries for
eventual validation after selection.

## Validation and artifacts

Model/trainer/evaluation code is unchanged. The preceding 19-test pass still
applies; tests were not rerun for this config-only experiment. The new checkpoint
exactly replayed its first 256 saved test samples, recomputed test metrics matched
within 1e-6, and MoG sigma stayed fixed. All ten entries pass the registry's pinned
source/protocol/reference checks. Both plots were inspected. Training and
verification processes exited.

[Full leaderboard and viewers](README.md) · [Per-class audit](class_distance_audit.json)
· [Previous readout](ROUND4.md).
Config: `configs/transition/concat_class6_context2.yaml`.
Run: `results/transition/conditioning/branches_concat_class6_context2`.
Logs: `tail -F results/transition/live.log`.

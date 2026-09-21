# Context scale 2: more coverage, same score plateau

The intermediate scale gives **0.10655 joint SW1**, **0.01909 residual**, and
**0.212 coverage**. Relative to context scale 1, residual improves 12.3% and coverage
8.9% (1.74 percentage points), while SW1 worsens 0.9%. It ranks third of nine;
`concat_class8_marginals` remains the score leader at **0.10027**.

This configuration-only experiment retains class scale 8, one concat joint
critic, MoG1024, bcap, 28,000 updates, batch256 and seed24002. Only context scale
and output path differ from its baseline. G/D parameters remain 65,286/135,169.
Training took 252.1 seconds; wall times across runs are not controlled throughput
comparisons because GPU workloads can differ. No seed-only repeats.

```text
G1 -> st
G2 -> at
G3 -> st+1
D_joint(st, at, st+1)
```

All three independent Gs receive exactly the same noisy MoG draw, class indicators
scaled by 8, and geometry/time inputs scaled by 2. No physics loss or forced
transition identity is used.

| Setup (class scale 8) | Joint SW1 ↓ | Residual ↓ | Coverage ↑ |
|---|---:|---:|---:|
| Context 1, joint + marginal Ds | **0.10027** | 0.02393 | 0.158 |
| Context 1, joint D | 0.10563 | 0.02177 | 0.195 |
| **Context 2, joint D** | 0.10655 | 0.01909 | **0.212** |
| Context 4, joint D | 0.10692 | **0.01550** | 0.194 |

Reference SW1 floor: 0.03788. Reference coverage: about 0.954. These geometries
are a reused development benchmark, not an untouched generalization test.
The three joint-only scores occupy a narrow range; this single-seed comparison
does not establish that their small SW1 differences are robust.

![Distribution, consistency and coverage tradeoffs](conditioning_tradeoffs.png)

## Interpretation

| Diagnostic | Context 1 | Context 2 | Context 4 |
|---|---:|---:|---:|
| Interpolation SW1 | 0.08454 | 0.08701 | 0.09279 |
| Extrapolation SW1 | 0.16888 | 0.16518 | 0.14933 |
| Interpolation residual | 0.01850 | 0.01677 | 0.01293 |
| Extrapolation residual | 0.03157 | 0.02606 | 0.02323 |
| Interpolation coverage | 0.25938 | 0.27793 | 0.25397 |
| Extrapolation coverage | 0.00156 | 0.01563 | 0.01563 |
| Class 0 SW1 | 0.11785 | 0.13030 | 0.12479 |
| Class 1 SW1 | 0.09340 | 0.08280 | 0.08905 |
| Class 0 upper frequency (target .8) | 0.87158 | 0.90088 | 0.88477 |
| Class 1 upper frequency (target .3) | 0.32373 | 0.29004 | 0.31299 |

Scale 2 retains some consistency improvement and reduces scale 4's interpolation
regression, as hoped. It also improves coverage over both scales. But overall
SW1 is still slightly worse than scale 1. Class 1 improves while class 0 gets
worse: matching one conditional mixture has not solved the other. The upper-side
audit is only a mixture diagnostic, not proof of correct support.

State/action/next SW1 is 0.08906 / 0.12468 / 0.09320. Train SW1/residual/coverage
is 0.07446 / 0.00539 / 0.571, versus held-out 0.10655 / 0.01909 / 0.212. Held-out
residual remains about 52% of mean reference displacement, and extrapolation
coverage is only 1.6%. Shuffling the generated blocks increases joint SW1 to
0.16763 and residual to 0.16745: useful coordination persists, with substantial
remaining error.

## Recommendation

Keep the existing marginal-critic model as score leader. Context scale 2 offers
more coverage among the class-scale-8 challengers; context scale 4 offers better
consistency. Neither repairs the generalization gap.

The next bounded comparison is **class scale 6 at context scale 2**, with one
joint D and the same recipe/budget. Earlier class scale 4 fit class 0 much better
but underfit class 1, while class scale 8 shows the opposite imbalance. An
intermediate class scale tests that tradeoff; monotonic improvement is not
assumed. No new run/config has been launched. If that does not improve both
class fits, prioritize diagnosing conditional mode allocation over further
input-scale tuning. Reserve untouched geometries for eventual validation.

## Validation and artifacts

No model/trainer/evaluation code changed. The existing 19-test suite passed on
this implementation in the preceding round; it was not rerun for a config-only
change. The new checkpoint exactly reproduced its first 256 saved test samples;
recomputed test metrics matched within 1e-6, and MoG sigma stayed fixed. All nine
entries pass the registry's pinned source/protocol/reference checks. Plots were
inspected. Training and verification processes have exited.

See the [full leaderboard and viewers](README.md),
[per-class audit](class_distance_audit.json), and [previous readout](ROUND3.md).
Config: `configs/transition/concat_class8_context2.yaml`.
Run: `results/transition/conditioning/branches_concat_class8_context2`.
Logs: `tail -F results/transition/live.log`.

# Geometry/time scaling: better consistency, no new score leader

Increasing G geometry/time scale from 1 to 4 at class scale 8 reduces held-out
transition residual by **28.8%**, but joint SW1 worsens **1.2%** to **0.10692**.
Overall coverage is nearly unchanged. The marginal-critic model remains the
primary-score leader at **0.10027**. The new run ranks third of eight.

Only `g_context_scale` changes in the matched comparison, apart from output path.
Both runs use one concat joint critic, 1,024 MoG components, the default MoG/bcap
recipe, 28,000 updates, batch 256, seed 24002 and the same reference observations.
G has 65,286 parameters; D has 135,169. Training took 265.8 seconds, compared with
283.1 for the baseline; GPU workloads differed, so this is not a throughput claim.

```text
G1 -> st
G2 -> at
G3 -> st+1
D_joint(st, at, st+1)
```

Each independent G receives the same noisy z, class indicators scaled by 8, and
four geometry/time inputs scaled by 4. There is no shared trunk, physics loss or
forced transition identity. Positive input scaling leaves the representable
function family unchanged because first-layer weights can absorb it.

| Setup | Joint SW1 ↓ | Residual ↓ | Coverage ↑ | Upper class 0 / 1 |
|---|---:|---:|---:|---:|
| Class 8, context 1, joint + marginal Ds | **0.10027** | 0.02393 | 0.158 | 0.882 / 0.309 |
| Class 8, context 1, joint D | 0.10563 | 0.02177 | 0.195 | 0.872 / 0.324 |
| **New: class 8, context 4, joint D** | 0.10692 | 0.01550 | 0.194 | 0.885 / 0.313 |
| Class 4, context 1, joint D | 0.14179 | **0.01464** | **0.301** | 0.826 / 0.485 |

Reference SW1 floor: 0.03788; reference coverage: about 0.954. Analytic upper-route
probabilities are 0.8 / 0.3. These reused geometries are a development benchmark,
not an untouched generalization test. No seed-only repeats were performed.

![Distribution, consistency and coverage tradeoffs](conditioning_tradeoffs.png)

## What the matched comparison shows

| Diagnostic | Context 1 | Context 4 |
|---|---:|---:|
| Interpolation SW1 | 0.08454 | 0.09279 |
| Extrapolation SW1 | 0.16888 | 0.14933 |
| Interpolation residual | 0.01850 | 0.01293 |
| Extrapolation residual | 0.03157 | 0.02323 |
| Interpolation coverage | 0.25938 | 0.25397 |
| Extrapolation coverage | 0.00156 | 0.01563 |
| Class 0 SW1 | 0.11785 | 0.12479 |
| Class 1 SW1 | 0.09340 | 0.08905 |

Extrapolation SW1 improves 11.6%, while interpolation worsens 9.8%. Coverage on the
extrapolation geometry rises but remains only 1.6%. Class 1 improves slightly;
class 0 worsens and still overproduces the upper route. The hypothesis receives
partial support: input scaling affects consistency and geometric fit, but this
setting does not repair the coverage gap or improve overall distribution distance.
Because geometry and time were scaled together, this run cannot attribute the
change to either input separately.

The new run's state/action/next SW1 is 0.09718 / 0.12322 / 0.09368. Training SW1 is
0.07255 and residual 0.00541, versus held-out residual 0.01550 (about 42% of the
mean reference displacement 0.03656). Held-out p95 residual is 0.03601. Training
coverage is 0.634 versus 0.194 held out: the generalization gap remains substantial.
Shuffling generated blocks within context increases joint SW1 to 0.17493 and
residual to 0.18402, showing useful but imperfect coordination between the Gs.

## Recommendation

Keep `concat_class8_marginals.yaml` as the score leader. Context scale 4 is a useful
consistency/score compromise, but it does not dominate the existing models.
The next small diagnostic is **context scale 2**, retaining class scale 8 and one
joint critic: test whether an intermediate scale retains some consistency gain
without the interpolation regression. This is a hypothesis, not a prediction;
no further run has been launched. Keep coverage and per-class distances visible,
and avoid treating input-scale tuning as a solution to missing-data recovery.
An untouched geometry set should assess generalization after model selection.

## Validation and artifacts

19 transition/trajectory tests pass, including shared inputs, unchanged parameter
initialization/counts, invalid-scale rejection, old state-dict compatibility and
training/checkpoint configuration. The new checkpoint and previous class-8
checkpoint exactly replayed their first 256 saved test samples on CUDA using the
updated model. Recomputed metrics match within 1e-6; MoG sigma remains fixed.
All eight leaderboard entries pass the pinned protocol/reference checks. Plots
were inspected. The training and verification processes have exited.

See the [full leaderboard and viewers](README.md),
[per-class distance audit](class_distance_audit.json), and [previous round](ROUND2.md).
Configuration: `configs/transition/concat_class8_context4.yaml`.
Outputs: `results/transition/conditioning/branches_concat_class8_context4`.
Logs: `tail -F results/transition/live.log`.

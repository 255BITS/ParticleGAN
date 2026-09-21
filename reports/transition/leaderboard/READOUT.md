# Encoder and shared-state-critic round — 2026-09-20

**The shared-state encoder is the new generation-score leader, at 0.08565 joint
SW1.** Both encoder variants improve the original prior-sampling benchmark over
the previous leader. Sharing the state critic further improves distribution fit,
with a coverage tradeoff. This is a paired-supervision experiment: the encoder
adds real reconstruction/prediction losses and capacity.

## What changed

```text
G1 -> st
G2 -> at
G3 -> st+1

E(real st, real at) -> z -> G1/G2 reconstruction + G3 next-state prediction
G1/G2 -> (st, at) -> E -> z_hat -> G3 -> synthetic st+1
```

Every network also receives observed geometry, time and preference class. All
three original Gs still share the same noisy MoG draw. E never sees real st+1.
Both arms use the existing MoG1024/bcap recipe, 28k updates, batch 256, seed 24002,
fixed sigma and the same real draw budget. They adapt `particle_ae` routing to
that recipe; they do not adopt the different `ae_gan` training preset.

Arm 1 retains four separate critics. Arm 2 shares one state critic between G1/G3,
using common physical-state normalization and t versus t+dt, while retaining
joint/action critics. The three marginal roles keep their original weighting.
The two arms have identical archived source, recipe, normalization and prior
metadata; only the shared-state flag and output directory differ in their configs.
[Exact objectives and gradient paths](../../../docs/transition-gan-encoder.md).

## Original prior-sampling leaderboard

These are unchanged benchmark inputs/evaluation, without E at inference.
Lower SW1/residual and higher coverage are better. This table shows the current
round and its previous leader; [the full board](README.md) contains 12 entries.

| Run | Full-board rank | Joint SW1 | Residual | Coverage | Precision |
|---|---:|---:|---:|---:|---:|
| Encoder, shared state D | 1 | **0.085647** | **0.017362** | 24.82% | 23.82% |
| Encoder, separate Ds | 2 | 0.095381 | 0.018028 | **27.71%** | **25.55%** |
| Previous leader: class 8 + marginals | 3 | 0.100269 | 0.023931 | 15.75% | 15.68% |

Against the previous leader, shared state D improves SW1 by 14.6%, residual by 27.5%,
and coverage by 9.07 percentage points. Against the separate encoder it improves
SW1 by 10.2% and residual by 3.7%, while losing 2.89 points of coverage. It is not an
all-metric winner. The reference-vs-reference SW1 floor remains 0.03788.

| Run | Interpolation SW1 / coverage | Extrapolation SW1 / coverage | Train SW1 / residual |
|---|---:|---:|---:|
| Shared state D | 0.06526 / 32.85% | 0.14681 / 0.72% | 0.05323 / 0.01210 |
| Separate Ds | 0.07449 / 36.35% | 0.15804 / 1.80% | 0.05875 / 0.01247 |
| Previous leader | 0.08008 / 21.00% | 0.16084 / 0.02% | 0.06379 / 0.00578 |

The encoders improve held-out consistency despite worse training consistency.
Extrapolation coverage remains extremely low; lower SW1 does not mean the model
covers the reference support. These reused geometries are a development benchmark.

Sharing particularly improves class 0 distribution fit. Class 0/1 SW1 is
0.09258/0.07871, versus 0.11152/0.07924 for separate critics and 0.11779/0.08275
for the old leader. Midpoint upper-route frequencies are 0.838/0.303 for sharing,
0.865/0.278 for separate critics, and 0.882/0.309 for the old leader; targets
are 0.800/0.300. These frequencies measure mixture balance, not support validity.

## New prediction and synthetic paths

| Encoder | Real-input next-state L2 / p 95 | Synthetic SW1 | Synthetic residual | Synthetic coverage |
|---|---:|---:|---:|---:|
| Shared state D | **0.01665 / 0.04780** | **0.08672** | 0.01471 | 25.24% |
| Separate Ds | 0.01794 / 0.04958 | 0.09595 | **0.01322** | **31.80%** |

The synthetic graph lowers residual relative to direct prior generation in both
arms, with nearly unchanged SW1. Separate critics retain stronger consistency
and coverage on this path. Shared state D reduces real-input prediction error
by 7.2%; its gain is mainly in class 1. Prediction error for sharing is 0.01145 on
interpolation and 0.03225 on extrapolation (separate:0.01237/0.03463).

The mean reference displacement is 0.03656, so aggregate prediction errors remain
about 46% and 49% of a typical step. Since action is displacement, the analytic
`st + at` control has zero error. We have added useful inference machinery, but
this does not establish a general dynamics model or a benefit over a dedicated
supervised predictor, which has not yet been compared.

## Routing and action-response diagnostics

Real-input E uses 7 of 1,024 components for sharing (entropy-effective 4.95) and 5
for separate critics (effective 4.12). Original G sampling still draws uniformly
across all 1,024 components. Few encoded components alone do not prove failure:
context and continuous offsets also carry information. The two latent input
distributions are nevertheless substantially different.

A frozen finite-difference audit perturbed each physical action coordinate by
±0.0001 while holding state/context fixed, using 32 saved samples per context.
These perturbations leave the training route manifold. The ideal displacement
response is the 2×2 identity; median response-matrix error was 13.68 for sharing
and 14.14 for separate critics. Component switches occurred in 2.07% and 2.42% of
axis probes. Errors remain large without switches. About 31%/25% of encoded
offset coordinates are near the imposed bound in this audit.

This diagnostic limits claims about arbitrary action inputs; it is not part of
the leaderboard or a causal-identification test. The observational route data
allows state, action and context shortcuts. A separate, coarser ±0.005 probe was
also tried on the first arm; the matched comparison uses ±0.0001 for both.

## Cost, validation and recommendation

G stays 65,286 parameters; E adds 42,688 in each arm. D has 238,084 parameters with
separate critics and 203,779 with sharing. The prior has 32,768 learned parameters.
Training took 826.9s and 954.7s, versus 565.1s for the old leader. These are not
controlled throughput comparisons. Each run uses 14,336,000 fresh real training
draws plus 32,768 normalization draws. Added real MSE reuses the generator batch.

22 scoped tests passed, including encoder gradients, shared-state inputs/time,
role weighting and checkpoint loading. Both trained checkpoints replay the first
256 prior samples, reconstructions, compositions and routing IDs exactly on CUDA
for train/test. Recomputed full-context metrics agree within 1e-6; sigma is fixed.
All 12 registry entries pass pinned source/protocol/reference and capacity checks.

**Use the shared-state encoder as the score baseline and retain the separate
encoder as the coverage/consistency control.** Next, prioritize a transition
benchmark with independently varied actions at a given state, so learning the
conditional outcome requires action response. Give that dataset its own board;
its scores cannot be ranked against this route benchmark. A matched supervised
predictor control there would test whether learning the full joint distribution
helps prediction. Avoid forcing uniform encoder routing merely to increase the
component count. No additional training runs were started.

Artifacts: [comparison JSON](encoder_comparison.json),
[separate verification](encoder_separate_verification.json),
[shared verification](encoder_shared_state_verification.json),
[separate action audit](encoder_separate_action_audit.json),
[shared action audit](encoder_shared_state_action_audit.json).
The preceding conditioning readout is preserved as [ROUND5](ROUND5.md).

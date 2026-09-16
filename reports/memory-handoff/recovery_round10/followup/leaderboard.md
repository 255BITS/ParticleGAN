# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| plain_mixed_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| plain_pair50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.0166 | 0.0082 / 0.0083 | 0.0067 / 0.0068 | 0.4% / 0.4% | 0.054 / 0.059 |
| proposal_mixed_pair25_5k | 0.0092 | 0.0067 / 0.0070 | 0.0053 / 0.0053 | 0.3% / 0.3% | 0.051 / 0.055 |
| proposal_clean_s25 | 0.0245 | 0.0062 / 0.0064 | 0.0058 / 0.0061 | 0.3% / 0.3% | 0.054 / 0.054 |
| plain_pair50 | 0.0211 | 0.0059 / 0.0064 | 0.0044 / 0.0044 | 0.3% / 0.4% | 0.043 / 0.057 |
| plain_mixed_pair25 | 0.0287 | 0.0057 / 0.0057 | 0.0044 / 0.0049 | 0.3% / 0.3% | 0.052 / 0.041 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| plain_mixed_pair25 | 8 | 1.708 | 0.265 | 51.8% | 0.093 | 1.783 / 2.721 |
| plain_mixed_pair25 | 32 | 1.707 | 0.265 | 51.7% | 0.097 | 1.608 / 2.861 |
| proposal_mixed_pair25_5k | 8 | 1.559 | 0.266 | 52.3% | 0.098 | 1.435 / 2.508 |
| proposal_mixed_pair25_5k | 32 | 1.565 | 0.264 | 51.6% | 0.087 | 1.252 / 2.517 |
| plain_pair50 | 8 | 1.690 | 0.266 | 51.1% | 0.094 | 1.733 / 2.698 |
| plain_pair50 | 32 | 1.679 | 0.265 | 51.0% | 0.090 | 1.381 / 2.777 |
| proposal_clean_s25 | 8 | 1.497 | 0.267 | 50.9% | 0.096 | 1.572 / 2.543 |
| proposal_clean_s25 | 32 | 1.487 | 0.268 | 50.7% | 0.096 | 1.448 / 2.573 |
| proposal_mixed_pair25 | 8 | 1.136 | 0.270 | 50.0% | 0.099 | 1.704 / 2.018 |
| proposal_mixed_pair25 | 32 | 1.159 | 0.264 | 50.7% | 0.095 | 1.528 / 2.158 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_mixed_pair25 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.0844 |
| proposal_mixed_pair25_5k | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.0871 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Auxiliaries disabled |
|---|---|---|---|---:|---|
| plain_mixed_pair25 | mixed | none | True | 4 / 4 | True |
| proposal_mixed_pair25_5k | mixed | proposal | True | 8 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| plain_mixed_pair25 | 0.5 | fixed / 0.25 | 0.25 |
| proposal_mixed_pair25_5k | 0.5 | fixed / 0.25 | 0.25 |

Mixed judging averages separate clean/shared GAN losses and their default B-cap penalties.
The optional pair head judges two consecutive generated points against a real pair,
conditioned on real memory before both points. Its branch has one generated write
and is independent of point-loss exploration. Point/pair losses and penalties are
convexly weighted; the prior regularizer is applied once. No third generated point.
Uniform replacements range from zero to the configured maximum; mild_full selects
the mild value or one (default 25% full). Each update reuses its strengths in D and G.

## Local memory dynamics settings

| Run | Slow coordinates / rate | G adapter bottleneck | Repair target / weight / noise | Stability D / G weight / max gain |
|---|---:|---:|---|---|
| plain_mixed_pair25 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_pair25_5k | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

Scouts use local point GAN losses and optionally a two-point transition GAN. There is no full generated training
rollout or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Auxiliary heads do not change the GAN negative class. Compare measured cost too.
Optional G repair trains a stateless read adapter. Local stability adds two parallel
one-step feedback branches per enabled phase, with detached prefix anchors and particles.
These local branches do not feed into another generated prediction; writer updates remain D-only.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).

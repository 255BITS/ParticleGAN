# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| future10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future_arch_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match25_future10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match25_recover10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match25_recover10_future10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_nearest10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_nearest25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_nearest50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| recover_noise10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| recover_noise30 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| match_nearest25 | 0.0115 | 0.0100 / 0.0099 | 0.0082 / 0.0076 | 0.5% / 0.5% | 0.065 / 0.066 |
| future_arch_control | 0.0174 | 0.0087 / 0.0089 | 0.0073 / 0.0071 | 0.4% / 0.4% | 0.052 / 0.048 |
| match25_future10 | 0.0187 | 0.0090 / 0.0084 | 0.0067 / 0.0063 | 0.5% / 0.4% | 0.064 / 0.058 |
| proposal_mixed_pair25 | 0.0166 | 0.0082 / 0.0083 | 0.0067 / 0.0068 | 0.4% / 0.4% | 0.054 / 0.059 |
| match_nearest50 | 0.0146 | 0.0075 / 0.0074 | 0.0049 / 0.0050 | 0.3% / 0.3% | 0.056 / 0.051 |
| match25_recover10_future10 | 0.0121 | 0.0074 / 0.0073 | 0.0054 / 0.0057 | 0.4% / 0.4% | 0.063 / 0.067 |
| future25 | 0.0179 | 0.0071 / 0.0079 | 0.0064 / 0.0062 | 0.4% / 0.4% | 0.050 / 0.066 |
| match_nearest10 | 0.0144 | 0.0071 / 0.0068 | 0.0051 / 0.0048 | 0.4% / 0.4% | 0.055 / 0.055 |
| recover_noise30 | 0.0304 | 0.0054 / 0.0058 | 0.0051 / 0.0048 | 0.3% / 0.3% | 0.040 / 0.046 |
| recover_noise10 | 0.0236 | 0.0049 / 0.0053 | 0.0039 / 0.0033 | 0.2% / 0.2% | 0.044 / 0.056 |
| match25_recover10 | 0.0227 | 0.0048 / 0.0047 | 0.0045 / 0.0039 | 0.2% / 0.2% | 0.033 / 0.036 |
| future10 | 0.0118 | 0.0041 / 0.0046 | 0.0017 / 0.0021 | 0.2% / 0.2% | 0.043 / 0.041 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| future10 | 8 | 2.516 | 0.319 | 49.2% | 0.106 | 1.710 / 3.758 |
| future10 | 32 | 2.421 | 0.304 | 52.3% | 0.099 | 1.538 / 3.651 |
| future25 | 8 | 1.492 | 0.272 | 50.7% | 0.105 | 1.720 / 2.377 |
| future25 | 32 | 1.477 | 0.269 | 50.8% | 0.084 | 1.266 / 2.361 |
| future_arch_control | 8 | 1.142 | 0.290 | 50.7% | 0.088 | 1.675 / 2.064 |
| future_arch_control | 32 | 1.101 | 0.287 | 51.6% | 0.104 | 1.586 / 2.096 |
| match25_future10 | 8 | 1.619 | 0.286 | 53.0% | 0.098 | 1.382 / 2.652 |
| match25_future10 | 32 | 1.624 | 0.291 | 53.5% | 0.088 | 1.422 / 2.595 |
| match25_recover10 | 8 | 1.297 | 0.323 | 50.5% | 0.100 | 1.942 / 2.301 |
| match25_recover10 | 32 | 1.285 | 0.323 | 49.9% | 0.106 | 1.757 / 2.340 |
| match25_recover10_future10 | 8 | 1.817 | 0.256 | 53.5% | 0.089 | 1.488 / 3.024 |
| match25_recover10_future10 | 32 | 1.808 | 0.256 | 53.3% | 0.090 | 1.442 / 2.871 |
| match_nearest10 | 8 | 1.890 | 0.262 | 51.7% | 0.089 | 1.521 / 2.963 |
| match_nearest10 | 32 | 1.854 | 0.267 | 51.2% | 0.091 | 1.453 / 3.048 |
| match_nearest25 | 8 | 0.990 | 0.294 | 52.4% | 0.083 | 1.319 / 1.946 |
| match_nearest25 | 32 | 0.994 | 0.290 | 52.3% | 0.085 | 1.293 / 2.003 |
| match_nearest50 | 8 | 1.204 | 0.310 | 54.3% | 0.082 | 1.448 / 2.126 |
| match_nearest50 | 32 | 1.233 | 0.319 | 53.8% | 0.084 | 1.603 / 2.154 |
| match_shuffle25 | 8 | 0.960 | 0.261 | 51.9% | 0.085 | 1.242 / 1.869 |
| match_shuffle25 | 32 | 0.945 | 0.261 | 51.8% | 0.088 | 1.237 / 1.909 |
| recover_noise10 | 8 | 1.396 | 0.273 | 49.3% | 0.079 | 1.397 / 2.434 |
| recover_noise10 | 32 | 1.386 | 0.272 | 49.5% | 0.080 | 1.346 / 2.395 |
| recover_noise30 | 8 | 1.359 | 0.279 | 48.7% | 0.102 | 1.809 / 2.356 |
| recover_noise30 | 32 | 1.347 | 0.275 | 49.3% | 0.107 | 1.622 / 2.406 |
| proposal_mixed_pair25 | 8 | 1.136 | 0.270 | 50.0% | 0.099 | 1.704 / 2.018 |
| proposal_mixed_pair25 | 32 | 1.159 | 0.264 | 50.7% | 0.095 | 1.528 / 2.158 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| future10 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1330 |
| future25 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1325 |
| future_arch_control | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.0890 |
| match25_future10 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1661 |
| match25_recover10 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1316 |
| match25_recover10_future10 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1730 |
| match_nearest10 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1172 |
| match_nearest25 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1194 |
| match_nearest50 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1172 |
| match_shuffle25 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1209 |
| recover_noise10 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1025 |
| recover_noise30 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1057 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| future10 | mixed | proposal | True | 14 / 14 | True |
| future25 | mixed | proposal | True | 14 / 14 | True |
| future_arch_control | mixed | proposal | True | 8 / 8 | True |
| match25_future10 | mixed | proposal | True | 14 / 14 | True |
| match25_recover10 | mixed | proposal | True | 8 / 8 | True |
| match25_recover10_future10 | mixed | proposal | True | 14 / 14 | True |
| match_nearest10 | mixed | proposal | True | 8 / 8 | True |
| match_nearest25 | mixed | proposal | True | 8 / 8 | True |
| match_nearest50 | mixed | proposal | True | 8 / 8 | True |
| match_shuffle25 | mixed | proposal | True | 8 / 8 | True |
| recover_noise10 | mixed | proposal | True | 8 / 8 | True |
| recover_noise30 | mixed | proposal | True | 8 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| future10 | 0.5 | fixed / 0.25 | 0.25 |
| future25 | 0.5 | fixed / 0.25 | 0.25 |
| future_arch_control | 0.5 | fixed / 0.25 | 0.25 |
| match25_future10 | 0.5 | fixed / 0.25 | 0.25 |
| match25_recover10 | 0.5 | fixed / 0.25 | 0.25 |
| match25_recover10_future10 | 0.5 | fixed / 0.25 | 0.25 |
| match_nearest10 | 0.5 | fixed / 0.25 | 0.25 |
| match_nearest25 | 0.5 | fixed / 0.25 | 0.25 |
| match_nearest50 | 0.5 | fixed / 0.25 | 0.25 |
| match_shuffle25 | 0.5 | fixed / 0.25 | 0.25 |
| recover_noise10 | 0.5 | fixed / 0.25 | 0.25 |
| recover_noise30 | 0.5 | fixed / 0.25 | 0.25 |

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
| future10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future_arch_control | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match25_future10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match25_recover10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match25_recover10_future10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match_nearest10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match_nearest25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match_nearest50 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| match_shuffle25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| recover_noise10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| recover_noise30 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## Local history and future objectives

| Run | Mismatch weight / donors | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---:|
| future10 | 0 / nearest | 0 / 0.5 | 0.1 / [0, 4, 12] | 4 |
| future25 | 0 / nearest | 0 / 0.5 | 0.25 / [0, 4, 12] | 4 |
| future_arch_control | 0 / nearest | 0 / 0.5 | 0 / [0, 4, 12] | 4 |
| match25_future10 | 0.25 / nearest | 0 / 0.5 | 0.1 / [0, 4, 12] | 4 |
| match25_recover10 | 0.25 / nearest | 0.1 / 0.5 | 0 / [0, 4, 12] | 0 |
| match25_recover10_future10 | 0.25 / nearest | 0.1 / 0.5 | 0.1 / [0, 4, 12] | 4 |
| match_nearest10 | 0.1 / nearest | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| match_nearest25 | 0.25 / nearest | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| match_nearest50 | 0.5 / nearest | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| match_shuffle25 | 0.25 / shuffle | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| recover_noise10 | 0 / nearest | 0.1 / 0.5 | 0 / [0, 4, 12] | 0 |
| recover_noise30 | 0 / nearest | 0.3 / 0.5 | 0 / [0, 4, 12] | 0 |

Mismatch training ranks real continuations from other histories with the existing point head;
its D loss and default B-cap are normalized by 1+weight. G point loss stays unchanged.
Recovery perturbs prefix observations read by G; the pair judge retains the clean reference.
Future queries read identical prefix memory independently, with fixed z and explicit offsets.
Their joint GAN uses real future targets, no generated writes, and default exact B-cap.
Future weight convexly mixes this branch with the existing GAN; prior regularization stays once.

Scouts use local point GAN losses and optionally a two-point transition GAN. There is no full generated training
rollout or cold/warm path loss. Configured feedback adds at most one
generated write before each target; configs control G gradients through that write.
Longer rollouts are evaluation only.
Dense scouts use four points per episode; the older handoff_only trainer used one.
Architecture, context, corruption and optional D-only local auxiliary losses are explicit
config changes. Mismatched-history ranking changes the point head negative examples when enabled. Compare measured cost too.
Optional G repair trains a stateless read adapter. Local stability adds two parallel
one-step feedback branches per enabled phase, with detached prefix anchors and particles.
These local branches do not feed into another generated prediction; writer updates remain D-only.
No seed sweeps. B-cap defaults unchanged. Prior regularization applied once per update.

Continuous fidelity, interventions, state statistics, and diversity: [results.json](results.json).

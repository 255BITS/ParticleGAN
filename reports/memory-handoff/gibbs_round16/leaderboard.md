# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| joint_match_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| read_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| read_w01 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| read_w10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| state_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| state_match_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| joint_g10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| state_w10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| uncond_w10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.0216 | 0.0110 / 0.0111 | 0.0085 / 0.0097 | 0.5% / 0.5% | 0.070 / 0.074 |
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| uncond_w10 | 0.0262 | 0.0093 / 0.0094 | 0.0088 / 0.0083 | 0.4% / 0.4% | 0.045 / 0.041 |
| read_w01 | 0.0232 | 0.0096 / 0.0092 | 0.0086 / 0.0085 | 0.4% / 0.4% | 0.061 / 0.054 |
| joint_match_g10 | 0.0282 | 0.0090 / 0.0090 | 0.0083 / 0.0088 | 0.4% / 0.4% | 0.054 / 0.052 |
| joint_g10 | 0.0223 | 0.0086 / 0.0091 | 0.0082 / 0.0072 | 0.4% / 0.4% | 0.055 / 0.064 |
| state_match_g10 | 0.0266 | 0.0088 / 0.0083 | 0.0086 / 0.0079 | 0.4% / 0.3% | 0.058 / 0.056 |
| state_g10 | 0.0308 | 0.0074 / 0.0075 | 0.0064 / 0.0070 | 0.3% / 0.3% | 0.063 / 0.060 |
| read_g10 | 0.0317 | 0.0071 / 0.0069 | 0.0065 / 0.0062 | 0.3% / 0.3% | 0.046 / 0.045 |
| read_w10 | 0.0397 | 0.0050 / 0.0049 | 0.0046 / 0.0043 | 0.2% / 0.2% | 0.023 / 0.022 |
| state_w10 | 0.0367 | 0.0044 / 0.0045 | 0.0047 / 0.0041 | 0.1% / 0.1% | 0.014 / 0.015 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| joint_match_g10 | 8 | 0.855 | 0.273 | 49.1% | 0.089 | 1.375 / 1.781 |
| joint_match_g10 | 32 | 0.844 | 0.272 | 49.5% | 0.091 | 1.259 / 1.781 |
| read_g10 | 8 | 0.977 | 0.270 | 50.5% | 0.091 | 1.346 / 2.055 |
| read_g10 | 32 | 0.972 | 0.271 | 50.1% | 0.095 | 1.444 / 2.025 |
| read_w01 | 8 | 0.789 | 0.270 | 50.8% | 0.084 | 1.223 / 1.578 |
| read_w01 | 32 | 0.801 | 0.273 | 50.5% | 0.085 | 1.283 / 1.675 |
| read_w10 | 8 | 1.413 | 0.271 | 50.6% | 0.117 | 2.517 / 2.202 |
| read_w10 | 32 | 1.410 | 0.271 | 50.5% | 0.136 | 2.324 / 2.523 |
| state_g10 | 8 | 1.182 | 0.270 | 50.2% | 0.082 | 1.365 / 2.073 |
| state_g10 | 32 | 1.173 | 0.270 | 50.2% | 0.085 | 1.283 / 2.188 |
| state_match_g10 | 8 | 0.910 | 0.272 | 50.4% | 0.087 | 1.224 / 1.818 |
| state_match_g10 | 32 | 0.909 | 0.275 | 50.0% | 0.088 | 1.318 / 1.805 |
| joint_g10 | 8 | 0.775 | 0.266 | 51.4% | 0.081 | 1.196 / 1.554 |
| joint_g10 | 32 | 0.770 | 0.266 | 51.6% | 0.083 | 1.235 / 1.626 |
| state_w10 | 8 | 0.937 | 0.271 | 50.1% | 0.151 | 1.855 / 1.710 |
| state_w10 | 32 | 0.934 | 0.271 | 50.0% | 0.151 | 1.963 / 1.833 |
| uncond_w10 | 8 | 0.757 | 0.270 | 50.1% | 0.120 | 1.392 / 1.710 |
| uncond_w10 | 32 | 0.759 | 0.270 | 50.2% | 0.107 | 1.429 / 1.759 |
| match_shuffle25 | 8 | 0.960 | 0.261 | 51.9% | 0.085 | 1.242 / 1.869 |
| match_shuffle25 | 32 | 0.945 | 0.261 | 51.8% | 0.088 | 1.237 / 1.909 |
| match_shuffle25_5k | 8 | 0.916 | 0.262 | 52.5% | 0.080 | 1.053 / 1.832 |
| match_shuffle25_5k | 32 | 0.912 | 0.260 | 52.8% | 0.082 | 0.986 / 1.789 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint_match_g10 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1435 |
| read_g10 | 512 | 63 | 32 | 0 / 0 | 4 / 7 | 0.5 / 0.25 | 0.1438 |
| read_w01 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1533 |
| read_w10 | 512 | 63 | 32 | 0 / 0 | 7 / 7 | 0.5 / 0.25 | 0.1532 |
| state_g10 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1378 |
| state_match_g10 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1408 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| joint_match_g10 | mixed | proposal | True | 8 / 10 | True |
| read_g10 | mixed | proposal | True | 8 / 14 | True |
| read_w01 | mixed | proposal | True | 14 / 14 | True |
| read_w10 | mixed | proposal | True | 14 / 14 | True |
| state_g10 | mixed | proposal | True | 8 / 10 | True |
| state_match_g10 | mixed | proposal | True | 8 / 10 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| joint_match_g10 | 0.5 | fixed / 0.25 | 0.25 |
| read_g10 | 0.5 | fixed / 0.25 | 0.25 |
| read_w01 | 0.5 | fixed / 0.25 | 0.25 |
| read_w10 | 0.5 | fixed / 0.25 | 0.25 |
| state_g10 | 0.5 | fixed / 0.25 | 0.25 |
| state_match_g10 | 0.5 | fixed / 0.25 | 0.25 |

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
| joint_match_g10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| read_g10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| read_w01 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| read_w10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| state_g10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| state_match_g10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## Local history and future objectives

| Run | Mismatch weight / donors | Mismatch context / strength / writer gradient | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---|---:|
| joint_match_g10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| read_g10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| read_w01 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| read_w10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| state_g10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| state_match_g10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |

Mismatch training ranks real continuations from other histories with the existing point head;
its D loss and default B-cap are normalized by 1+weight. G point loss stays unchanged.
Mismatch context can be clean, one generated replacement write, or an equal loss mixture.
The writer-gradient control detaches only mismatch context; existing writer losses remain active.
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

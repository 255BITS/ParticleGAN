# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| plain_mixed_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| plain_pair50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_uniform50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_fixed4375 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_mild_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_shared25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_shared75 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_uniform50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_uniform_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_pair50 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_shared_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| shared_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.0166 | 0.0082 / 0.0083 | 0.0067 / 0.0068 | 0.4% / 0.4% | 0.054 / 0.059 |
| proposal_pair25 | 0.0185 | 0.0071 / 0.0067 | 0.0052 / 0.0053 | 0.3% / 0.3% | 0.056 / 0.061 |
| proposal_mixed_uniform50 | 0.0207 | 0.0066 / 0.0072 | 0.0055 / 0.0061 | 0.3% / 0.3% | 0.053 / 0.055 |
| shared_s25 | 0.0172 | 0.0065 / 0.0066 | 0.0052 / 0.0059 | 0.3% / 0.4% | 0.052 / 0.049 |
| proposal_mixed_pair50 | 0.0140 | 0.0065 / 0.0064 | 0.0034 / 0.0039 | 0.4% / 0.4% | 0.063 / 0.062 |
| proposal_clean_s25 | 0.0245 | 0.0062 / 0.0064 | 0.0058 / 0.0061 | 0.3% / 0.3% | 0.054 / 0.054 |
| plain_pair50 | 0.0211 | 0.0059 / 0.0064 | 0.0044 / 0.0044 | 0.3% / 0.4% | 0.043 / 0.057 |
| proposal_pair50 | 0.0149 | 0.0059 / 0.0061 | 0.0036 / 0.0037 | 0.3% / 0.4% | 0.055 / 0.064 |
| proposal_mixed_fixed4375 | 0.0197 | 0.0057 / 0.0058 | 0.0037 / 0.0048 | 0.3% / 0.3% | 0.051 / 0.049 |
| proposal_mixed_uniform_pair25 | 0.0414 | 0.0054 / 0.0055 | 0.0046 / 0.0046 | 0.2% / 0.2% | 0.037 / 0.039 |
| plain_mixed_s25 | 0.0115 | 0.0047 / 0.0048 | 0.0035 / 0.0040 | 0.2% / 0.2% | 0.037 / 0.034 |
| proposal_mixed_shared25 | 0.0283 | 0.0044 / 0.0046 | 0.0034 / 0.0040 | 0.2% / 0.2% | 0.038 / 0.039 |
| clock_control | 0.0126 | 0.0040 / 0.0043 | 0.0028 / 0.0030 | 0.2% / 0.2% | 0.035 / 0.040 |
| proposal_mixed_s25 | 0.0170 | 0.0037 / 0.0037 | 0.0027 / 0.0033 | 0.2% / 0.2% | 0.032 / 0.029 |
| proposal_control | 0.0261 | 0.0039 / 0.0037 | 0.0031 / 0.0030 | 0.2% / 0.2% | 0.033 / 0.036 |
| proposal_clean_uniform50 | 0.0114 | 0.0038 / 0.0034 | 0.0022 / 0.0023 | 0.2% / 0.2% | 0.043 / 0.040 |
| proposal_mixed_shared75 | 0.0227 | 0.0026 / 0.0027 | 0.0017 / 0.0018 | 0.1% / 0.1% | 0.025 / 0.029 |
| proposal_mixed_mild_full | 0.0219 | 0.0022 / 0.0026 | 0.0014 / 0.0020 | 0.1% / 0.1% | 0.031 / 0.031 |
| proposal_shared_s25 | 0.0271 | 0.0022 / 0.0026 | 0.0017 / 0.0019 | 0.1% / 0.1% | 0.020 / 0.028 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| plain_mixed_s25 | 8 | 2.021 | 0.268 | 50.4% | 0.103 | 1.548 / 3.200 |
| plain_mixed_s25 | 32 | 2.026 | 0.269 | 50.4% | 0.111 | 1.493 / 3.146 |
| plain_pair50 | 8 | 1.690 | 0.266 | 51.1% | 0.094 | 1.733 / 2.698 |
| plain_pair50 | 32 | 1.679 | 0.265 | 51.0% | 0.090 | 1.381 / 2.777 |
| proposal_clean_s25 | 8 | 1.497 | 0.267 | 50.9% | 0.096 | 1.572 / 2.543 |
| proposal_clean_s25 | 32 | 1.487 | 0.268 | 50.7% | 0.096 | 1.448 / 2.573 |
| proposal_clean_uniform50 | 8 | 2.222 | 0.269 | 50.7% | 0.100 | 1.528 / 3.460 |
| proposal_clean_uniform50 | 32 | 2.241 | 0.270 | 50.2% | 0.098 | 1.430 / 3.375 |
| proposal_mixed_fixed4375 | 8 | 1.461 | 0.266 | 51.1% | 0.096 | 1.307 / 2.645 |
| proposal_mixed_fixed4375 | 32 | 1.460 | 0.267 | 50.9% | 0.092 | 1.257 / 2.597 |
| proposal_mixed_mild_full | 8 | 2.164 | 0.270 | 50.6% | 0.109 | 1.777 / 3.288 |
| proposal_mixed_mild_full | 32 | 2.152 | 0.269 | 50.7% | 0.108 | 1.651 / 3.226 |
| proposal_mixed_pair25 | 8 | 1.136 | 0.270 | 50.0% | 0.099 | 1.704 / 2.018 |
| proposal_mixed_pair25 | 32 | 1.159 | 0.264 | 50.7% | 0.095 | 1.528 / 2.158 |
| proposal_mixed_pair50 | 8 | 2.151 | 0.266 | 51.2% | 0.081 | 1.398 / 3.291 |
| proposal_mixed_pair50 | 32 | 2.158 | 0.268 | 50.9% | 0.083 | 1.358 / 3.354 |
| proposal_mixed_s25 | 8 | 1.870 | 0.271 | 50.2% | 0.117 | 1.848 / 3.150 |
| proposal_mixed_s25 | 32 | 1.871 | 0.269 | 50.5% | 0.116 | 1.830 / 3.133 |
| proposal_mixed_shared25 | 8 | 1.773 | 0.264 | 51.1% | 0.108 | 1.839 / 2.949 |
| proposal_mixed_shared25 | 32 | 1.777 | 0.265 | 51.1% | 0.106 | 1.603 / 2.995 |
| proposal_mixed_shared75 | 8 | 1.627 | 0.268 | 50.7% | 0.109 | 1.826 / 2.822 |
| proposal_mixed_shared75 | 32 | 1.638 | 0.267 | 51.2% | 0.113 | 1.641 / 2.812 |
| proposal_mixed_uniform50 | 8 | 1.356 | 0.273 | 50.3% | 0.102 | 1.459 / 2.491 |
| proposal_mixed_uniform50 | 32 | 1.355 | 0.270 | 50.6% | 0.099 | 1.405 / 2.481 |
| proposal_mixed_uniform_pair25 | 8 | 1.433 | 0.278 | 49.1% | 0.104 | 1.841 / 2.462 |
| proposal_mixed_uniform_pair25 | 32 | 1.424 | 0.278 | 49.0% | 0.105 | 1.759 / 2.546 |
| proposal_pair25 | 8 | 1.608 | 0.268 | 50.1% | 0.090 | 1.581 / 2.727 |
| proposal_pair25 | 32 | 1.636 | 0.270 | 50.3% | 0.090 | 1.543 / 2.725 |
| proposal_pair50 | 8 | 2.038 | 0.266 | 51.3% | 0.083 | 1.440 / 3.086 |
| proposal_pair50 | 32 | 2.041 | 0.267 | 51.5% | 0.083 | 1.416 / 3.130 |
| proposal_shared_s25 | 8 | 1.581 | 0.272 | 50.2% | 0.120 | 1.659 / 2.649 |
| proposal_shared_s25 | 32 | 1.577 | 0.271 | 50.2% | 0.105 | 1.650 / 2.596 |
| clock_control | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_control | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| proposal_control | 8 | 1.675 | 0.268 | 50.8% | 0.108 | 1.412 / 2.815 |
| proposal_control | 32 | 1.675 | 0.270 | 50.3% | 0.105 | 1.456 / 2.784 |
| shared_s25 | 8 | 1.727 | 0.270 | 50.6% | 0.102 | 1.425 / 2.847 |
| shared_s25 | 32 | 1.726 | 0.270 | 50.5% | 0.104 | 1.365 / 2.702 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_mixed_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0442 |
| plain_pair50 | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0793 |
| proposal_clean_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0426 |
| proposal_clean_uniform50 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.5 | 0.0429 |
| proposal_mixed_fixed4375 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.4375 | 0.0468 |
| proposal_mixed_mild_full | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0476 |
| proposal_mixed_pair25 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.0878 |
| proposal_mixed_pair50 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.0882 |
| proposal_mixed_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0467 |
| proposal_mixed_shared25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0474 |
| proposal_mixed_shared75 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0462 |
| proposal_mixed_uniform50 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.5 | 0.0470 |
| proposal_mixed_uniform_pair25 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.5 | 0.0860 |
| proposal_pair25 | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0818 |
| proposal_pair50 | 512 | 63 | 32 | 0 / 0 | 3 / 3 | 0 / 1 | 0.0798 |
| proposal_shared_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0430 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Auxiliaries disabled |
|---|---|---|---|---:|---|
| plain_mixed_s25 | mixed | none | True | 2 / 2 | True |
| plain_pair50 | shared | none | False | 3 / 3 | True |
| proposal_clean_s25 | clean | proposal | True | 4 / 4 | True |
| proposal_clean_uniform50 | clean | proposal | True | 4 / 4 | True |
| proposal_mixed_fixed4375 | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_mild_full | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_pair25 | mixed | proposal | True | 8 / 8 | True |
| proposal_mixed_pair50 | mixed | proposal | True | 8 / 8 | True |
| proposal_mixed_s25 | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_shared25 | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_shared75 | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_uniform50 | mixed | proposal | True | 4 / 4 | True |
| proposal_mixed_uniform_pair25 | mixed | proposal | True | 8 / 8 | True |
| proposal_pair25 | shared | proposal | False | 6 / 6 | True |
| proposal_pair50 | shared | proposal | False | 6 / 6 | True |
| proposal_shared_s25 | shared | proposal | True | 4 / 4 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| plain_mixed_s25 | 0.5 | fixed / 0.25 | 0 |
| plain_pair50 | 1 | fixed / 1 | 0.5 |
| proposal_clean_s25 | 0 | fixed / 0.25 | 0 |
| proposal_clean_uniform50 | 0 | uniform / 0.5 | 0 |
| proposal_mixed_fixed4375 | 0.5 | fixed / 0.4375 | 0 |
| proposal_mixed_mild_full | 0.5 | mild_full / 0.25 | 0 |
| proposal_mixed_pair25 | 0.5 | fixed / 0.25 | 0.25 |
| proposal_mixed_pair50 | 0.5 | fixed / 0.25 | 0.5 |
| proposal_mixed_s25 | 0.5 | fixed / 0.25 | 0 |
| proposal_mixed_shared25 | 0.25 | fixed / 0.25 | 0 |
| proposal_mixed_shared75 | 0.75 | fixed / 0.25 | 0 |
| proposal_mixed_uniform50 | 0.5 | uniform / 0.5 | 0 |
| proposal_mixed_uniform_pair25 | 0.5 | uniform / 0.5 | 0.25 |
| proposal_pair25 | 1 | fixed / 1 | 0.25 |
| proposal_pair50 | 1 | fixed / 1 | 0.5 |
| proposal_shared_s25 | 1 | fixed / 0.25 | 0 |

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
| plain_mixed_s25 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| plain_pair50 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_clean_s25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_clean_uniform50 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_fixed4375 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_mild_full | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_pair25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_pair50 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_s25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_shared25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_shared75 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_uniform50 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_mixed_uniform_pair25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_pair25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_pair50 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_shared_s25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

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

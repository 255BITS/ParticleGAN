# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| explored_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| explored_mild | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| mixed_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| mixed_mild | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| mixed_mild_headonly | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.0216 | 0.0110 / 0.0111 | 0.0085 / 0.0097 | 0.5% / 0.5% | 0.070 / 0.074 |
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| explored_mild | 0.0347 | 0.0090 / 0.0093 | 0.0085 / 0.0072 | 0.4% / 0.4% | 0.062 / 0.059 |
| mixed_mild | 0.0245 | 0.0088 / 0.0090 | 0.0080 / 0.0083 | 0.4% / 0.4% | 0.053 / 0.066 |
| mixed_full | 0.0255 | 0.0086 / 0.0089 | 0.0074 / 0.0071 | 0.4% / 0.4% | 0.060 / 0.067 |
| explored_full | 0.0158 | 0.0073 / 0.0074 | 0.0059 / 0.0060 | 0.3% / 0.3% | 0.049 / 0.051 |
| mixed_mild_headonly | 0.0235 | 0.0061 / 0.0063 | 0.0051 / 0.0047 | 0.3% / 0.3% | 0.041 / 0.044 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| explored_full | 8 | 1.099 | 0.265 | 53.5% | 0.083 | 1.455 / 2.000 |
| explored_full | 32 | 1.105 | 0.259 | 54.8% | 0.087 | 1.350 / 2.101 |
| explored_mild | 8 | 0.884 | 0.274 | 50.1% | 0.087 | 1.353 / 1.697 |
| explored_mild | 32 | 0.888 | 0.274 | 50.0% | 0.089 | 1.285 / 1.839 |
| mixed_full | 8 | 0.913 | 0.263 | 51.4% | 0.083 | 1.186 / 1.754 |
| mixed_full | 32 | 0.913 | 0.261 | 51.9% | 0.086 | 1.198 / 1.826 |
| mixed_mild | 8 | 0.896 | 0.268 | 50.9% | 0.086 | 1.149 / 1.788 |
| mixed_mild | 32 | 0.891 | 0.265 | 51.1% | 0.087 | 1.194 / 1.816 |
| mixed_mild_headonly | 8 | 1.785 | 0.268 | 50.7% | 0.094 | 1.663 / 2.482 |
| mixed_mild_headonly | 32 | 1.772 | 0.268 | 50.8% | 0.097 | 1.427 / 2.678 |
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
| explored_full | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1199 |
| explored_mild | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1219 |
| mixed_full | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1252 |
| mixed_mild | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1231 |
| mixed_mild_headonly | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1018 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| explored_full | mixed | proposal | True | 10 / 8 | True |
| explored_mild | mixed | proposal | True | 10 / 8 | True |
| mixed_full | mixed | proposal | True | 10 / 8 | True |
| mixed_mild | mixed | proposal | True | 10 / 8 | True |
| mixed_mild_headonly | mixed | proposal | True | 10 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| explored_full | 0.5 | fixed / 0.25 | 0.25 |
| explored_mild | 0.5 | fixed / 0.25 | 0.25 |
| mixed_full | 0.5 | fixed / 0.25 | 0.25 |
| mixed_mild | 0.5 | fixed / 0.25 | 0.25 |
| mixed_mild_headonly | 0.5 | fixed / 0.25 | 0.25 |

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
| explored_full | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| explored_mild | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| mixed_full | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| mixed_mild | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| mixed_mild_headonly | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## Local history and future objectives

| Run | Mismatch weight / donors | Mismatch context / strength / writer gradient | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---|---:|
| explored_full | 0.25 / shuffle | explored / 1 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| explored_mild | 0.25 / shuffle | explored / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| mixed_full | 0.25 / shuffle | mixed / 1 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| mixed_mild | 0.25 / shuffle | mixed / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| mixed_mild_headonly | 0.25 / shuffle | mixed / 0.25 / False | 0 / 0.5 | 0 / [0, 4, 12] | 0 |

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

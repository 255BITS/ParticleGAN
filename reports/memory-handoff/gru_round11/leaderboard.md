# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| gru16 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gru16_no_d | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gru16_no_repair | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gru16_read_d | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gru8 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_mixed_pair25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.0166 | 0.0082 / 0.0083 | 0.0067 / 0.0068 | 0.4% / 0.4% | 0.054 / 0.059 |
| gru8 | 0.0237 | 0.0070 / 0.0072 | 0.0044 / 0.0047 | 0.4% / 0.4% | 0.059 / 0.061 |
| gru16_no_d | 0.0179 | 0.0062 / 0.0062 | 0.0059 / 0.0057 | 0.3% / 0.3% | 0.039 / 0.036 |
| gru16 | 0.0190 | 0.0052 / 0.0052 | 0.0027 / 0.0032 | 0.3% / 0.3% | 0.056 / 0.051 |
| gru16_read_d | 0.0109 | 0.0044 / 0.0046 | 0.0023 / 0.0025 | 0.3% / 0.3% | 0.058 / 0.064 |
| gru16_no_repair | 0.0091 | 0.0015 / 0.0017 | 0.0001 / 0.0001 | 0.1% / 0.1% | 0.034 / 0.039 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| gru16 | 8 | 2.342 | 0.261 | 52.2% | 0.089 | 1.641 / 3.625 |
| gru16 | 32 | 2.346 | 0.261 | 52.1% | 0.084 | 1.611 / 3.639 |
| gru16_no_d | 8 | 1.853 | 0.265 | 50.9% | 0.113 | 1.818 / 2.882 |
| gru16_no_d | 32 | 1.881 | 0.265 | 50.9% | 0.131 | 1.774 / 2.951 |
| gru16_no_repair | 8 | 3.091 | 0.268 | 51.1% | 0.090 | 1.963 / 4.291 |
| gru16_no_repair | 32 | 3.081 | 0.269 | 50.9% | 0.093 | 1.883 / 4.364 |
| gru16_read_d | 8 | 2.563 | 0.262 | 51.7% | 0.081 | 1.425 / 3.788 |
| gru16_read_d | 32 | 2.568 | 0.263 | 51.8% | 0.077 | 1.259 / 3.713 |
| gru8 | 8 | 1.723 | 0.264 | 51.2% | 0.083 | 1.512 / 2.902 |
| gru8 | 32 | 1.719 | 0.264 | 51.4% | 0.085 | 1.316 / 2.894 |
| proposal_mixed_pair25 | 8 | 1.136 | 0.270 | 50.0% | 0.099 | 1.704 / 2.018 |
| proposal_mixed_pair25 | 32 | 1.159 | 0.264 | 50.7% | 0.095 | 1.528 / 2.158 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru16 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1173 |
| gru16_no_d | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1192 |
| gru16_no_repair | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1158 |
| gru16_read_d | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1218 |
| gru8 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1205 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Auxiliaries disabled |
|---|---|---|---|---:|---|
| gru16 | mixed | proposal | True | 8 / 8 | True |
| gru16_no_d | mixed | proposal | True | 8 / 8 | True |
| gru16_no_repair | mixed | none | True | 4 / 4 | True |
| gru16_read_d | mixed | proposal | True | 8 / 8 | True |
| gru8 | mixed | proposal | True | 8 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| gru16 | 0.5 | fixed / 0.25 | 0.25 |
| gru16_no_d | 0.5 | fixed / 0.25 | 0.25 |
| gru16_no_repair | 0.5 | fixed / 0.25 | 0.25 |
| gru16_read_d | 0.5 | fixed / 0.25 | 0.25 |
| gru8 | 0.5 | fixed / 0.25 | 0.25 |

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
| gru16 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gru16_no_d | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gru16_no_repair | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gru16_read_d | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gru8 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## G-owned observation recurrence

| Run | G state size | State update reads D | G reads D | Prefix GRU updates / phase |
|---|---:|---|---|---:|
| gru16 | 16 | False | True | 126 |
| gru16_no_d | 16 | False | False | 126 |
| gru16_no_repair | 16 | False | True | 126 |
| gru16_read_d | 16 | True | True | 126 |
| gru8 | 8 | False | True | 126 |

G state encodes real observations with full real-prefix BPTT, then at most one generated write.
State advances once per observation; proposal/final reads share the same state.
D owns M, G owns S. Both start at zero for cold evaluation. G has no MSE objective.
Memory access controls are separately trained; interventions alone do not establish comparative benefit.

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

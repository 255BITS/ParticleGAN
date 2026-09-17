# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| gibbs1_arch | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gibbs1_joint10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gibbs1_joint25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gibbs3_arch | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gibbs3_joint10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| gibbs3_joint25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.0216 | 0.0110 / 0.0111 | 0.0085 / 0.0097 | 0.5% / 0.5% | 0.070 / 0.074 |
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| gibbs1_joint25 | 0.0430 | 0.0072 / 0.0072 | 0.0061 / 0.0058 | 0.3% / 0.3% | 0.050 / 0.045 |
| gibbs1_arch | 0.0249 | 0.0059 / 0.0060 | 0.0051 / 0.0052 | 0.3% / 0.3% | 0.049 / 0.050 |
| gibbs1_joint10 | 0.0253 | 0.0056 / 0.0058 | 0.0046 / 0.0048 | 0.2% / 0.3% | 0.048 / 0.046 |
| gibbs3_joint25 | 0.0167 | 0.0051 / 0.0052 | 0.0038 / 0.0046 | 0.2% / 0.2% | 0.032 / 0.036 |
| gibbs3_arch | 0.0169 | 0.0044 / 0.0045 | 0.0037 / 0.0039 | 0.2% / 0.2% | 0.031 / 0.038 |
| gibbs3_joint10 | 0.0127 | 0.0035 / 0.0037 | 0.0029 / 0.0032 | 0.2% / 0.2% | 0.031 / 0.034 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| gibbs1_arch | 8 | 1.188 | 0.270 | 50.9% | 0.081 | 1.454 / 2.158 |
| gibbs1_arch | 32 | 1.182 | 0.271 | 50.3% | 0.086 | 1.399 / 2.266 |
| gibbs1_joint10 | 8 | 1.198 | 0.267 | 51.1% | 0.090 | 1.488 / 2.168 |
| gibbs1_joint10 | 32 | 1.201 | 0.266 | 51.1% | 0.085 | 1.482 / 2.300 |
| gibbs1_joint25 | 8 | 0.893 | 0.265 | 51.7% | 0.083 | 1.437 / 1.793 |
| gibbs1_joint25 | 32 | 0.891 | 0.265 | 51.6% | 0.086 | 1.408 / 1.915 |
| gibbs3_arch | 8 | 2.031 | 0.268 | 50.7% | 0.091 | 1.788 / 3.023 |
| gibbs3_arch | 32 | 2.027 | 0.268 | 50.7% | 0.091 | 1.780 / 3.068 |
| gibbs3_joint10 | 8 | 2.132 | 0.272 | 50.1% | 0.095 | 1.651 / 3.190 |
| gibbs3_joint10 | 32 | 2.129 | 0.271 | 50.2% | 0.088 | 1.724 / 3.269 |
| gibbs3_joint25 | 8 | 1.969 | 0.267 | 50.9% | 0.098 | 1.333 / 3.158 |
| gibbs3_joint25 | 32 | 1.969 | 0.267 | 50.9% | 0.092 | 1.431 / 3.176 |
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
| gibbs1_arch | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1182 |
| gibbs1_joint10 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1415 |
| gibbs1_joint25 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1399 |
| gibbs3_arch | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1302 |
| gibbs3_joint10 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1518 |
| gibbs3_joint25 | 512 | 63 | 32 | 0 / 0 | 4 / 5 | 0.5 / 0.25 | 0.1535 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| gibbs1_arch | mixed | proposal | True | 8 / 8 | True |
| gibbs1_joint10 | mixed | proposal | True | 8 / 10 | True |
| gibbs1_joint25 | mixed | proposal | True | 8 / 10 | True |
| gibbs3_arch | mixed | proposal | True | 24 / 24 | True |
| gibbs3_joint10 | mixed | proposal | True | 24 / 30 | True |
| gibbs3_joint25 | mixed | proposal | True | 24 / 30 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| gibbs1_arch | 0.5 | fixed / 0.25 | 0.25 |
| gibbs1_joint10 | 0.5 | fixed / 0.25 | 0.25 |
| gibbs1_joint25 | 0.5 | fixed / 0.25 | 0.25 |
| gibbs3_arch | 0.5 | fixed / 0.25 | 0.25 |
| gibbs3_joint10 | 0.5 | fixed / 0.25 | 0.25 |
| gibbs3_joint25 | 0.5 | fixed / 0.25 | 0.25 |

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
| gibbs1_arch | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gibbs1_joint10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gibbs1_joint25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gibbs3_arch | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gibbs3_joint10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| gibbs3_joint25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## Local history and future objectives

| Run | Mismatch weight / donors | Mismatch context / strength / writer gradient | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---|---:|
| gibbs1_arch | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| gibbs1_joint10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| gibbs1_joint25 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| gibbs3_arch | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| gibbs3_joint10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| gibbs3_joint25 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |

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

# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| future_clean10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future_full10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future_mixed10 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future_mixed10_detachwrite | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| future_mixed25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.0216 | 0.0110 / 0.0111 | 0.0085 / 0.0097 | 0.5% / 0.5% | 0.070 / 0.074 |
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| future_full10 | 0.0239 | 0.0100 / 0.0100 | 0.0092 / 0.0088 | 0.4% / 0.5% | 0.060 / 0.064 |
| future_mixed25 | 0.0114 | 0.0084 / 0.0081 | 0.0065 / 0.0066 | 0.4% / 0.4% | 0.052 / 0.056 |
| future_mixed10 | 0.0252 | 0.0076 / 0.0074 | 0.0069 / 0.0065 | 0.4% / 0.4% | 0.054 / 0.056 |
| future_mixed10_detachwrite | 0.0291 | 0.0068 / 0.0066 | 0.0056 / 0.0052 | 0.3% / 0.3% | 0.047 / 0.048 |
| future_clean10 | 0.0260 | 0.0065 / 0.0066 | 0.0058 / 0.0058 | 0.3% / 0.3% | 0.058 / 0.051 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| future_clean10 | 8 | 1.322 | 0.265 | 51.6% | 0.083 | 1.815 / 2.183 |
| future_clean10 | 32 | 1.333 | 0.267 | 51.2% | 0.087 | 1.781 / 2.337 |
| future_full10 | 8 | 0.915 | 0.269 | 51.0% | 0.085 | 1.269 / 1.857 |
| future_full10 | 32 | 0.909 | 0.272 | 50.2% | 0.083 | 1.236 / 1.880 |
| future_mixed10 | 8 | 1.061 | 0.269 | 50.8% | 0.089 | 1.411 / 1.945 |
| future_mixed10 | 32 | 1.071 | 0.268 | 50.8% | 0.087 | 1.578 / 2.050 |
| future_mixed10_detachwrite | 8 | 1.108 | 0.271 | 50.2% | 0.087 | 1.738 / 1.719 |
| future_mixed10_detachwrite | 32 | 1.092 | 0.270 | 50.3% | 0.089 | 1.756 / 1.952 |
| future_mixed25 | 8 | 1.078 | 0.309 | 49.6% | 0.087 | 1.484 / 2.045 |
| future_mixed25 | 32 | 1.063 | 0.317 | 50.4% | 0.082 | 1.538 / 2.029 |
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
| future_clean10 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1514 |
| future_full10 | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1627 |
| future_mixed10 | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1638 |
| future_mixed10_detachwrite | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1599 |
| future_mixed25 | 512 | 63 | 32 | 0 / 0 | 5 / 4 | 0.5 / 0.25 | 0.1668 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| future_clean10 | mixed | proposal | True | 8 / 8 | True |
| future_full10 | mixed | proposal | True | 10 / 8 | True |
| future_mixed10 | mixed | proposal | True | 10 / 8 | True |
| future_mixed10_detachwrite | mixed | proposal | True | 10 / 8 | True |
| future_mixed25 | mixed | proposal | True | 10 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| future_clean10 | 0.5 | fixed / 0.25 | 0.25 |
| future_full10 | 0.5 | fixed / 0.25 | 0.25 |
| future_mixed10 | 0.5 | fixed / 0.25 | 0.25 |
| future_mixed10_detachwrite | 0.5 | fixed / 0.25 | 0.25 |
| future_mixed25 | 0.5 | fixed / 0.25 | 0.25 |

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
| future_clean10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future_full10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future_mixed10 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future_mixed10_detachwrite | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| future_mixed25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## Local history and future objectives

| Run | Mismatch weight / donors | Mismatch context / strength / writer gradient | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---|---:|
| future_clean10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| future_full10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| future_mixed10 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| future_mixed10_detachwrite | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| future_mixed25 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |

Mismatch training ranks real continuations from other histories with the existing point head;
its D loss and default B-cap are normalized by 1+weight. G point loss stays unchanged.
Mismatch context can be clean, one generated replacement write, or an equal loss mixture.
The writer-gradient control detaches only mismatch context; existing writer losses remain active.
Recovery perturbs prefix observations read by G; the pair judge retains the clean reference.
Future queries read identical prefix memory independently, with fixed z and explicit offsets.
Their joint GAN uses real future targets, no generated writes, and default exact B-cap.
Future weight convexly mixes this branch with the existing GAN; prior regularization stays once.

## Future history ranking through the point head

| Run | Added D weight | Offsets | Context / write strength | Explored writer gradient |
|---|---:|---|---|---|
| future_clean10 | 0.1 | [4, 12] | clean / 0.25 | True |
| future_full10 | 0.1 | [4, 12] | mixed / 1 | True |
| future_mixed10 | 0.1 | [4, 12] | mixed / 0.25 | True |
| future_mixed10_detachwrite | 0.1 | [4, 12] | mixed / 0.25 | False |
| future_mixed25 | 0.25 | [4, 12] | mixed / 0.25 | True |

D ranks same-episode future observations against other histories at explicit offsets.
The horizon projection enters the shared point head; G always uses offset zero.
Future losses and default B-cap average across offsets and contexts, then normalize
with the existing D objective by 1+weight. G loss is unchanged.
Mixed contexts average clean and one-generated-write losses. The gradient control
detaches only the explored future context, retaining clean-future writer gradients.
This is distinct from direct future G queries; no extra generated trajectory is trained.

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

# Local memory handoff scouts (no full-rollout training)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_dclock | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| embedded8 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| embedded8_dclock | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| hybrid8 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| hybrid8_dclock | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| intent8 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| intent8_dclock | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| match_shuffle25_5k | 5000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuous orbit progress (evaluation only)

Warm orbit quality: mean 1/((1+(relative radial error/.1)^2)*(1+(per-step signed angular error/.03)^2)), in [0,1]. Good steps require abs radial error<.1 and abs signed angular error<.03. Include the handoff transition from the last clean reference point. Good-arc turns are the longest consecutive good angular arc / (2*pi). Quality is a diagnostic, not a success probability, and does not measure absolute phase. Cold uses a single circle fit and signed speed from the first32 generated points, then holds them fixed; radius outside [.5,1.6] and speed outside [.08,.45] are penalized. Warm uses the true reference orbit. Report early and late scores separately.

Sorted by worst-prefix warm quality; full-circle/original-orbit passes remain primary.

| Run | Cold self-fit quality | Warm quality 8 / 32 | Late quality 8 / 32 | Good steps 8 / 32 | Longest good arc (turns) 8 / 32 |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.0216 | 0.0110 / 0.0111 | 0.0085 / 0.0097 | 0.5% / 0.5% | 0.070 / 0.074 |
| match_shuffle25 | 0.0230 | 0.0109 / 0.0112 | 0.0105 / 0.0109 | 0.5% / 0.5% | 0.064 / 0.064 |
| embedded8_dclock | 0.0196 | 0.0076 / 0.0076 | 0.0069 / 0.0074 | 0.4% / 0.4% | 0.046 / 0.043 |
| baseline_dclock | 0.0246 | 0.0062 / 0.0058 | 0.0057 / 0.0051 | 0.3% / 0.3% | 0.048 / 0.036 |
| embedded8 | 0.0218 | 0.0054 / 0.0051 | 0.0040 / 0.0044 | 0.2% / 0.2% | 0.034 / 0.032 |
| hybrid8_dclock | 0.0314 | 0.0015 / 0.0015 | 0.0012 / 0.0009 | 0.1% / 0.1% | 0.018 / 0.019 |
| intent8_dclock | 0.0367 | 0.0011 / 0.0010 | 0.0005 / 0.0005 | 0.1% / 0.1% | 0.018 / 0.016 |
| intent8 | 0.0176 | 0.0010 / 0.0010 | 0.0005 / 0.0005 | 0.1% / 0.0% | 0.016 / 0.012 |
| hybrid8 | 0.0444 | 0.0010 / 0.0011 | 0.0006 / 0.0005 | 0.0% / 0.1% | 0.015 / 0.016 |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_dclock | 8 | 1.639 | 0.270 | 50.6% | 0.083 | 1.763 / 2.622 |
| baseline_dclock | 32 | 1.639 | 0.272 | 50.3% | 0.105 | 1.781 / 2.739 |
| embedded8 | 8 | 1.572 | 0.267 | 50.8% | 0.110 | 1.348 / 2.458 |
| embedded8 | 32 | 1.577 | 0.267 | 50.6% | 0.109 | 1.379 / 2.304 |
| embedded8_dclock | 8 | 1.429 | 0.264 | 52.3% | 0.098 | 1.354 / 2.343 |
| embedded8_dclock | 32 | 1.429 | 0.264 | 52.4% | 0.101 | 1.333 / 2.068 |
| hybrid8 | 8 | 2.612 | 0.272 | 49.6% | 0.110 | 2.675 / 3.672 |
| hybrid8 | 32 | 2.612 | 0.272 | 49.7% | 0.108 | 2.523 / 3.750 |
| hybrid8_dclock | 8 | 2.239 | 0.271 | 49.7% | 0.102 | 2.712 / 3.172 |
| hybrid8_dclock | 32 | 2.245 | 0.271 | 49.8% | 0.103 | 2.710 / 3.254 |
| intent8 | 8 | 2.077 | 0.271 | 50.6% | 0.110 | 2.413 / 3.171 |
| intent8 | 32 | 2.091 | 0.271 | 50.4% | 0.133 | 2.463 / 3.290 |
| intent8_dclock | 8 | 2.250 | 0.270 | 51.0% | 0.110 | 3.060 / 3.261 |
| intent8_dclock | 32 | 2.250 | 0.270 | 51.1% | 0.112 | 3.000 / 3.440 |
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
| baseline_dclock | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1183 |
| embedded8 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1693 |
| embedded8_dclock | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1726 |
| hybrid8 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1699 |
| hybrid8_dclock | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1647 |
| intent8 | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1671 |
| intent8_dclock | 512 | 63 | 32 | 0 / 0 | 4 / 4 | 0.5 / 0.25 | 0.1693 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Legacy auxiliaries disabled |
|---|---|---|---|---:|---|
| baseline_dclock | mixed | proposal | True | 8 / 8 | True |
| embedded8 | mixed | proposal | True | 8 / 8 | True |
| embedded8_dclock | mixed | proposal | True | 8 / 8 | True |
| hybrid8 | mixed | proposal | True | 8 / 8 | True |
| hybrid8_dclock | mixed | proposal | True | 8 / 8 | True |
| intent8 | mixed | proposal | True | 8 / 8 | True |
| intent8_dclock | mixed | proposal | True | 8 / 8 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local transition and recovery scouts

| Run | Shared judging weight | Replacement distribution / max or mild | Local pair GAN weight |
|---|---:|---|---:|
| baseline_dclock | 0.5 | fixed / 0.25 | 0.25 |
| embedded8 | 0.5 | fixed / 0.25 | 0.25 |
| embedded8_dclock | 0.5 | fixed / 0.25 | 0.25 |
| hybrid8 | 0.5 | fixed / 0.25 | 0.25 |
| hybrid8_dclock | 0.5 | fixed / 0.25 | 0.25 |
| intent8 | 0.5 | fixed / 0.25 | 0.25 |
| intent8_dclock | 0.5 | fixed / 0.25 | 0.25 |

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
| baseline_dclock | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| embedded8 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| embedded8_dclock | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| hybrid8 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| hybrid8_dclock | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| intent8 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| intent8_dclock | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |

## G-owned observation recurrence

| Run | G state size | State update reads D | G reads D | Prefix GRU updates / phase |
|---|---:|---|---|---:|
| baseline_dclock | 0 | False | True | 0 |
| embedded8 | 8 | False | False | 126 |
| embedded8_dclock | 8 | False | False | 126 |
| hybrid8 | 8 | False | False | 126 |
| hybrid8_dclock | 8 | False | False | 126 |
| intent8 | 8 | False | False | 126 |
| intent8_dclock | 8 | False | False | 126 |

G state encodes real observations with full real-prefix BPTT, then at most one generated write.
State advances once per observation; proposal/final reads share the same state.
D owns M, G owns S. Both start at zero for cold evaluation. G has no MSE objective.
Memory access controls are separately trained; interventions alone do not establish comparative benefit.

## Local history and future objectives

| Run | Mismatch weight / donors | Mismatch context / strength / writer gradient | Recovery noise / probability | Future weight / offsets | Query bands |
|---|---|---|---|---|---:|
| baseline_dclock | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| embedded8 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| embedded8_dclock | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| hybrid8 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| hybrid8_dclock | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| intent8 | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |
| intent8_dclock | 0.25 / shuffle | clean / 0.25 / True | 0 / 0.5 | 0 / [0, 4, 12] | 0 |

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

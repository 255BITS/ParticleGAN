# Single-point handoff scouts (no trajectory loss)

Completed runs only. Cold = zero-memory start. Warm = real prefix then generated writes only.
Reference-orbit pass requires radial RMSE < 0.1, direction consistency > 0.95,
signed-speed error < 0.03 rad/step, and first-point error < 0.2 reference radii.
This new composite is a diagnostic; see continuous errors and particle coverage in results.json.

| Run | Updates | Cold 256 / 1024 | Cold CW / CCW (256) | Stopped | Prefix8 reference 256 / 1024 | Prefix32 reference 256 / 1024 |
|---|---:|---:|---:|---:|---:|---:|
| clean_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clean_s25_detach | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| proposal_control | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| residual_clean_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| shared_full | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| shared_s25 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6 | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| clock_fourier6_shared | 2000 | 0.0% / 0.0% | 0 / 0 | 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Continuation errors (1,024 generated points)

| Run | Prefix | Radial RMSE | Speed MAE | Direction agreement | Startup error | Position error first32 / last128 |
|---|---:|---:|---:|---:|---:|---:|
| clean_full | 8 | 2.057 | 0.268 | 50.5% | 0.100 | 1.206 / 3.217 |
| clean_full | 32 | 2.063 | 0.269 | 50.2% | 0.096 | 1.211 / 3.295 |
| clean_s25 | 8 | 2.056 | 0.269 | 50.5% | 0.102 | 1.723 / 3.264 |
| clean_s25 | 32 | 2.056 | 0.269 | 50.6% | 0.108 | 1.609 / 3.205 |
| clean_s25_detach | 8 | 2.459 | 0.267 | 50.9% | 0.098 | 1.476 / 3.737 |
| clean_s25_detach | 32 | 2.467 | 0.267 | 50.9% | 0.100 | 1.414 / 3.731 |
| clock_control | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_control | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| proposal_clean_full | 8 | 1.802 | 0.270 | 50.5% | 0.108 | 1.726 / 2.944 |
| proposal_clean_full | 32 | 1.803 | 0.271 | 50.5% | 0.108 | 1.799 / 2.917 |
| proposal_clean_s25 | 8 | 1.497 | 0.267 | 50.9% | 0.096 | 1.572 / 2.543 |
| proposal_clean_s25 | 32 | 1.487 | 0.268 | 50.7% | 0.096 | 1.448 / 2.573 |
| proposal_control | 8 | 1.675 | 0.268 | 50.8% | 0.108 | 1.412 / 2.815 |
| proposal_control | 32 | 1.675 | 0.270 | 50.3% | 0.105 | 1.456 / 2.784 |
| residual_clean_s25 | 8 | 2.536 | 0.269 | 50.5% | 0.089 | 1.521 / 3.591 |
| residual_clean_s25 | 32 | 2.547 | 0.270 | 50.7% | 0.089 | 1.577 / 3.770 |
| shared_full | 8 | 2.330 | 0.265 | 51.3% | 0.189 | 1.325 / 3.627 |
| shared_full | 32 | 2.318 | 0.266 | 51.4% | 0.128 | 1.153 / 3.574 |
| shared_s25 | 8 | 1.727 | 0.270 | 50.6% | 0.102 | 1.425 / 2.847 |
| shared_s25 | 32 | 1.726 | 0.270 | 50.5% | 0.104 | 1.365 / 2.702 |
| clock_fourier6 | 8 | 2.588 | 0.271 | 49.9% | 0.106 | 1.638 / 3.654 |
| clock_fourier6 | 32 | 2.597 | 0.270 | 50.0% | 0.112 | 1.466 / 3.879 |
| clock_fourier6_shared | 8 | 1.619 | 0.264 | 50.6% | 0.122 | 1.642 / 2.691 |
| clock_fourier6_shared | 32 | 1.619 | 0.266 | 50.5% | 0.138 | 1.692 / 2.701 |

## Interpretation and next-run candidates

- No new scout produced a passing cold-start circle at the long horizon; there is no winner on that metric.
- No scout yet passes reference-orbit fidelity for both prefix lengths; do not call a cold oscillator a solved handoff.
- All queued jobs have finished; rankings do not automatically promote a run or establish scientific success.

## Training cost

| Run | Point examples / update | Max real prefix | Memory size | D prediction / temporal weights | G calls D / G phase | Feedback probability / strength | Seconds / update |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_full | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 1 | 0.0416 |
| clean_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0419 |
| clean_s25_detach | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0408 |
| clock_control | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0389 |
| proposal_clean_full | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 1 | 0.0427 |
| proposal_clean_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0430 |
| proposal_control | 512 | 63 | 32 | 0 / 0 | 1 / 1 | 0 / 1 | 0.0401 |
| residual_clean_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0423 |
| shared_full | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 1 | 0.0409 |
| shared_s25 | 512 | 63 | 32 | 0 / 0 | 2 / 2 | 0.5 / 0.25 | 0.0419 |

## Adversarial memory exploration

| Run | D judging memory | G adapter | Proposal gradient | Reader calls D / G phase | Auxiliaries disabled |
|---|---|---|---|---:|---|
| clean_full | clean | none | True | 2 / 2 | True |
| clean_s25 | clean | none | True | 2 / 2 | True |
| clean_s25_detach | clean | none | False | 2 / 2 | True |
| clock_control | shared | none | False | 1 / 1 | True |
| proposal_clean_full | clean | proposal | True | 4 / 4 | True |
| proposal_clean_s25 | clean | proposal | True | 4 / 4 | True |
| proposal_control | shared | proposal | False | 2 / 2 | True |
| residual_clean_s25 | clean | residual | True | 2 / 2 | True |
| shared_full | shared | none | True | 2 / 2 | True |
| shared_s25 | shared | none | True | 2 / 2 | True |

Clean judging: G reads the generated-write state, while both candidate scores
and B-cap use the same real-history memory, strictly before the target.
The proposal adapter uses two point-reader passes and stores no private state.
Reader calls include those internal passes; G calls count complete G evaluations.

## Local memory dynamics settings

| Run | Slow coordinates / rate | G adapter bottleneck | Repair target / weight / noise | Stability D / G weight / max gain |
|---|---:|---:|---|---|
| clean_full | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| clean_s25 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| clean_s25_detach | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| clock_control | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_clean_full | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_clean_s25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| proposal_control | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| residual_clean_s25 | 0 / 0.1 | 16 | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| shared_full | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |
| shared_s25 | 0 / 0.1 | off | raw / 0 / 0.05 | 0 / 0 / 1.1 |

All new scouts use local next-point GAN losses. There is no full generated training
rollout, trajectory critic, or cold/warm path loss. Configured feedback adds at most one
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

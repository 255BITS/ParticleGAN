# Completed: bounded-feedback experiment session

20 fresh 2k scouts and one exact continuation to 5k completed through the shared
two-GPU pipeline. All 21 valid jobs have **zero full cold256/1024 circle passes**
and **zero 1024-step original-orbit continuation passes** at both real prefixes.
One short-horizon exception: recent_bound_control passed1/128 at256 steps after
prefix32, falling to0/128 at1024. Every other warm256 rate was zero.
The toy remains unsolved.

No full generated training rollout or path critic was introduced; feedback
experiments used at most two G evaluations and one write per phase. D owns writer
parameters. Fixed particles, default API B-cap, no clipping/EMA, config-controlled
comparisons, no seed sweeps. All valid queues finished without failures.

| Comparison | Jobs | Outcome |
|---|---:|---|
| Detached one-write feedback | 8 | No full cold/long warm passes; strong replacement often caused alternating motion |
| Recent-point slots and residual G | 6 | No full cold/long warm passes; one short warm pass in the bounded GRU control |
| Corrected G gradient through one write | 4 | No full cold/long warm passes; no convincing gain over detached counterparts |
| Residual model continuation to5k | 1 | Earlier late-circle lead disappeared;84.4% late stopping |
| Noiseless controls | 2 | Both failed; removing observation noise was insufficient at2k |

## What we learned

A single generated replacement did not repair autonomy in the tested forms.
Recovery targets can be inconsistent with a badly wrong preceding point; the
frequency, interpolation, ramp and mature-prefix variants did not solve this.
Passing G gradients through the write was also insufficient. Explicit recent
history and residual output improved some late behavior, but not reliable startup
or original-orbit preservation. These results do not establish impossibility.

The strongest secondary signal was recent4_delta at2k:60.2% of final128-point
windows passed the circle diagnostic at1024 steps. All77 were CCW;76 also passed
over the final512 points. Full-path success stayed zero. At5k that late rate fell
to zero, stopping rose to84.4%, and prefix32 radial error grew from1.085 to21.085
reference radii. This was a useful negative extension, not a winner.

Noiseless controls use clean training observations and clean warm evaluation
prefixes, an easier setting. Cold metrics have the same definition. Neither
passed, so observation noise alone does not explain failure at the tested budget.
A particle-swap evaluation intervention also gave no meaningful repair (at most
1/128); it violates fixed-particle policy and is excluded from model rankings.

## Correctness fix and validation

The first two-step gradient implementation omitted the real score's dependence
on generated M. The paired loss requires both real/fake branches; otherwise a
shared memory-only score offset creates a spurious G gradient. This was fixed
and tested. Two completed runs from backprop_round5 were invalidated and excluded;
two active jobs were intentionally stopped and one pending job cancelled. All
five intended jobs ran in the corrected queue. Earlier detached-feedback and
architecture runs are unaffected.

60 focused tests plus one invalidated-report exclusion test passed. Tests cover
causal timing, active B-cap, ownership, exact resume, bounded G calls, recent-slot
ordering, residual output, proposal gradient flow, and paired-score offset
cancellation. Corrected GPU smoke passed. Reporter/checkpoint guards prevent
reuse of invalidated experiments. The final raw-metric audit caught and retained
the single short warm pass rather than rounding the entire batch to zero.

## Recommendation

Do not extend these models again on current evidence. Next, use a clearly
labelled **local prediction-loss diagnostic** on the same memory/reader to separate
adversarial optimization from representation and feedback stability. Restrict it
to sufficiently informative real prefixes, retain the GAN branch, and configure
its weight explicitly. This changes the G objective and must be reported as such;
it is not evidence that pure pointwise GAN training works. No rollout loss is
needed. This is proposed, not implemented or queued.

Preserve full-path metrics and direction coverage. One-point fit on real context
and long-term self-consistency are distinct tests; a burn-in-only circle criterion
would hide the startup failure.

## Completed run table

All full cold and1024-step warm rates are zero. The one short warm exception is
noted above. Late-window/stopping numbers below are secondary diagnostics.

| Run | Updates | Observation noise | Late-circle rate at1024 | Late stopped at1024 |
|---|---:|---:|---:|---:|
| feedback_p100 | 2000 | 0.03 | 0.0% | 3.1% |
| feedback_p100_mix50 | 2000 | 0.03 | 0.0% | 20.3% |
| feedback_p25 | 2000 | 0.03 | 10.2% | 17.2% |
| feedback_p50 | 2000 | 0.03 | 0.0% | 0.0% |
| feedback_p50_ctx16 | 2000 | 0.03 | 0.0% | 0.0% |
| feedback_p50_mature | 2000 | 0.03 | 0.0% | 10.2% |
| feedback_p50_mix50 | 2000 | 0.03 | 14.8% | 48.4% |
| feedback_p50_ramp | 2000 | 0.03 | 0.0% | 3.1% |
| recent2_delta | 2000 | 0.03 | 21.1% | 25.8% |
| recent4_absolute | 2000 | 0.03 | 6.2% | 70.3% |
| recent4_delta | 2000 | 0.03 | 60.2% | 14.8% |
| recent4_delta_fb25 | 2000 | 0.03 | 10.9% | 35.9% |
| recent4_delta_fb50_mature | 2000 | 0.03 | 0.0% | 0.0% |
| recent_bound_control | 2000 | 0.03 | 14.1% | 56.2% |
| feedback_p25_bp_v2 | 2000 | 0.03 | 11.7% | 53.9% |
| feedback_p50_mature_bp_v2 | 2000 | 0.03 | 0.0% | 3.9% |
| recent4_delta_5k_v2 | 5000 | 0.03 | 0.0% | 84.4% |
| recent4_delta_fb25_bp_v2 | 2000 | 0.03 | 1.6% | 29.7% |
| recent4_delta_fb50_mature_bp_v2 | 2000 | 0.03 | 5.5% | 8.6% |
| clean_gru | 2000 | 0 | 24.2% | 49.2% |
| clean_recent4_delta | 2000 | 0 | 0.0% | 58.6% |

[Session metrics](feedback-session.json).
Detailed leaderboards: [feedback](feedback_round3/leaderboard.md),
[recent memory](recent_round4/leaderboard.md),
[corrected gradients and extension](backprop_round5_corrected/leaderboard.md),
[noiseless controls](clean_round6/leaderboard.md).

All valid queues are finished with no failures or pending jobs. Central log:
`tail -F runs/memory_path/core_round1/train.log`. Changes remain uncommitted on
feat/sequential-memory-path; unrelated files were preserved.

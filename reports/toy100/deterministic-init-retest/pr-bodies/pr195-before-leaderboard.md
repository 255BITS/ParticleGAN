This draft now includes `develop`'s **new deterministic weight initialization**, merged from `c720645` (PR194). Fresh network weights and recipe-created priors use the new initialization; sampling remains stochastic. The merge preserves KA2's optimizer behavior, EMA synchronization and compatible checkpoint continuation.

**The first new-initialization test fails.** On the short eight-cluster coverage problem, default KA2 finishes with **6 of 8 clusters**, constant-rate KA2 with **4 of 8**, and K3P with **6 of 8**. All three have **0/24 passing observations**. This does not support selecting KA2 as the default.

**The search is stopped at the user's request.** All 49 declared API quick-screen configurations completed. Four pass that screen, but each then fails a different quality task; none qualifies as the default. Historical research reruns and remaining work are recorded separately in the [retest closeout](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/closeout.md). Unrun checks are not claimed as passes or failures.

The requirement remains one learner that can run indefinitely without caller-controlled phases. Automatic reversible rate changes are allowed. KA2 has not demonstrated that behavior through the public API.

**Merge validation:** 246 CPU tests passed, one CUDA-only test skipped in the authoritative CPU run. An earlier test invocation also ran the existing CUDA unit test; this is separate from benchmark qualification. Initialization, pretrained preservation, EMA and old schema-4 checkpoint continuation are covered; K3P schema1–3 remain incompatible with KA2.

This PR remains **draft, unmerged, and based on develop**. New initialization does not itself remove the tested learner's schedules or establish continuous stability.

**Historical API checks, before deterministic initialization:** KA2 learns quickly at constant learning rates, then repeatedly loses stability. Automatic decay is also acceptable, provided it can decide when to lower rates and raise them again. The current API does not implement that reversible policy.

| Actual public `GANTrainer` run | Retention before the change | First reaches the new target | Passing observations afterward |
|---|---:|---:|---:|
| Constant learning rates | 61/120 | After 120 updates | 126/209 |
| Decay diagnostic | 120/120 | After 1,690 updates | 48/52 |

Both runs continue to update 4,600. Constant KA2 collapses before the target changes and has further long departures afterward. The decay diagnostic ran only after constant KA2 failed; it preserves the original distribution but adapts more slowly. This is about arrival time and subsequent stability, **not an 81/81 deadline requirement**. The tests do not establish that KA2 meets the release requirement or beats K3P in the public API.

**Why KA2 was selected in research:** it preserved the original distribution better than R2 and had fewer post-arrival departures than K3P in the shared research window. The public optimizer/penalty factories now reproduce that research run exactly, including the EMA buffer fix. The new experiment tests the actual trainer with public defaults; that broader check is why the merge remains blocked. Historical K3P's 22/22 score is not a KA2 result.

**Before merge:** demonstrate stable learning with constant rates or automatic reversible decay. For an automatic policy, verify that it lowers rates, raises them again after delayed/repeated target changes, and settles afterward without target-change notifications or benchmark quality scores. Compare against K3P under the same protocol and runtime. No adaptive controller has been added or claimed to pass.

**Historical validation:** the pair has identical initial models, optimizer state, RNG state, noise timings and sources; only the two LR floors differ. All 4,600 applied-rate records per run and arrival/stability summaries pass the independent evidence audit. Earlier library validation remains 1,058 CPU tests passed, 11 skipped, exact CUDA component replays, and package checks passed. Old K3P checkpoints require 0.8.0. No seed variants were run.

[Public trainer results and reproducible evidence](https://github.com/255BITS/ParticleGAN/blob/codex/ka2-default-candidate/reports/ka2-default-candidate/constant-lr-api/README.md) · [Research selection rationale](https://github.com/255BITS/ParticleGAN/blob/codex/ka2-default-candidate/reports/ka2-default-candidate/README.md)

# BCAP Tier 1 repairs and baseline research

This iteration diagnoses the two-pole stability and word runtime failures in [PR377](https://github.com/255BITS/ParticleGAN/pull/377), tests bounded repairs across all six Tier 1 tasks, and then compares five distinct research ideas from the resulting baseline. Compatibility with other techniques remains a requirement throughout. The previous [full comparison](../bcap-develop-integration/README.md) and its negative results retain their original identities.

## Phase 1 Parallel diagnosis

Three isolated workflows investigate two-pole causality, word performance, and compatibility. Two-pole has at most six original-contract diagnostic arms, with 1,800 seconds of full reservations. Profiling and targeted software verification share a separate 1,200-second allowance. Short profiling fixtures cannot certify a training gate. All bulk artifacts live under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next`.

Completed diagnosis: projection alone exactly reproduces the incumbent's two-pole path, while global transport is sufficient to reproduce the combination's short passing suffix. The incumbent has 17 consecutive terminal passes; transport and the combination have one, below the original requirement of five. Local-only transport has a late slope excursion. This fixture therefore points to transport changing the critic's operating path, rather than a projection conflict. [Certified ablations and saved-state audit](../bcap-tier1-stability/README.md).

Two preregistered repairs pass that screen. A constant training cap of 0.9 has nine consecutive terminal passes; a same-training-panel finite critic-step guard has seventeen. The latter rejects 49 of 80 critic displacements, so its better scoped slope bound may trade away learning on larger hosts. Neither short screen establishes full Tier 1 performance.

Full SVD accounts for 89.7% of CUDA self time in the bounded word profile. The optional CPU full-SVD backend reduces measured update time from 43.774 to 17.635 milliseconds, a 2.48× profile speedup. Its projected 353-second training time excludes full-run evaluation and checkpoint overhead. CUDA and CPU SVD rounding changes the actual training path; it is an explicit numerical trainer change requiring fresh gate measurements. Native defaults match the archived source bitwise in a separate matched replay. [Profiles, implementation and limitations](../bcap-tier1-performance/README.md).

Fresh integrated compatibility checks pass for both compound repairs: 78 checks each, including 42 exact state pairs against the actual frozen `develop` source. All previous comparators and receipts are preserved; an outside-Git adapter adds precisely two checked inactive metadata defaults. Source inventories include the new optimizer and recipe files. [Cap-margin receipt](compatibility-cap-margin.json), [finite-cap receipt](compatibility-finite-cap.json). Another 68 checks pass on the merged implementation.

## Phase 2 Repairs and baseline selection

Use the diagnosis to specify and test at most two global repairs. Their two-pole screens reserve 600 seconds. Freeze their implementations and the incumbent on a common source, then measure all six original Tier 1 questions for each complete configuration. Retain the existing view's optional 300-second clock audit as a separately visible diagnostic. Three complete arms reserve 7,560 seconds, including those audits. The total phase 1/2 paid ceiling is 10,000 seconds, including admitted retries; the planned full reservations total 9,960 seconds.

A repaired baseline requires 6/6 passes. Among passing repairs, select the one with fewer changed recipe fields relative to the combined reference; break ties by lexicographically ordered candidate ID. Prefer a passing combined repair over reverting to the incumbent. If neither repair passes all six, use the newly measured incumbent as the research baseline and preserve both repair failures. If the incumbent also fails on the frozen source, retain the previously measured incumbent as historical evidence and explicitly identify the new source as unresolved; no candidate acquires a passing baseline label.

The complete comparison selects the freshly measured **incumbent**, candidate `bcap-three-phase-incumbent-v1`. Both compound repairs pass five tasks and fail the ring task. All three complete the original 20,001-update word task within its 900-second allowance. The optional clock audits also pass, independently of the six required gates.

| Global configuration | Required Tier 1 | Ring terminal passing suffix (requires 5) | Ring covariance error (requires ≤ .85) | Word elapsed seconds |
|---|---:|---:|---:|---:|
| Incumbent | 6/6 | 26 | .49657 | 755.73 |
| Projection + transport + CPU SVD + cap .9 | 5/6 | 1 | .79889 | 392.60 |
| Projection + transport + CPU SVD + finite cap | 5/6 | 0 | 1.36934 | 470.25 |

The margin endpoint passes the ring thresholds, but its penultimate shape error is .86836, breaking the required sustained suffix. The finite-cap endpoint fails shape despite retaining all 16 modes. Ring projection blends are only 3/1, with zero projection stalls, so persistent objective conflict does not explain these failures. The finite guard rejects 299/1,600 ring critic displacements and accepts a mean proposal fraction of .60372. Reduced critic response is a plausible cause, alongside local transport geometry and changing probe panels; the compound backend/auxiliary/guard changes prevent isolated attribution. The guard enforces non-increase on the current training panel when its slope already exceeds the cap, rather than a global absolute slope bound. [Measured guard counters and checkpoint identities](phase2-finite-cap-counters.json).

The selected [baseline contract](baseline.json) records the exact global recipe, measured source `0f9787fa3f164f6d2034144714f8e7ba7ab75bbf`, and scientific digest. The later publication fork contains reports and helpers with the same scientific digest. Neither repair is eligible for baseline promotion, and Phase 2 adds no Tier 2 qualification.

[Final task metrics](phase2-results.json), [saved-state audit](phase2-audit.json), and [21 actual-training GIF receipts](media/index.json) retain all three configurations and the separate clock cohort. All twelve saved-state comparison groups pass the data and consumed-stream checks. Initial model/prior hashes are checked where recorded; the explicit two-pole fixture has no separate initial tensor dump. Phase 2 consumes 1,873.19 paid seconds across 21 attempts, with zero retries. Phases 1 and 2 total 1,934.85 paid seconds against the 10,000-second ceiling; their full reservations remain 9,960 seconds.

## Phase 3 Five research workflows

Create five isolated branches from the exact phase 2 baseline PR commit. Each workflow must read the diagnosis and experiment memory, choose a distinct falsifiable mechanism, implement it through the shared public API, and run a registered comparison against the unchanged baseline on its own frozen source. All five configurations use global settings across their declared task scope.

The five directions are optimistic game dynamics, anisotropic data geometry, debiased entropic transport, secant-based adaptive steps, and transport mobility based on finite-batch uncertainty. Equations, predictions, competing explanations, numerical falsifiers, task rosters, and full budget reservations must be frozen before any paid experiment. These are research directions rather than preselected successful mechanisms; the agents will use phase 1/2 findings to specify them. Five workflows run in two waves because this session has three subagent slots.

The proposed common phase 3 scope contains all six original Tier 1 questions and ten Tier 2 questions: Gaussian stability, word hold, trajectory, residual student, unequal mass, unequal width, anisotropic, grid100, rotated100, and staggered100. Each track has one baseline and one substantive candidate, with 45,840 seconds of full reservations and a 48,000-second ceiling including admitted retries. Five tracks therefore reserve 229,200 seconds within a 240,000-second paid ceiling; targeted software checks have a separate total allowance of 1,500 seconds. These are reservation ceilings, not predicted execution times or permission to add configurations.

Freeze each track's actual roster and mechanism after baseline measurement. These comparisons use a separately admitted research diagnostic scope, retain all original numerical gates and own-checkpoint dependencies, and grant no ordinary qualification. Each own Gaussian and word producer must pass completely before its hold can run; a candidate cannot borrow baseline states. Failed ordinary Tier 1 gates continue to block ordinary Tier 2. A research improvement is not a public default or merge decision. Eleven original Tier 2 questions remain outside this research subset, so a successful subset does not imply 21/21 success.

The [shared runner](phase3-workflow.md) validates the complete roster, reserves both arms, freezes committed source, and publishes only certified saved observations. Its sixteen software controls cover READY preparation, source/commit guards, original conditions, diagnostic-only publication, and rejection of borrowed or failed checkpoint producers. The [agent research contract](phase3-handoff.md) describes implementation, evidence and PR requirements.

## Comparison and publication

Keep seed 0, public deterministic initialization, each task's architecture, original fixture, actual batch sequence, prior, sampling, update allowance, schedule horizon, evaluation cadence, and numerical gates fixed across compared trainers. Declare every trainer delta; checkpoint all consumed named streams. Execution optimizations require exact state parity, otherwise they are numerical trainer changes. Disabled features must preserve other techniques and old checkpoint loading.

Use numerical gates and compact source-bound metrics to compare approaches. Keep the repository's [single current technique inventory](../technique-inventory.md) as the goal leaderboard; individual workflows publish evidence and readouts, not additional generated goal leaderboards. Completed public-API tests retain actual-training GIFs. Raw stdout, JSONL traces, JUnit, checkpoints, and state dumps stay outside Git. No seed experiments, task-specific repairs, threshold changes, endpoint rescue, or automatic promotion.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/*/logs/driver.log
```

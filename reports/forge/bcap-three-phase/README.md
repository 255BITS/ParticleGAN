# BCAP Tier 1 repairs and baseline research

This iteration diagnoses the two-pole stability and word runtime failures in [PR377](https://github.com/255BITS/ParticleGAN/pull/377), tests bounded repairs across all six Tier 1 tasks, and then compares five distinct research ideas from the resulting baseline. Compatibility with other techniques remains a requirement throughout. The previous [full comparison](../bcap-develop-integration/README.md) and its negative results retain their original identities.

## Phase 1 Parallel diagnosis

Three isolated workflows investigate two-pole causality, word performance, and compatibility. Two-pole has at most six original-contract diagnostic arms, with 1,800 seconds of full reservations. Profiling and targeted software verification share a separate 1,200-second allowance. Short profiling fixtures cannot certify a training gate. All bulk artifacts live under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next`.

## Phase 2 Repairs and baseline selection

Use the diagnosis to specify and test at most two global repairs. Their two-pole screens reserve 600 seconds. Freeze their implementations and the incumbent on a common source, then measure all six original Tier 1 questions for each complete configuration. Retain the existing view's optional 300-second clock audit as a separately visible diagnostic. Three complete arms reserve 7,560 seconds, including those audits. The total phase 1/2 paid ceiling is 10,000 seconds, including admitted retries; the planned full reservations total 9,960 seconds.

A repaired baseline requires 6/6 passes. Among passing repairs, select the one with fewer changed recipe fields relative to the combined reference; break ties by lexicographically ordered candidate ID. Prefer a passing combined repair over reverting to the incumbent. If neither repair passes all six, use the newly measured incumbent as the research baseline and preserve both repair failures. If the incumbent also fails on the frozen source, retain the previously measured incumbent as historical evidence and explicitly identify the new source as unresolved; no candidate acquires a passing baseline label.

## Phase 3 Five research workflows

Create five isolated branches from the exact phase 2 baseline PR commit. Each workflow must read the diagnosis and experiment memory, choose a distinct falsifiable mechanism, implement it through the shared public API, and run a registered comparison against the unchanged baseline on its own frozen source. All five configurations use global settings across their declared task scope.

The research directions will address different causes: finite-step critic dynamics, allocation and anisotropic density, noise in transport forces, shared-parameter geometry, and retention with a changing target. Equations, predictions, competing explanations, numerical falsifiers, task rosters, and full budget reservations must be frozen before any paid experiment. These are research directions rather than preselected successful mechanisms; the agents will use phase 1/2 findings to specify them. Five workflows run in two waves because this session has three subagent slots.

The proposed common phase 3 scope contains all six original Tier 1 questions and ten Tier 2 questions: Gaussian stability, word hold, trajectory, residual student, unequal mass, unequal width, anisotropic, grid100, rotated100, and staggered100. Each track has one baseline and one substantive candidate, with 45,840 seconds of full reservations and a 48,000-second ceiling including admitted retries. Five tracks therefore reserve 229,200 seconds within a 240,000-second paid ceiling; targeted software checks have a separate total allowance of 1,500 seconds. These are reservation ceilings, not predicted execution times or permission to add configurations.

Freeze each track's actual roster and mechanism after baseline measurement. These comparisons use a separately admitted research diagnostic scope, retain all original numerical gates and own-checkpoint dependencies, and grant no ordinary qualification. Each own Gaussian and word producer must pass completely before its hold can run; a candidate cannot borrow baseline states. Failed ordinary Tier 1 gates continue to block ordinary Tier 2. A research improvement is not a public default or merge decision. Eleven original Tier 2 questions remain outside this research subset, so a successful subset does not imply 21/21 success.

## Comparison and publication

Keep seed 0, public deterministic initialization, each task's architecture, original fixture, actual batch sequence, prior, sampling, update allowance, schedule horizon, evaluation cadence, and numerical gates fixed across compared trainers. Declare every trainer delta; checkpoint all consumed named streams. Execution optimizations require exact state parity, otherwise they are numerical trainer changes. Disabled features must preserve other techniques and old checkpoint loading.

Use numerical gates and compact source-bound metrics to compare approaches. Keep the repository's [single current technique inventory](../technique-inventory.md) as the goal leaderboard; individual workflows publish evidence and readouts, not additional generated goal leaderboards. Completed public-API tests retain actual-training GIFs. Raw stdout, JSONL traces, JUnit, checkpoints, and state dumps stay outside Git. No seed experiments, task-specific repairs, threshold changes, endpoint rescue, or automatic promotion.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/*/logs/driver.log
```

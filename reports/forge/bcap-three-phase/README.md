# BCAP Tier 1 repairs and baseline research

This iteration diagnoses the two-pole stability and word runtime failures in [PR377](https://github.com/255BITS/ParticleGAN/pull/377), tests bounded repairs across all six Tier 1 tasks, and then compares five distinct research ideas from the resulting baseline. Compatibility with other techniques remains a requirement throughout. The previous [full comparison](../bcap-develop-integration/README.md) and its negative results retain their original identities.

## Phase 1 Parallel diagnosis

Three isolated workflows investigate two-pole causality, word performance, and compatibility. Two-pole has at most six original-contract diagnostic arms, with 1,800 seconds of full reservations. Profiling and targeted software verification share a separate 1,200-second allowance. Short profiling fixtures cannot certify a training gate. All bulk artifacts live under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next`.

## Phase 2 Repairs and baseline selection

Use the diagnosis to specify and test at most two global repairs. Their two-pole screens reserve 600 seconds. Freeze their implementations and the incumbent on a common source, then measure all six original Tier 1 questions for each complete configuration. Three full arms reserve 6,660 seconds. The total phase 1/2 paid ceiling is 10,000 seconds, including admitted retries; the planned full reservations total 9,060 seconds.

A repaired baseline requires 6/6 passes. Among passing repairs, select the one with fewer changed recipe fields relative to the combined reference; break ties by lexicographically ordered candidate ID. Prefer a passing combined repair over reverting to the incumbent. If neither repair passes all six, use the newly measured incumbent as the research baseline and preserve both repair failures. If the incumbent also fails on the frozen source, retain the previously measured incumbent as historical evidence and explicitly identify the new source as unresolved; no candidate acquires a passing baseline label.

## Phase 3 Five research workflows

Create five isolated branches from the exact phase 2 baseline PR commit. Each workflow must read the diagnosis and experiment memory, choose a distinct falsifiable mechanism, implement it through the shared public API, and run a registered comparison against the unchanged baseline on its own frozen source. All five configurations use global settings across their declared task scope.

The research directions will address different causes: finite-step critic dynamics, allocation and anisotropic density, noise in transport forces, shared-parameter geometry, and retention with a changing target. Equations, predictions, competing explanations, numerical falsifiers, task rosters, and full budget reservations must be frozen before any paid experiment. These are research directions rather than preselected successful mechanisms; the agents will use phase 1/2 findings to specify them. Five workflows run in two waves because this session has three subagent slots.

Phase 3 budgets and task rosters are declared after the baseline is measured and before execution. Failed ordinary Tier 1 gates block ordinary Tier 2. Any deeper research uses a separately admitted diagnostic scope, keeps original numerical gates and own-checkpoint dependencies, and grants no ordinary qualification. A research improvement is not a public default or merge decision.

## Comparison and publication

Keep seed 0, public deterministic initialization, each task's architecture, original fixture, actual batch sequence, prior, sampling, update allowance, schedule horizon, evaluation cadence, and numerical gates fixed across compared trainers. Declare every trainer delta; checkpoint all consumed named streams. Execution optimizations require exact state parity, otherwise they are numerical trainer changes. Disabled features must preserve other techniques and old checkpoint loading.

Use numerical gates and compact source-bound metrics to compare approaches. Keep the repository's [single current technique inventory](../technique-inventory.md) as the goal leaderboard; individual workflows publish evidence and readouts, not additional generated goal leaderboards. Completed public-API tests retain actual-training GIFs. Raw stdout, JSONL traces, JUnit, checkpoints, and state dumps stay outside Git. No seed experiments, task-specific repairs, threshold changes, endpoint rescue, or automatic promotion.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/*/logs/driver.log
```

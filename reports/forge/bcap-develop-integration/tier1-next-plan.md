# BCAP Tier 1 repair plan

The next round should restore **6/6 Tier 1 passes under one global recipe** before expanding Tier 2. Start from draft [PR377](https://github.com/255BITS/ParticleGAN/pull/377), preserving its opt-in integration and compatibility with other techniques. This is a plan for execution after compaction; no new experiments are launched by this update.

## Failures to resolve

The [completed comparison](README.md) establishes two different blockers:

- **Two-pole stability:** the combination passes its endpoint but retains only one consecutive terminal passing observation; five are required. Its final five critic gradient medians are `1.0068, 1.0266, 1.0194, 1.0278, 0.9852`, against the unchanged upper bound of `1`. Transport runs on all 80 updates, while projection records zero conflicts, blends, or stalls. Transport changing the particle trajectory and finite critic steps overshooting the soft cap are hypotheses to distinguish.
- **Word execution:** training reaches update `19168/20001` before the unchanged 900-second timeout. Acquisition was observed earlier, but the incomplete run cannot certify a checkpoint or unlock word hold. Completing the run still requires its original generation and reconstruction gates; late reconstruction also fluctuated.

## Parallel work after compaction

Use three subagents in isolated worktrees, with the parent integrating their changes:

1. **Two-pole causality and stability.** Analyze the existing curves and saved states first. Register a bounded diagnostic comparison of the incumbent, projection only, transport only, projection with global transport only, projection with local transport only, and the full combination. Reuse exact compatible controls where possible; changed source or runtime bindings require fresh controls. Each arm keeps all 80 updates and 24 checks. Compare passing suffix, movement, critic gradient excursions, and mechanism activation. This diagnostic subset grants no ordinary qualification.
2. **Word performance.** Profile the public joint host, separating transport kernels, projected sorting, backward passes, and synchronization costs. Optimize measured redundant work while preserving the objective, original forwards, optimizer history, and consumed RNG streams. Establish exact state/output parity for execution-only changes. Any numerical change becomes an explicit trainer delta. Short timing runs remain software diagnostics. Aim for useful headroom inside 900 seconds without reducing updates or evaluation cadence.
3. **Compatibility and evidence.** Verify disabled-feature parity against actual develop, enabled checkpoint continuation, host contracts, and isolated/checkpointed streams. Prepare fresh Forge declarations, source bindings, artifact receipts, and the single current leaderboard for this goal. Numerical gates determine results; retain actual-training GIFs for completed public-API tests.

## Repairs and selection

Use the causal results to choose at most two substantive global repairs. If supported, investigate scale-aware transport control and finite-step critic damping. Specify each equation, changed factor, prediction, and falsifier before training it. A constant/coefficient change must be identified separately from a structural change. Keep both transport signals where the evidence supports doing so, but do not claim preservation of the three Tier 2 repairs until they are measured again.

Screen these repairs on the original two-pole question, then freeze the source and at most two candidate configurations for a complete Tier 1 comparison with the incumbent. A candidate must pass all six tasks as one configuration; task-specific switches, relaxed gradient bounds, early stopping, and pooling task winners cannot rescue it. If both candidates pass, retain the smaller declared trainer delta, using a preregistered deterministic tie-break.

If the bounded round produces no complete pass, retain the incumbent, report the negative results, and propose another hypothesis before further spending. Tier 1 success enables the next Tier 2 investigation; it does not satisfy the existing merge bar or provisional-profile calibration.

## Contracts and budget

Keep protocol seed 0, public deterministic initialization, original explicit fixed fixtures, architecture, actual data batches, priors, sampling laws, update budgets, schedule horizons, evaluation cadence, and numerical gates fixed across candidates. Never substitute random initialization for two-pole's original stored critic/zero-particle fixture. Preserve prior evidence under its actual source and cohort. Future qualification on changed source needs fresh compatible evidence.

Proposed paid reservations are **1,800 seconds** for up to six two-pole diagnostic arms, **600 seconds** for at most two repair screens, and **6,660 seconds** for three complete Tier 1 arms: **9,060 seconds total**, within a **10,000-second campaign ceiling** including any admitted execution retries. Each full Tier 1 arm reserves its original **2,220 seconds**: Gaussian 120, two-pole 300, ring 300, unused token 300, AE 300, and words 900. Reused compatible evidence reduces spending; it does not authorize additional candidates. Profiling and targeted compatibility verification have a separate proposed **1,200-second software allowance**. No Tier 2 training is included in this round.

These are draft ceilings. Before execution, create schema-v3 candidates and reviewed ready studies with full reservations, immutable source/runtime bindings, declared selection and stopping rules, and fresh campaign IDs. Follow [EXPERIMENTATION.md](../../../EXPERIMENTATION.md) and [compiled experiment memory](../EXPERIMENT_MEMORY.md); do not rewrite or rerun the completed integration cohort merely for this plan.

Keep raw logs, traces, checkpoints, and JUnit outside Git under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next`. The next workflow should expose `logs/driver.log` and per-attempt logs for `tail -F`; commit compact metrics, explanations, provenance, reproduction sources, and one current leaderboard after execution.

## Resume after compaction

Planning source is integration commit `1c128562f8fb9ab326a0ee8a8bc0551516c24648` in `/home/martyn/dev/ParticleGAN-bcap-develop-integration`, branch `integration/bcap-direction-transport`. The original `/home/martyn/dev/ParticleGAN` worktree is separate and should remain untouched. Read this plan, the completed report, `EXPERIMENTATION.md`, and compiled memory before implementing. Keep PR377 in draft during the repair round.

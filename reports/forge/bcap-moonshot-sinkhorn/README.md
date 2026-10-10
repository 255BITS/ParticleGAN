# Debiased entropic transport: complete paired readout

The candidate repairs **anisotropic** but regresses **two-pole** and **unused-token hold**. The baseline passes **7/16**, including **6/6 original Tier 1**; the candidate passes **6/16**, including **4/6 original Tier 1**. All 32 declared cells are measured, with no final unknown, BLOCKED, INVALID or INCOMPLETE cells. Stop this exact global package. Its main preregistered width prediction is **falsified**: final covariance error is **21.078472**, above .85, with zero terminal passing checks.

[Final metrics and certified attempt history](phase3-results.json) · [Saved-state audit](phase3-audit.json) · [Counter receipt](sinkhorn-counters.json) · [Publication provenance](publication-receipt.json) · [All actual-training media receipts](media/index.json).

## Scope and mechanism

Frozen scientific source is `80d165fc3550625e08cfb7e696cf7e86adfa91ec`, digest `1f494ad8975b2b26fe53519c04144ce154840e376c49d92a9c714ca13172b668`. Both arms inherit the freshly measured Phase 2 incumbent. The baseline adds zero recipe overrides. The complete candidate delta is global transport weight 1, `kinetic_transport_mode="sinkhorn"`, epsilon .1, 24 solver rounds, at most 128 deterministic existing batch rows, and the explicit CPU full-SVD backend. Local/sliced transport, direction projection and finite critic guarding are disabled. This compound comparison cannot isolate Sinkhorn from backend rounding.

The finite objective is cross free energy minus half of each self free energy, with squared mean-coordinate cost divided by twice the detached original-real-panel coordinate variance, floored at machine epsilon. Autograd differentiates all 24 damped simultaneous log-domain rounds, the dual mass correction and both fake arguments of the fake self term. Potentials reset each call. No detached-plan derivative, solver retry, warm start, target oracle or random draw enters. Every original row remains in the original trainer; only the auxiliary term subsamples existing rows. The 2,048-row native batch therefore contributes 128 auxiliary rows while preserving its full original adversarial/penalty panels.

[Preregistered derivation and competing explanations](theory.md) · [Spec](spec.json) · [READY registration](registration.json). [Feydy et al.](https://arxiv.org/html/1810.08278v1) motivate the two self corrections; [Cuturi](https://arxiv.org/abs/1306.0895) motivates entropic matrix scaling. Their converged divergence guarantees are not inherited by this finite implementation. This smooth many-to-many objective differs from the original round-five detached block-128 bijection, prior-only routing and four-iteration output-kernel filter. Those original negatives retain source `653c38045618ad240524237a9c141c8d06b28c03`.

Seed 0, public deterministic initialization, each task's architecture, law, prior, full batch draws, sampling, update budget, evaluation cadence and all numerical gates are fixed. The original two-pole stored-weight/zero fixture remains its separate explicit cohort. These six original Tier 1 and ten original Tier 2 questions are a separately admitted research diagnostic subset; eleven original Tier 2 questions remain outside it. No ordinary qualification, default promotion, robustness or 21/21 claim follows.

## Complete original task matrix and media

Each linked PASS/FAIL opens the actual-training GIF for that cell. GIFs use certified saved observations; publication adds zero updates and zero sampling draws. Gates determine these results.

| Original tier | Task | Baseline | Candidate |
| --- | --- | --- | --- |
| 1 | gaussian1d_smoke | [PASS](media/baseline-gaussian1d_smoke.gif) | [PASS](media/candidate-gaussian1d_smoke.gif) |
| 1 | two_pole | [PASS](media/baseline-two_pole.gif) | [FAIL](media/candidate-two_pole.gif) |
| 1 | unused_token_hold | [PASS](media/baseline-unused_token_hold.gif) | [FAIL](media/candidate-unused_token_hold.gif) |
| 1 | ae_gan_hold | [PASS](media/baseline-ae_gan_hold.gif) | [PASS](media/candidate-ae_gan_hold.gif) |
| 1 | ring16_acquisition | [PASS](media/baseline-ring16_acquisition.gif) | [PASS](media/candidate-ring16_acquisition.gif) |
| 1 | five_word_joint_smoke | [PASS](media/baseline-five_word_joint_smoke.gif) | [PASS](media/candidate-five_word_joint_smoke.gif) |
| 2 | gaussian1d_stability | [FAIL](media/baseline-gaussian1d_stability.gif) | [FAIL](media/candidate-gaussian1d_stability.gif) |
| 2 | five_word_joint_hold | [PASS](media/baseline-five_word_joint_hold.gif) | [PASS](media/candidate-five_word_joint_hold.gif) |
| 2 | trajectory | [FAIL](media/baseline-trajectory.gif) | [FAIL](media/candidate-trajectory.gif) |
| 2 | residual_student | [FAIL](media/baseline-residual_student.gif) | [FAIL](media/candidate-residual_student.gif) |
| 2 | vector_unequal_mass | [FAIL](media/baseline-vector_unequal_mass.gif) | [FAIL](media/candidate-vector_unequal_mass.gif) |
| 2 | vector_unequal_width | [FAIL](media/baseline-vector_unequal_width.gif) | [FAIL](media/candidate-vector_unequal_width.gif) |
| 2 | vector_anisotropic | [FAIL](media/baseline-vector_anisotropic.gif) | [PASS](media/candidate-vector_anisotropic.gif) |
| 2 | grid100 | [FAIL](media/baseline-grid100.gif) | [FAIL](media/candidate-grid100.gif) |
| 2 | rotated100 | [FAIL](media/baseline-rotated100.gif) | [FAIL](media/candidate-rotated100.gif) |
| 2 | staggered100 | [FAIL](media/baseline-staggered100.gif) | [FAIL](media/candidate-staggered100.gif) |

## Numerical repairs and failures

The anisotropic task is the sole repaired gate. Mass TV improves **.195964 → .012777**, normalized SW1 **.197952 → .055539**, minimum component eigen ratio **.204013 → .605963**, and covariance error **.449625 → .397653**. Its terminal passing suffix is **0 → 21**, above the original requirement of five. This is a sustained improvement within the measured compound package.

The main width hypothesis fails strongly. Covariance error worsens **6.287559 → 21.078472**, with suffix **0 → 0**. Mass TV improves **.291504 → .037842** and SW1 **.295865 → .078245**, so allocation/distance improvement does not establish local shape repair. Minimum eigen ratio reaches .549177, yet the original covariance gate still fails. Unequal mass similarly improves covariance **3.691653 → .420369** and TV **.070654 → .019707**, but minimum eigen ratio remains **.122499 < .15** and suffix zero; it is not a repaired gate.

Two-pole movement remains above .3 (**.958502 → .703582**), but final critic slope worsens **.952346 → 1.151975 > 1** and suffix **17 → 0**. The earlier Phase 1 transport ablation already linked added transport to the critic's operating trajectory; this new finite objective also changes that trajectory. No critic-path measurement here uniquely separates objective geometry, shared parameters, normalization or CPU rounding.

Unused-token hold falls **.990721 → .496022 < .85**, while concept movement remains passing at .992000. The original fixture trains eight repetitions of `[0,1]`; its row-centered real coordinate variance is exactly zero. The float32 normalizer becomes **2^-23**, multiplying mean-coordinate squared error by **2^22 = 4,194,304** before weight 1. From the fixture's initial concept output `[0,0]`, the declared cost is **2^21 = 2,097,152**, matching the saved maximum auxiliary loss. The checkpoint records 200 calls, loss sum **30,237,111.853775**, and relative marginal residual zero. [Exact fixture/checkpoint proof](normalization-analysis.json).

That zero residual describes marginal consistency on duplicate rows; it does not certify intended identity, useful loss scale, a measured parameter force or converged population transport. Entropy temperature .1, scalar weight 1 and the variance-floor amplification are different quantities. Amplification can plausibly overwhelm the original masked hold objective through the shared student, but this package comparison does not isolate that cause.

Ring remains PASS with suffix **26 → 52**, covariance **.496567 → .476052** and TV **.065186 → .046631**. AE hold also remains PASS. Trajectory identity error **.239862 → .251957** and residual-student identity error **.061036 → .061269** remain failures; residual success and wrong-pad rates both remain .5.

Both Gaussian acquisition tasks pass, but both complete retention/shift tasks fail. The candidate improves final KS **.320623 → .065952** and width ratio **.662330 → 1.040055**, without repairing the full temporal gate. An improved endpoint is not sustained retention evidence.

All three native 100-mode gates remain FAIL with zero passing checks in their final-five accuracy summaries. The independent 100,000-draw holdout precision changes **.24072 → .25175** on grid, **.25552 → .18942** on rotated, and **.30168 → .26729** on staggered. TV improves on all three, but concentrated native density remains poor. The deterministic 128-row auxiliary panel represents only 6.25% of each original native batch; finite-panel geometry and shared-network response remain plausible limitations. These holdouts remain separate from the scheduled 20,000-draw observations.

Both word producers complete all **20,001 updates** and pass their full generation/inverse contract; both own holds pass every required check. The selected baseline word run has **740.308783 certified execution seconds**, and the candidate **761.409200 seconds**. Shared GPU/host contention and the compound auxiliary/backend delta prevent optimizer speed ranking.

## Finite solver counters and checkpoint comparison

[Certified counters](sinkhorn-counters.json), reproduced by [analyze_saved.py](analyze_saved.py), confirm exactly 72 total mapping rounds per active call: three costs times 24 rounds. All sixteen candidate checkpoints have counters, with no negative-loss calls. Absence of observed negatives proves neither exact divergence positivity nor convergence.

Maximum relative marginal residual reaches **2.211722** in Gaussian acquisition/continuation, **1.193287** in unequal mass, **.580589** in width, **.448450** in anisotropic and **.741184** in word acquisition. Native maxima are **.245796 / .229563 / .221759** on grid/rotated/staggered. These are historical maxima across all three costs, not final residuals. This finite approximation therefore cannot be described as an exact OT solution. The software derivative check concerns the implemented finite scalar, not its equality to converged free energy.

The saved audit verifies **1,207 scientific files**, shared initial model/prior metadata and named stream bindings across all sixteen pairs. **Fifteen** pairs have exactly equal complete consumed non-evaluation streams and actual target-batch identities. Two-pole initialization remains source-bound under its explicit fixed fixture rather than claiming a missing separate initial-tensor dump.

All four executed Gaussian/word continuations restore their own fully passing producer exactly, without history reset. Gaussian prefixes are 1,000 updates in both arms. Word holds restore each producer's earliest confirmed state after that producer completes its full budget: baseline prefix **834**, candidate prefix **2,501**, followed by 4,000 updates. Their cumulative endpoints are **4,834 / 6,501**, so matching consumed stream histories is explicitly **unverified** for that pair. No other arm's checkpoint is borrowed. Hold counters include their restored prefix and duplicate serialized copies are counted once.

## Accounting, interruption repair and software evidence

Total charged worker time is **11,763.451632810375 seconds**: baseline **9,514.13474305224**, candidate **2,249.3168897581363**. There are **35 paid attempts**, comprising 32 final completed measurements and three separately authorized execution repairs. Campaign/arm paid ceilings remain 48,000/24,000 seconds, with zero outstanding reservation. Original paired full reservations are 45,840 seconds; the three original-allowance repairs add 8,100, giving **53,940 cumulative executed allowances**. These allowances are separate from charged runtime.

The user authorized infrastructure recovery after reboot and documented host memory/I/O pressure. No completed quality failure was retried, tuned or regraded. All predecessor identities, hashes, reasons and charges remain in the [final attempt history](phase3-results.json) and [authorization receipt](publication-receipt.json):

| Baseline predecessor | Original task | Original status | Charged seconds |
| --- | --- | --- | ---: |
| `35131b71037848818236a1cb3e86dc66` | grid100 | INCOMPLETE, interrupted | 3,453.8705615997314 |
| `3d777395319c4583aae36c3458ad4987` | rotated100 | INCOMPLETE, interrupted | 3,309.460136413574 |
| `48c6c0a8293d4b8ebbe74b271e0c80d6` | five_word_joint_smoke | INCOMPLETE, timeout | 900.7708788529271 |

The timeout's complete charge, including its overrun beyond the original 900-second allowance, is preserved. The linked final native attempts remain FAIL; the final word producer is PASS. These repairs are not independent replications or new scientific configurations.

Before paid admission, **56 targeted software/orchestration checks passed in 14.83 seconds**, within the separate 300-second software allowance. They cover null loss/force, finite-scalar derivatives, row/RNG ownership, inactive checkpoint packets and exact active continuation. No additional training, software test, sampling or worker was launched for publication. JUnit, logs, checkpoints, full source snapshots and per-update streams stay in `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/sinkhorn`; compact evidence and all 32 GIF bytes are committed here.

## Recommendation and reproduction

Stop this exact global revision. Keep the sustained anisotropic improvement as source-bound research evidence, alongside the width falsification, two Tier 1 regressions and all native/temporal failures. A later theory would need a separately bounded proposal that handles constant-panel normalization and local shape without borrowing these partial successes. This result authorizes no extra experiment, tuning, seed study, merge or default change. The repository's single technique inventory and archived qualification evidence are unchanged.

Saved publication used the original frozen CLI through the root's report-only adapter (`f90591b85383052e803903088b775f4e9e95d8e1`, SHA256 `70302ffba5fbaf673375d3e4877363edf8ac80026b67cd9f872beb5ebe9c88e5`). Reproduction requires the pinned frozen source and retained archive; missing evidence blocks verification rather than authorizing a rerun. Read counters without model construction or draws:

```sh
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python   reports/forge/bcap-moonshot-sinkhorn/analyze_saved.py   --repository /home/martyn/dev/ParticleGAN-bcap-moonshot-sinkhorn   --publication /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/sinkhorn/publication   --output /your/archive/sinkhorn-counters.json

tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/sinkhorn/logs/driver.log
```

[Draft implementation and evidence PR #379](https://github.com/255BITS/ParticleGAN/pull/379), stacked on `integration/bcap-direction-transport`. No merge requested.

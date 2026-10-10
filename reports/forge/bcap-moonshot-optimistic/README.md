# Raw-gradient optimism: completed paired diagnostic

Retain the Phase 2 incumbent and stop this exact optimistic revision. The control preserves **6/6 original Tier 1 gates**; the candidate preserves **3/6**, regressing Gaussian, ring and word acquisition. Across the sixteen original questions, baseline is **7 PASS / 9 FAIL** and candidate **3 PASS / 11 FAIL / 2 BLOCKED**. There are **zero repaired failures**, three retained passes, three gate regressions, eight persistent failures and two unresolved hold comparisons. The candidate is ineligible under the [replacement criterion](https://github.com/255BITS/ParticleGAN/blob/bac09130e58ef075ccd068d8d03bc3b5529ce27b/reports/forge/bcap-three-phase/README.md).

[Final metrics, study decisions and all attempt histories](phase3-results.json) · [paired state audit](phase3-audit.json) · [paired classification and costs](paired-summary.json) · [checkpoint history/counter receipt](optimism-counters.json) · [source/archive identities](artifacts.json). These are research diagnostics, with no ordinary qualification, default promotion or 21/21 claim; eleven original Tier 2 questions are outside scope.

| Original tier | Task | Baseline | Candidate | Full-gate comparison | Actual training |
| --- | --- | --- | --- | --- | --- |
| 1 | gaussian1d_smoke | PASS | FAIL | regressed | [baseline](media/baseline-gaussian1d_smoke.gif) / [candidate](media/candidate-gaussian1d_smoke.gif) |
| 1 | two_pole | PASS | PASS | retained pass | [baseline](media/baseline-two_pole.gif) / [candidate](media/candidate-two_pole.gif) |
| 1 | unused_token_hold | PASS | PASS | retained pass | [baseline](media/baseline-unused_token_hold.gif) / [candidate](media/candidate-unused_token_hold.gif) |
| 1 | ae_gan_hold | PASS | PASS | retained pass | [baseline](media/baseline-ae_gan_hold.gif) / [candidate](media/candidate-ae_gan_hold.gif) |
| 1 | ring16_acquisition | PASS | FAIL | regressed | [baseline](media/baseline-ring16_acquisition.gif) / [candidate](media/candidate-ring16_acquisition.gif) |
| 1 | five_word_joint_smoke | PASS | FAIL | regressed | [baseline](media/baseline-five_word_joint_smoke.gif) / [candidate](media/candidate-five_word_joint_smoke.gif) |
| 2 | gaussian1d_stability | FAIL | BLOCKED | unresolved | [baseline](media/baseline-gaussian1d_stability.gif) |
| 2 | five_word_joint_hold | PASS | BLOCKED | unresolved | [baseline](media/baseline-five_word_joint_hold.gif) |
| 2 | trajectory | FAIL | FAIL | persistent failure | [baseline](media/baseline-trajectory.gif) / [candidate](media/candidate-trajectory.gif) |
| 2 | residual_student | FAIL | FAIL | persistent failure | [baseline](media/baseline-residual_student.gif) / [candidate](media/candidate-residual_student.gif) |
| 2 | vector_unequal_mass | FAIL | FAIL | persistent failure | [baseline](media/baseline-vector_unequal_mass.gif) / [candidate](media/candidate-vector_unequal_mass.gif) |
| 2 | vector_unequal_width | FAIL | FAIL | persistent failure | [baseline](media/baseline-vector_unequal_width.gif) / [candidate](media/candidate-vector_unequal_width.gif) |
| 2 | vector_anisotropic | FAIL | FAIL | persistent failure | [baseline](media/baseline-vector_anisotropic.gif) / [candidate](media/candidate-vector_anisotropic.gif) |
| 2 | grid100 | FAIL | FAIL | persistent failure | [baseline](media/baseline-grid100.gif) / [candidate](media/candidate-grid100.gif) |
| 2 | rotated100 | FAIL | FAIL | persistent failure | [baseline](media/baseline-rotated100.gif) / [candidate](media/candidate-rotated100.gif) |
| 2 | staggered100 | FAIL | FAIL | persistent failure | [baseline](media/baseline-staggered100.gif) / [candidate](media/candidate-staggered100.gif) |

The candidate changes only **`optimizer_optimism="raw_gradient"`**. The control has zero recipe overrides and inherits the exact incumbent. Neither enables transport, protected projection, finite critic guards or CPU SVD. Seed 0, public deterministic initialization, each original architecture/data/prior/sampling law, seen batch sequence, schedule horizon, update allowance, numerical gates and evaluation cadence remain fixed. The two-pole fixture retains its explicitly separate stored-weight/zero-particle contract.

For each owned network parameter application, use `h_t = 2*g_t - g_previous`, then the existing smoothed FullDualNorm map; the first application uses `g_t`. History refers to the previous update of that parameter, and sampled prior rows use their previous own application. Duplicate sampled IDs consume the aggregate autograd row gradient once; unsampled rows retain their history. The raw forecast precedes polar/bias/row normalization, so changed orientation survives magnitude normalization. Checkpointed histories, Boolean row masks and counters resume exactly, with no additional draws, data requests, forwards or backwards. The original word host owns encoder parameters in a generator-labelled group; its actual encoder histories are verified by the software resume control.

The [preregistered hypothesis and prior negatives](protocol.md) predicted unequal-mass terminal covariance error <=0.85 while preserving six Tier 1 gates. The observed **0.968191 exceeds its 0.85 bound**, satisfying the scalar falsifier. The generic study decision remains `incomplete` because both holds lack passing candidate producers; its recorded scalar falsifier is nevertheless satisfied. Its descriptive `request_missing_evidence` action authorizes no extra work. No quality failure was retried, and blocked holds remain blocked.

| Metric (full verdicts above remain authoritative) | Baseline | Candidate |
| --- | ---: | ---: |
| Gaussian endpoint KS | 0.0718422 | 0.0611164 |
| Gaussian endpoint mean error / sigma | 0.125898 | 0.0207773 |
| Two-pole median slope | 0.952346 | 0.203126 |
| Two-pole movement | 0.958502 | 0.957738 |
| Unused-token hold | 0.990721 | 0.983171 |
| AE hold | 0.0119253 | 0.00750767 |
| AE reconstruction MSE | 0.00285519 | 0.00564155 |
| Word endpoint modes | 5 | 3 |
| Word reconstruction exactness | 1 | 0 |
| Word reconstruction NLL | 3.4861e-05 | 2.92751 |
| Trajectory identity MSE | 0.239862 | 0.245526 |
| Residual identity MSE | 0.061036 | 0.0616327 |
| Unequal-mass covariance error | 3.69165 | 0.968191 |
| Unequal-mass minimum mass ratio | 0.20752 | 0 |
| Unequal-mass minimum eigen ratio | 0.00907291 | 0 |
| Unequal-width covariance error | 6.28756 | 15.6241 |
| Anisotropic covariance error | 0.449625 | 2.59895 |
| Grid100 precision | 0.24072 | 0.21829 |
| Rotated100 precision | 0.25552 | 0.23215 |
| Staggered100 precision | 0.30168 | 0.27515 |

Gaussian finishes all 1,000 updates and 24 paired observations. Baseline confirms states 375 and 875. Candidate has one primary pass at update 959, whose independent same-state confirmation fails the standard-deviation upper bound: **1.219538 > 1.2**. Neither endpoint alone satisfies the KS <=0.05 bound. Candidate Gaussian retention is BLOCKED because its own producer FAILs. Baseline retention itself FAILs (stationary 2/72, reacquisition FAIL, shifted hold 0/24, frozen checks 0/48); no candidate retention repair can be claimed.

Ring finishes all 1,600 updates and 96 observations. Baseline has 26 passing observations and a 26-check terminal suffix, confirming at update 1,250. Candidate has five passing observations but a **2-check suffix against the required 5**. Its late covariance errors .873230 at update 1,550 and .860429 at 1,567 exceed .85. All final endpoint bounds pass, including covariance .834902, HQ .892822 and mass TV .093262; an endpoint rescue would change the frozen question. [Saved bound explanations](regression-details.json) retain the original thresholds.

Word finishes all 20,001 updates in 819.763 charged seconds. Candidate has **0/24 passing and confirmed observations**, endpoint modes 3, quality fraction .801758, mass TV .417188 and reconstruction exactness 0. Its word hold is BLOCKED and borrows no checkpoint. Baseline word hold PASS has 25/25 confirmed checks (restore plus 24 observations), 4,000 additional updates from its own acquired prefix 834, and exact restoration with no history reset. Only baseline holds have own passing, full-budget producers.

All three vector gates remain FAIL with zero passing terminal suffix. Unequal mass has minimum mass/eigen ratios zero despite HQ .983887; majority-cluster precision cannot replace missing rare density. Width and anisotropic shape errors worsen. All three native runs finish their full 7,000-update budgets and FAIL. The lower two-pole slope and AE hold are retained passes, with no repair credit; both conditional identity failures persist.

The [saved audit](phase3-audit.json) PASS verifies the executed frozen source and admission, matching initial models/priors and named training-stream bindings/states for **14 complete equal-budget paired runs**. Recorded batch digests are compared where available. The two baseline-only hold comparisons cannot establish candidate consumption parity; their missing measurements are explicit. Own baseline checkpoint file/state hashes and exact restore continuity are retained. Publication uses certified saved observations, adds **zero updates and zero sampling draws**, and commits all **30 actual-training GIFs** (8,190,810 bytes) with [frame/input/render hashes](media/index.json).

| Candidate final checkpoint | Owned parameter applications / extrapolated | Sampled rows / extrapolated |
| --- | ---: | ---: |
| gaussian1d_smoke | 13,000 / 12,987 | 100,934 / 100,678 |
| ring16_acquisition | 20,800 / 20,787 | 161,390 / 161,134 |
| five_word_joint_smoke | 380,019 / 380,000 | 100,005 / 100,000 |
| vector_unequal_mass | 15,600 / 15,587 | 121,074 / 120,818 |
| grid100 | 119,000 / 118,983 | 13,625,515 / 13,605,515 |

The [read-only extractor](export_counters.py) audits all **30 final checkpoint dictionaries**, checking file/state SHA-256, history availability, finite raw gradients, Boolean row ownership and consistency between consumed parameter steps and application counters. Disabled control checkpoints contain no optimism history. Gaussian/ring each have 256 first-consumption rows; word has five and grid 20,000. This proves actual history consumption on every owned role. Optimizer `steps` count optimizer calls; generator/prior can share one optimizer. Hold counters include their producer prefixes, so summing checkpoint totals would double-count work.

| Paid accounting | Seconds |
| --- | ---: |
| Baseline, including both predecessors | 6212.748216 |
| Candidate | 1734.367239 |
| Total paid | 7947.115455 |
| Selected complete baseline runs | 1860.986330 |
| Original paired full reservation | 45,840 |
| Original paid ceiling | 48,000 |
| Cumulative executed task allowances | 49,440 |

Thirty-two paid attempts comprise 30 complete final measurements plus exactly two explicit infrastructure repairs. Original baseline word timeout `99cd2b12eb4a40a2b7804d8115d134ba` retains its **900.265826** charged seconds and INCOMPLETE status. Original grid interruption `e939011f991640c3a7676abc9c595e41` retains **3,451.496060** charged seconds and INCOMPLETE status. The user-authorized replacements preserve frozen source, seed, recipe, gates and original 900/3,600-second allowances. The cumulative executed allowances include those retries and exclude the two blocked candidate holds; they are not paid spend. All charges stay under the unchanged ceiling. [Authorization/predecessor result hashes](artifacts.json) and every original status/certificate/cost remain visible in [attempt history](phase3-results.json). Shared GPU contention and the host reboot preclude an isolated speed ranking.

Software verification passed **91 focused checks plus 78 integration compatibility checks**, including **42 exact inactive-state pairs** against archived develop. Recorded software execution is <=60 seconds under the 300-second allowance; [software receipt](software.json) and [compatibility receipt](compatibility.json) retain test identities and external JUnit/log hashes. Two pre-freeze development failures caught Boolean-mask casting on load and the original encoder role label; both were corrected before paid spend. Saved publication took 67.066 seconds with no training or sampling, recorded separately in [its command receipt](publication-command.json).

The [optimistic-GAN paper](https://arxiv.org/abs/1711.00141) and [coherent-game analysis](https://arxiv.org/abs/1807.02629) motivate anticipating rotational dynamics. Their convergence assumptions do not hold for this alternating stochastic, sampled-row, normalized nonconvex implementation. The moving-critic archived negative was an evaluator-only payoff forecast; an earlier raw-gradient-before-Adam package also failed but used accumulating second-moment preconditioning. The substantive difference here is anticipation before memoryless smoothed spectral polar/row normalization. No optional backend/projection/transport change is mixed into this comparison. Cross-archive initialization and prior differences still limit attribution.

Counters establish activation, not damping of rotation: no gradient-angle/cycle telemetry or Jacobian spectrum was registered. Nonlinear normalization, amplified minibatch differences, stale sampled-row histories and nonrotational density/critic-information failures remain competing causes. This exact package fails its numerical prediction and preservation requirement. Retain the incumbent, keep the [draft PR #378](https://github.com/255BITS/ParticleGAN/pull/378) as a source-bound research review, and do not adopt or automatically continue this revision.

Frozen scientific source: **`5cb95a39f799a493efd731c0c49960b8f7fa8bb8`**, digest `84b179b329810589cf248c01ef7453523a0081819f1bfc14a910f0ba5b1ae0d6`. Fork publication `9005ed73a09a740174f590fbfd543440935040eb` preserved measured incumbent source `0f9787fa3f164f6d2034144714f8e7ba7ab75bbf` and digest `d944540c70e280d8f981367368792920a70bbec0d919d681b4bdf0770a96f59f`. [Registration](registration.json), [spec](spec.json), archived request revisions, all sixteen original task contracts, Python/package runtime and certificate hashes remain linked in the receipts. The report-only publisher adapter is from `f90591b85383052e803903088b775f4e9e95d8e1` with SHA-256 `70302ffba5fbaf673375d3e4877363edf8ac80026b67cd9f872beb5ebe9c88e5`; it labels known missing-producer cells BLOCKED without modifying any measured grade.

Raw artifacts and easy-tail logs remain under `/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic`. Reproduce saved publication using the exact frozen-worktree command in [publication-command.json](publication-command.json); reporting commits do not alter the executed source. Export histories through the public saved-state format with:

```sh
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-moonshot-optimistic/export_counters.py --results reports/forge/bcap-moonshot-optimistic/phase3-results.json --output /tmp/optimistic-counters.json
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic/logs/driver.log /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/moonshots/optimistic/logs/publish.log
```

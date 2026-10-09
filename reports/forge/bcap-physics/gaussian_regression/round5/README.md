# Gaussian regression: finite acceptance versus direction

**Removing finite acceptance preserves the rare/broad repair but does not restore Gaussian retention.** Direction plus local-v2 finishes 3 PASS / 2 FAIL, as do local-v2 and strict finite plus local-v2. Direction reaches endpoint KS .033669 within the .05 bound, while only 6/72 stationary checks and 2/24 shifted-hold checks pass and deadline reacquisition fails. Its unequal-width covariance regresses. No global repair or ordinary qualification follows.

This completed three-arm bounded diagnostic is [PR374](https://github.com/255BITS/ParticleGAN/pull/374), stacked on [PR368](https://github.com/255BITS/ParticleGAN/pull/368), ready and unmerged. Each arm uses one global trainer configuration, protocol seed 0, original tasks and complete gates. Parent owns the single current technique leaderboard; this is a scoped comparison table.

| Original task | Exact local-v2 | PR367 direction + local-v2 | PR368 strict finite + local-v2 |
| --- | --- | --- | --- |
| gaussian1d_smoke | **PASS**, first confirmation 167; KS 0.020606 | **PASS**, first confirmation 375; KS 0.032341 | **PASS**, first confirmation 459; KS 0.093780 |
| gaussian1d_stability | **FAIL**, stationary 28/72, deadline FAIL, hold 11/24; KS 0.060653 | **FAIL**, stationary 6/72, deadline FAIL, hold 2/24; KS 0.033669 | **FAIL**, stationary 8/72, deadline FAIL, hold 1/24; KS 0.241544 |
| vector_unequal_mass | **PASS**, full covariance 0.522577; suffix 5 | **PASS**, full covariance 0.386670; suffix 9 | **PASS**, full covariance 0.350241; suffix 9 |
| vector_two_broad | **PASS**, full covariance 0.226564; suffix 24 | **PASS**, full covariance 0.331816; suffix 9 | **PASS**, full covariance 0.190535; suffix 24 |
| vector_unequal_width | **FAIL**, full covariance 2.480438; suffix 0 | **FAIL**, full covariance 3.590511; suffix 0 | **FAIL**, full covariance 2.526261; suffix 0 |

The registered endpoint KS forecast is observed for direction. The complete continuous-learning hypothesis fails: all 72 stationary checks, the deadline's last five reacquisition checks, and all 24 shifted-hold checks are required. A terminal pass or attractive endpoint cannot replace that rule. Smoke independently confirms an earlier scheduled state while completing 1,000 updates; stability restores each arm's own completed 1,000-update checkpoint, without resetting optimizer history or streams. First confirmation 167 / 375 / 459 therefore describes acquisition timing, not the checkpoint step restored or retention.

## Density repair and guardrails

| Unequal-mass endpoint | Local-v2 | Direction | Finite |
| --- | --- | --- | --- |
| component_covariance_error | 0.522577 | 0.386670 | 0.350241 |
| component_min_eigen_ratio | 0.366567 | 0.797882 | 0.581130 |
| mass_tv | 0.016426 | 0.018623 | 0.018867 |
| hq | 0.969727 | 0.950928 | 0.978760 |
| min_mass_ratio | 0.899564 | 0.877028 | 0.877028 |
| max_component_spill | 0.140351 | 0.111111 | 0.126984 |

All three retain the rare task's complete five-terminal-check gate and broad-mixture PASS. Direction extends the rare passing suffix from 5 to 9 and reduces covariance, but slightly worsens mass TV and HQ; finite retains covariance .350241 and suffix 9. These are descriptive deterministic comparisons, not statistical superiority or a pooled family winner.

Unequal width remains FAIL in every arm. Direction worsens uncensored full covariance 2.480438 → 3.590511, against the unchanged .85 bound. [Temporal diagnostics](temporal-diagnostics.json) retain every endpoint failure, covariance per component, core covariance, spill, mass and actual sample counts. All four components remain represented; lower averaged or core covariance cannot replace the full served-law gate. Gaussian draws remain finite, and failure includes CDF shape and temporal instability rather than an exception or borrowed checkpoint.

## What the finite layer changes

| Task | Direction conflicts / steps | Finite conflicts / steps | Finite backtracking reductions / rejections |
| --- | --- | --- | --- |
| gaussian1d_smoke | 200/1000 | 195/1000 | 14 / 0 |
| gaussian1d_stability | 1292/6000 | 1407/6000 | 151 / 0 |
| vector_unequal_mass | 58/1200 | 66/1200 | 52 / 0 |
| vector_two_broad | 158/1200 | 199/1200 | 103 / 0 |
| vector_unequal_width | 10/1200 | 18/1200 | 29 / 0 |

Stability counters include the 1,000-update smoke prefix. Direction always applies the unchanged common-descent blend at full scale and never calls a finite evaluator. Strict finite backtracks on conflicting proposals to certify same-batch protected adversarial decrease. All 1,407 Gaussian conflicts are accepted with zero rejection, 151 reductions, and minimum scale .25. Direction takes a different path after scaling is removed; lower endpoint KS does not identify a unique historical cause. Endpoint aggregate activity plus scheduled quality curves cannot assign every later quality failure to a particular conflict event.

The exact blend mixes the nonascent projection and negative normalized protected gradient. Its local directional interpretation is related to [Sener and Koltun](https://arxiv.org/abs/1810.04650); finite sufficient decrease follows an [Armijo-style construction](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-s.pdf). Their deterministic properties do not establish distribution convergence in this moving stochastic game. No transport term becomes another protected loss and no target labels or oracle geometry enter training.

## Source, protocol and execution

All arms execute commit `8728dac82a318124c1e4f761454ae17893c72c6d`, digest `21e4a3278bb8bd3e7a0386fce8b7ba5f4fe27610f5397e75b3dc3e07f1769552`. [Audit](audit.json) verifies 1204 frozen files, 471 current scientific files, all 15 final certificate sets and actual-training GIFs, and 8 bitwise archived-control comparisons. Local-v2 and finite reproduce round-four smoke/stability/rare/broad observations, samples, model/optimizer tensors and consumed streams; archived results are context, not reused qualification. Width is also reproduced as a matched new-source control in all arms.

The [four-state saved restoration](prior-evidence.json) preceded preregistration and adds zero updates or sampling. The direction module is byte-exact PR367; strict and local-v2 modules are byte-exact PR368. All 67 original task and qualification/telemetry snapshots remain unchanged. Each task's actual initial model hashes, consumed context and trainer streams, and batch digests match across all three arms. Vector batches are reconstructed with disposable RNGs from the recorded initial seed and checked against the exact final data-stream state; the frozen adapter did not log their digest. Read-only replay adds no training or served-model draws. Archived array parity excludes only opaque training-state hashes containing unused ambient CPU/CUDA states; actual outputs, metrics, model/optimizer tensors and every consumed named stream are compared. Constructor, data, prior/noise and evaluation streams are isolated and checkpointed. The frozen Gaussian/vector adapters retain their actual same-real-tensor D/G reuse.

The exact saved BCAP winner is nonsaturating, full DualNorm smoothing .001/momentum0/per_offset, G .012, D .018, prior .030, constant floors1, cap/coefficient1 every update, zero additive training noise and clean/live scoring. Bare `get_recipe("bcap")` is historical Adam. Local-v2 has global/local weights 1 and 32 projections. Only geometry mode differs: none, direction_blend, strict_progress. Gaussian uses learned uniform 256-MoG sigma .1; vector priors retain sigma .025, fixed uniform weights and learnable locations. Architecture, target law, initializer, actual batches, sampling, update budget and evaluation cadence are fixed within each task.

All 15 final jobs complete: 9 PASS / 6 FAIL, no final BLOCKED, INCOMPLETE or INVALID. There are 16 paid attempts including one retained strict-smoke timeout. Main paid worker wall time is **2504.600769 seconds**; initial full reservations 18,360 plus retry 120 total **18,480**. Queue ceiling 21,180 plus saved restoration 120 and bounded software/scorer 300 allowances equals the 21,600 track cap. Certified complete arms account for 28,800 updates; the discarded timeout has an uncertified partial prefix, its last logged observation 42. Its 121.149060-second INCOMPLETE receipt remains unchanged in [execution history](execution-history.json).

Transient host RAM contention then blocked pending work without paid launches. The unchanged-source recovery uses physical GPU 1/logical cuda:0, five-second queue polling and pre-import BLAS/OpenMP threads 1; scientific Torch thread budget 1 remains unchanged. Bitwise archived parity and complete stream checks verify that this execution recovery changed no compared training trajectory. Contention and startup time are accounting, not optimizer-speed claims. All subscriptions are concluded, reservations zero, and workers/watchers stopped. Bulk stdout, JSONL, JUnit, checkpoints and tensors remain on the artifact drive.

The focused software/protocol suite passes 58 checks, covering CPU/CUDA float32/64, public recipe construction, checkpoint resume, inactive parity, nonlinear overshoot without evaluator calls, unchanged critic/streams and adversarial-only protection with both transport terms. [Oracle/destructive controls](scorer-controls.json) validate full Gaussian and vector scorers. Publication recomputes 648 saved primary metric sets and adds no updates or sampling. Forge validates; summaries-only compilation/check preserves published qualification/telemetry. Later files are report/audit helpers, not another measured trainer revision.

## Actual-training media and decision

| Task | Local-v2 | Direction | Finite |
| --- | --- | --- | --- |
| gaussian1d_smoke | [GIF](media/local-gaussian1d_smoke.gif) | [GIF](media/direction-gaussian1d_smoke.gif) | [GIF](media/finite-gaussian1d_smoke.gif) |
| gaussian1d_stability | [GIF](media/local-gaussian1d_stability.gif) | [GIF](media/direction-gaussian1d_stability.gif) | [GIF](media/finite-gaussian1d_stability.gif) |
| vector_unequal_mass | [GIF](media/local-vector_unequal_mass.gif) | [GIF](media/direction-vector_unequal_mass.gif) | [GIF](media/finite-vector_unequal_mass.gif) |
| vector_two_broad | [GIF](media/local-vector_two_broad.gif) | [GIF](media/direction-vector_two_broad.gif) | [GIF](media/finite-vector_two_broad.gif) |
| vector_unequal_width | [GIF](media/local-vector_unequal_width.gif) | [GIF](media/direction-vector_unequal_width.gif) | [GIF](media/finite-vector_unequal_width.gif) |

**Retain local-v2 as the reference; do not adopt either composition as a global replacement.** Direction retains the scoped rare/broad repair and removes finite evaluator work, but it fails retention and worsens the width guardrail. Finite retains its same-batch certificate and slightly better rare covariance, with the original Gaussian regression. Neither finite removal, batch descent, acquisition nor endpoint KS alone resolves continuous learning. Stop this comparison without a sweep, new seed, continuation, merge or promotion. Any further question should inspect the saved Gaussian trajectory and competing adversarial/empirical transport fields rather than tune finite acceptance from this endpoint.

[Final metrics](results.json), [compact receipts](receipts.json), [matched provenance](provenance.json), [concluded studies](readouts.json) and [media identities](media/index.json) preserve exact evidence. Read-only reproduction:

```sh
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/gaussian_regression/round5/publish.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/gaussian_regression/round5/audit.py
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue/events.jsonl
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/logs/driver.log
```

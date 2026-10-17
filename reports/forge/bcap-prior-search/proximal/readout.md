# Initial-position proximal prior: completed search

The three tested positive strengths lose required Tier 1 gates. None reaches Tier 2, so this round establishes no Tier 2 repair and leaves the Tier 2 drift hypothesis unmeasured. Keep the BCAP leader with strength zero; stop this finite bracket without adopting a new default.

| Actual global strength | Required Tier 1 | Required Tier 1 failures | Tier 2 measured | Tier 2 status |
| ---: | ---: | --- | ---: | --- |
| 0, fresh matched leader | 6/6 PASS | none | 21/21 | 9 PASS, 12 FAIL |
| 0.1 | 4/6 PASS | Gaussian acquisition, Ring16 acquisition | 0/21 | 21 BLOCKED |
| 1 | 5/6 PASS | Ring16 acquisition | 0/21 | 21 BLOCKED |
| 10 | 4/6 PASS | AE/GAN hold, Ring16 acquisition | 0/21 | 21 BLOCKED |

There are no scientific ERROR/INCOMPLETE outcomes. The five positive-arm Tier 1 FAILs are numerical regressions relative to this branch's newly executed control. The nine control Tier 2 passes remain unresolved for every positive strength. No Tier 3 task or independent confirmation was run.

[Every task and measurement with its score/target](README.md), [compact results](results.json), [saved provenance audit](audit.json), [actual-training GIF index](media/README.md), and [terminal configuration-search registration](../../configuration-search/bcap-prior-proximal-search-v1.json) retain the original gates and executed identities. The 112 arm/task cells comprise 49 measured PASS/FAIL tasks and 63 unmeasured BLOCKED Tier 2 tasks. Blocked scores remain unavailable with their original targets; evaluator diagnostics add no gate. The goal's [current leaderboard](../../technique-inventory.md) remains the sole generated leaderboard.

## What fails

Gaussian acquisition requires at least one independently confirmed scheduled state. Endpoint KS is a separate recorded measurement: the fresh control passes acquisition even though its endpoint KS fails. Strength 0.1 has zero passing primary observations out of 24 and zero confirmed states, so its acquisition failure is directly observed.

| Strength | Confirmed states / >=1 | Endpoint KS / <=0.05 | Mean error in sigma / <=0.2 | Std ratio / [0.8,1.2] | Original acquisition gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 2 | 0.0718422 | 0.125898 | 1.06304 | PASS |
| 0.1 | 0 | 0.0782190 | 0.0867332 | 0.923900 | FAIL |
| 1 | 4 | 0.0261354 | 0.0130435 | 1.01937 | PASS |
| 10 | 1 | 0.153820 | 0.239813 | 0.879232 | PASS |

All four Ring16 endpoints retain 16/16 modes and acceptable mass TV. Every positive strength fails full-component covariance and has zero passing observations out of 96. The larger two strengths also lose high-quality mass. Coverage alone therefore hides the shape failure.

| Strength | Terminal suffix / >=5 | Modes / >=16 | HQ / >=0.85 | Mass TV / <=0.15 | Component covariance error / <=0.85 | Min eigen ratio / >=0.15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26 | 16 | 0.960938 | 0.0651855 | 0.496567 | 0.242467 |
| 0.1 | 0 | 16 | 0.862549 | 0.0642090 | 3.51422 | 0.372671 |
| 1 | 0 | 16 | 0.729492 | 0.0454102 | 3.36332 | 0.977600 |
| 10 | 0 | 16 | 0.656494 | 0.0405273 | 4.68711 | 1.66494 |

At strength 0.1, one component's covariance error is 33.7424; the full component array remains in the report. This is a local shape defect despite a smaller aggregate global covariance diagnostic. Those diagnostics have no separate qualification credit.

AE/GAN strength 10 passes its endpoint but has a terminal suffix of four, below the required five. The final required window is steps 209, 219, 230, 240, 250. At step 209, reconstruction MSE is **0.0812076 / <=0.05**; the endpoint has recovered to 0.00820117. This saved witness explains the original sustained FAIL without changing the grade.

| Strength | Terminal suffix / >=5 | Endpoint reconstruction MSE / <=0.05 | Endpoint hold / <=0.35 | Original hold gate |
| ---: | ---: | ---: | ---: | --- |
| 0 | 22 | 0.00435301 | 0.0157810 | PASS |
| 0.1 | 22 | 0.00316379 | 0.0114921 | PASS |
| 1 | 9 | 0.00562503 | 0.00950276 | PASS |
| 10 | 4 | 0.00820117 | 0.0230128 | FAIL |

The witness is bound in [audit.json](audit.json) to the original observations SHA256 `cb0068eadd900c80deceb87acb49a452f9e61f2d1c10fb4f9397c68c4c7a404b` and step-209 packet SHA256 `1ccb6a61c939e20f50753ba50ff1b1632d9ccf5c26acfd8277aa7435ba804102`. Full per-update observations stay in the archive.

## The remaining Tier 2 failures

These are the fresh strength-zero control's failures. Positive strengths have no measurements on these tasks. Native accuracy and coverage cohorts remain separately reported, including EMA/oracle diagnostics with no qualification credit.

| Fresh control task | Failed sustained gate | Selected failed numerical measurements / targets |
| --- | --- | --- |
| Gaussian stability | Stationary 2/72; reacquisition false/true; shifted hold 0/24 | Mean error 0.255238/<=0.2; std ratio 0.662330/[0.8,1.2]; KS 0.320623/<=0.05 |
| Mode hold | Terminal suffix 3/>=5 | Endpoint passes; terminal-window witnesses are in the audit |
| Unequal mass vector | Suffix 0/>=5 | Component covariance 3.69165/<=0.85; min eigen ratio 0.00907291/>=0.15; min mass ratio 0.207520/>=0.25 |
| Unequal width vector | Suffix 0/>=5 | SW1 0.295865/<=0.18; mass TV 0.291504/<=0.15; component covariance 6.28756/<=0.85 |
| Anisotropic vector | Suffix 0/>=5 | SW1 0.197952/<=0.18; mass TV 0.195964/<=0.15 |
| Overlap vector | Suffix 0/>=5 | SW1 0.255769/<=0.18; mean error 0.388766/<=0.15 |
| Image bars4 | Suffix 0/>=5 | Modes 2/>=4 |
| Image blobs4 | Suffix 0/>=5 | Modes 2/>=4; HQ 0.75/>=0.9 |
| Image intensity2 | Suffix 0/>=5 | HQ 0.6875/>=0.9 |
| Grid100 | Coverage false/true; live accuracy checks 0/5 | Live holdout precision 0.24072/>=0.97; holdout coverage modes 8/100 |
| Rotated100 | Coverage false/true; live accuracy checks 0/5 | Live holdout precision 0.25552/>=0.97; holdout coverage modes 14/100 |
| Staggered100 | Coverage false/true; live accuracy checks 0/5 | Live holdout precision 0.30168/>=0.97; holdout coverage modes 19/100 |

The native 100-Gaussian failures are substantial under the leader, but this proximal bracket never measures their response to the spring. It cannot establish whether an initial-position prior anchor solves those tasks.

## Mechanism and matched comparison

The opt-in public parameter is `Recipe.prior_proximal_strength`, default zero. After the whole ordinary prior optimizer update, including direction blend, the spring computes

`z_next = (z_base + lr_prior * strength * z_initial) / (1 + lr_prior * strength)`.

At the leader's prior rate 0.03, strengths 0.1, 1, 10 restore fractions 0.00299103, 0.0291262, 0.230769 per consumed update. The preceding DualNorm step uses the actual smoothed direction `g / sqrt(||g||² + 0.001²)`. Both the prior step and spring act on unique G-sampled update-owned rows. Duplicate sampled rows move once; unsampled rows stay fixed. The anchor is captured after the public initializer and preserved in the optimizer checkpoint. Frozen or direct generated-position policies are inactive.

This bypasses a normalized-gradient spring's loss of magnitude, but ties latent locations to their original geometry while the generator adapts. The Ring16 failures show the tested anchor strengths degrade component shape. The spring occurs after direction blend, so its final displacement may alter the protected derivative; the base diagnostic describes the preceding displacement and this round includes no final-spring derivative assay.

All arms use the exact BCAP leader: direction blend, non-saturating loss, G/D/prior rates 0.012/0.018/0.030, DualNorm smoothing 0.001 with per-offset/native-SVD behavior, no EMA or extra train/output noise, and zero legacy prior regularization/L2. Only the proximal strength changes globally. Seed 0, public deterministic initialization, task architecture/data law, capacity, sampling, frozen budgets, cadence and gates are fixed. [audit.py](audit.py) verifies original certificates, actual executed recipe/policy, initializer hashes, complete checkpoint hashes, isolated named RNG streams, task bindings, consumed update counts, and saved optimizer anchor/activation packets for all 49 rows.

Gaussian stability restores its own unchanged 1000-update acquisition prefix. Word hold restores the control's own certified acquisition prefix (834), then performs its declared 4000 new updates. Positive-arm holds are blocked. Optional clock is a separate smaller 128-update diagnostic probe (batch 8, capacity 16, latent dimension 2); it earns no prior-feature or Tier 2 credit.

Executed source is `a28c9631d59e2977616c0375e2a07c73171a9930`, scientific digest `652cc41cc17bd9c92fb79eee4e186f2e1e01036213ae1c7a02f10ce7b86b3a2a`, based on develop `0337bbbc2300fd351e77e9a222b92401137fd421`. Later publication commits contain report work. The audit confirms the fresh control's 28 gates and all archived metric values match the earlier prior-refactor control; those archived observations are diagnostic comparison only and never fill current results.

## Budget, infrastructure and validation

The original full reservation was 172080 seconds, within 45000 per arm and 180000 combined. Actual charged cost is 2597.349 seconds for control and 2600.027 seconds for the three-strength search; final reservations are zero. This is not a speed comparison.

Five initially stacked control word workers encountered demonstrated GPU context contention. The proximal cancellation `a504edff63cc4285aac6497f48accf6b` is retained and charged 367.245761 seconds. One infrastructure retry kept the same source, recipe, seed, task, 20001 updates, gates and 900-second allowance; it passed in 806.957151 seconds. There are zero scientific retries. [Common infrastructure receipt](../infrastructure.json), [MPS numerical probe](../mps_probe.json), [cross-control word audit](../control_word_audit.json), and [own repair receipt](execution-repair.json) preserve the repair.

The isolated legacy MPS configuration exposes the full GPU; all eight fixed tensor byte digests match non-MPS. The five controls' acquisition/endpoint model, optimizer, named consumed streams and observation packets agree exactly. Unused caller-global RNG snapshots differ and are explicitly excluded. Subsequent jobs use one new worker per physical GPU across the five queues. The transient CPU-only-claim resource BLOCKED state and live-job completion guard were repaired without new scientific attempts or altered task laws.

Feature acceptance previously passed 315 relevant software tests. The [64-update public-API toy](../../prior-proximal-software/README.md) passes its restoration, row ownership, finite-state and exact own-resume software metric and has an actual-training GIF; its two fidelity gates fail and grant no qualification. [Common report-helper validation](../report-helper-validation.json) retains 36 passing checks and 12 saved fixture receipts. Final audit and all 49 GIF exports use only original saved artifacts: zero added optimizer updates, evaluation draws or rescoring. Bulk logs remain tail-able at `/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/proximal/logs/`.

The terminal search publication enters summaries-only experiment memory as source-bound study/trial records with no qualification transfer or default adoption. The useful next conclusion is to prioritize alternatives that preserve component shape and all Tier 1 gates; this finite proximal bracket supplies no eligible configuration to promote.

Saved-only reproduction commands require the [proximal feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/a28c9631d59e2977616c0375e2a07c73171a9930) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.

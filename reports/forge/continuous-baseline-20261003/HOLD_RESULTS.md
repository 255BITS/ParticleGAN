# C6 persistence continuation: measured failure after the missing checks

Both named runs reached 1350 updates and **failed** the unchanged persistence gate. The five later checks are 1150, 1200, 1250, 1300, 1350; only 3/5 pass. The original 1200 results remain **original PASS / study INCOMPLETE**. This 150-update continuation is a separate measured supplement.

| Step | Phase | Projection KS (≤0.06) | Mean error (diagnostic) | Mass TV (≤0.15) | HQ (≥0.85) | Within-mode eigenratio (≥0.15) | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 900 | original acquisition | 0.030573 | 0.007647 | 0.002930 | 0.989746 | 0.806546 | PASS |
| 950 | original acquisition | 0.038210 | 0.017041 | 0.004395 | 0.987061 | 0.845684 | PASS |
| 1000 | original acquisition | 0.057709 | 0.027344 | 0.002441 | 0.989746 | 0.830472 | PASS |
| 1050 | original acquisition | 0.057518 | 0.025854 | 0.003418 | 0.989746 | 0.752366 | PASS |
| 1100 | original acquisition | 0.059292 | 0.002528 | 0.002686 | 0.994873 | 0.733380 | PASS |
| 1150 | original hold | 0.052060 | 0.028428 | 0.002686 | 0.986572 | 0.951878 | PASS |
| 1200 | original hold | 0.054444 | 0.023432 | 0.002686 | 0.992432 | 0.828480 | PASS |
| 1250 | appended hold | 0.105824 | 0.072696 | 0.001953 | 0.985107 | 0.812736 | FAIL |
| 1300 | appended hold | 0.077633 | 0.037943 | 0.001953 | 0.989014 | 0.790334 | FAIL |
| 1350 | appended hold | 0.051259 | 0.023797 | 0.001953 | 0.989502 | 0.905351 | PASS |

The recorded KS values are **0.1058236784588924 at 1250** and **0.07763309513521596 at 1300**, respectively 0.045824 and 0.017633 above the 0.06 bound. At 1350, KS recovers to 0.05125856533298834. Every other declared bound passes at all three new observations. Mass TV stays 0.001953125 and within-mode eigenratios remain 0.812736 / 0.790334 / 0.905351. These are observed distribution-shape failures rather than missing-mode or zero-width failures. Mean error increases and later declines; it is an unbounded diagnostic and does not establish the cause of KS failure.

## Fixed law and execution

The equal-mass target has means (−1,0)/(1,0), covariance 0.0625I, a 256-row/4-dimensional prior, 128-row training batches, 4096 held-out samples and seed 34002. Full recipes retain LR 0.0053125, prior multiplier 1.5 and `Recipe.total_steps=None`. The primary law uses selected public serving with **additive output noise off**; enabled DV12 latent perturbation remains. This is neither an exact 256-atom law nor the historical noisy Atlas19 protocol.

The frozen helper restores each complete 1200 state and exactly reproduces its retained arrays/metrics, then changes only the external execution cap and caller allowance to 1350. Each run appends 150 actual public updates and observes 1250/1300/1350. Full restore, checkpoint sampler parity, observer purity and unchanged original inputs are recorded. The source-owned `GANTrainer.extend_execution` changes only `max_steps` ([scientific source 8021, training.py:316](https://github.com/255BITS/ParticleGAN/blob/8021a1c50c4aff90ddea5010d368cffdc857b2f6/particlegan/training.py#L316)); the [helper](run_hold_continuation.py) preserves the recipe, data cursor and named streams. This is a continuation of these states, not a fresh 1350-step initialization or a retroactive study PASS.

## Measured equality and remaining differences

All 51 appended metric scalars match across Atlas/E22. The appended NPZ files and 12-frame GIFs have identical bytes; all 18 arrays (98,316 scalar values) match in shape, dtype and bytes. At 1200 and 1350 the 21 model tensors and 64 optimizer tensors are byte-identical across families, as are DV12 controller, birth/death, row-evidence, LR-state tensors and named/data RNG. Paired source-owned diagnostic NaNs count as missing sentinels for structural LR comparisons, not valid model values.

Whole policy/checkpoint equality does **not** follow. Global CPU/CUDA RNG, surprise memories/ratios, recipe/backend metadata and settled guard differ. At 1350 the recorded surprise ratio is 0.871102 for Atlas and 1.446095 for E22; both have 0 fires. Atlas guard critic scale changes 0.25→0.125; E22 has no guard. Effective saved critic LR nevertheless matches exactly at both endpoints (0.00597655784264745→0.005976560606208149), and sampled/model states remain matched. These snapshots do not prove every unsaved intermediate rate or action matches.

Atlas selects reference kNN because 256 rows fail its recorded finite-resolution feature-cell criterion; E22 uses its reference/DV12 path. The distinct high-population Atlas feature-cell backend is not exercised. At all three new boundaries, retained target arrays exactly match 1200. Sample-coordinate RMS changes from 1200 are 0.084493 / 0.085645 / 0.063195, with maxima 0.307558 / 0.326903 / 0.320329. These compare saved arrays, without new draws or gate rescoring. The target and evaluation seed stay fixed; the measured excursion is an output-state change rather than a new target or seed.

## Mechanism limits

Cumulative counters record no new discrete birth/death relocation, surprise fire or controller reopen during 1201–1350: moves/births stay 11, isolation moves 0, row-evidence resets 11, surprise fires 0 with an empty log, and controller reopens 0. Birth/death evaluations rise 600→675 and row-hold steps 835→943, so continuous observations/control remain active. These endpoints can exclude an observed new move/reopen/fire in the interval, but cannot identify why generator/prior dynamics crossed KS.

Learned parameters change together: maximum absolute changes are 0.071143 in G, 0.037058 in D, 0.070287 in the prior, 0.030503 in averaged G and 0.042299 in averaged prior. Controller bandwidth changes by at most 0.000262, and mobility/game-trust diagnostics also evolve. Endpoint serving says `fast`; 1250/1300 selected serving is not independently recorded. Their checkpoints and a complete per-update effective-rate/guard/row-hold/sampler-action trace are absent. Generator transport, prior transport, geometry-dependent sampling and continuous rate/row controls therefore remain coupled possible contributors, not established causes.

The extra horizon supplied all five later checks and exposed two genuine failures. A 1350 endpoint PASS cannot erase them. Passing 900–1200 and 1350 states argue against declaring this family unable to attain this measured tolerance, but do not establish persistent convergence, trainability from every initialization or a capacity theorem. Any causal intervention would require a separately named bounded diagnostic and preserve these FAILs.

## Read-only next hypothesis: adaptive serving memory

All 42 retained receipts across the four prior grids already use `serve_average=4.0` and `ema_decay=0.995`; those receipt hashes and values were checked again. Prior grids varied only LR and prior multiplier. Averaging was present. For a positive `serve_average`, the policy updates paired G/prior averages at `min(1, s/(serve_average*b))`; at both saved endpoints, the table tester has `s=1`, `b=64`, so its rate is 1/256 (0.00390625), rather than the 0.005 fallback from `ema_decay`. The policy selects those averages only when the table's last decisive result is STATIONARY (`last_decisive=-1`) on this reference backend. Both endpoints instead record `last_decisive=+1` and select fast weights. ([Scientific source 8021, policy.py:761](https://github.com/255BITS/ParticleGAN/blob/8021a1c50c4aff90ddea5010d368cffdc857b2f6/particlegan/policy.py#L761), [selection and rate at 1047](https://github.com/255BITS/ParticleGAN/blob/8021a1c50c4aff90ddea5010d368cffdc857b2f6/particlegan/policy.py#L1047))

A different positive `serve_average` could form a distinct, complete-config test of adaptive average memory while retaining family controls. It is **weak as a direct explanation or repair for these observed fast-state failures**: changing the positive magnitude does not change the stationary-selection condition or force EMA serving. Its relevance depends on the policy actually selecting averaged states. Intermediate serving selection and average-law scores are not retained; no such scores were drawn here.

The field is a public validated Recipe hyperparameter, but it is outside the current Forge `TUNABLE_FIELDS`, and the API audit override guard plus frozen family study admit no such contrast. A new named diagnostic needs explicit admission, a full effective Recipe/source binding and a capacity packet rebound to that Recipe's actual selected serving/controller/average snapshot. It must retain the original gates, target, architecture, seeds and complete-config case denominator; record selected serving, table decision and averaging rate at every scored boundary; preserve the paired G/prior average; and retain the noise-off law explicitly. If selection stays fast, report that null serving exposure. No forced EMA, hidden policy disable, borrowed capacity card, historical noisy credit, chosen winner or new budget is supplied by this proposal. ([Current boundary classification](../../../experiments/forge/boundaries.py), [API override guard](../../../benchmarks/toy_audit/api_contract.py))

## Provenance and preserved cost

Scientific source is **8021a1c50c4aff90ddea5010d368cffdc857b2f6**; all 148 scientific files were hash-checked. H2 orchestration source is **82c85cc32c43746074cb2809ae0f2aed271484bd**, helper SHA `adcfbd92b39e3affa2626965bf32101bd4ade8a0ad00c9a4e4bbd4a0df4a0b58`, frozen execution digest `429fc7d9a90a710346637da72ebe611a0a73515c05aede32c00640824fdfa760`. Original/new receipts, checkpoint/NPZ/GIF hashes/sizes and 12 decoded frames were checked without importing the scientific package or constructing models. Full recipes, runtime, endpoint summaries and raw identities are in [hold-results.json](hold-results.json).

Both runs used physical GPU1 (logical cuda:0), one CPU thread, RTX A6000, Python 3.12.13, Torch 2.13.0+cu126/CUDA 12.6 and declared GPU-memory fraction 0.2. Each had 180s acquisition plus 60s export allowance. Atlas paid 15.290890s and E22 paid 15.893738s. H1 namespace guard failures remain INCOMPLETE with 0 scientific updates and 6.818678s paid. H2 explicitly repairs only that engineering prerequisite and carries its cost: **38.003306s of the combined 480s cap**. No speed ranking or default adoption follows.

The new 12-frame GIFs retain actual old 0/150/300/450/600/750/900/1050/1200 states and append actual 1250/1300/1350 states; original nine-frame GIFs are unchanged. New GIF SHA is `b56db2674a5f785c7979ff4bfb8ae6d4c309880cb29c3f45a8a72bb16b2f7822`; appended NPZ SHA is `f49fc8c1f72e857c07e984ec54c06374b1f4acf1c431ab67a9c986137cbe0056`. These two runs confer no eight-case or 19/19 qualification. [HORIZON_AUDIT.md](HORIZON_AUDIT.md) stays the prior timing-only report; this supplement fills its formerly unknown future checks.

| Family | Result receipt SHA256 | Original 1200 state SHA256 | Continued 1350 state SHA256 |
|---|---|---|---|
| atlas | `1844252366b6b5dfa65a382736572ef236448e9320446653443ac994f1d42cd3` | `95401e3063a5f0bba29cae59835f50bd1e7f8387776944b09aceea5ba4738075` | `467455f881f843a08cfe9c143c323d216dd0714f8a018c32b62992bc6698ebd1` |
| e22 | `4c0d798b25208bf897dc0df3582ae2e96be4331e7ef965c721a85453cb8ad2b6` | `c68b983b89db276b3e1bc3352c2e176066d84e3d9eef54ac9c49c8201a1d89b8` | `45df3f3d1e5799f7be92506cffa90324e456fd95cd2bad83bb3a0435f0181426` |

Raw cohort: `/ml2/hypergan/forge-continuous-leaderboard-20261003/hold-continuation-v2`. The baseline prerequisite ledger SHA is its captured identity; the active ledger may later grow. This readout creates no models, samples, scores or updates.

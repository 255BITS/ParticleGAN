# RA4 toy missing-mode diagnosis

RA4 finishes at emitted precision0.75806 and21/25 covered modes; matched E22 finishes at precision0.71545 and25/25. This read-only diagnosis checks all20 existing saved toy checkpoints across the two runs. It uses their exact emitted mode counts, independently reproduces every saved served clean count on CPU, and checks the saved real FIFO against the original data stream. No new emitted sample, seed, action, optimizer update, source or fixture was introduced.

## Missing modes across checkpoints

Covered means at least1% accepted mass within the unchanged.09 oracle radius. It requires11 of1024 clean rows or82 of8192 emitted rows. Zero accepted rows is reported separately. Mode IDs are zero-based, in the original row-major5x5 grid from(-1,-1) to(1,1). The receipt includes initialization and100/250 checkpoints as well as the later rows below.

| Update | RA4 emitted / served clean / training coverage | Emitted uncovered modes | Served clean uncovered modes | Served clean zero modes | E22 emitted / clean |
|---:|---|---|---|---|---|
| 500 | 25 / 25 / 25 | none | none | none | 18 / 17 |
| 750 | 23 / 23 / 21 | 7, 12 | 7, 12 | 7 | 25 / 25 |
| 1000 | 23 / 22 / 17 | 12, 21 | 7, 12, 21 | 12 | 25 / 25 |
| 1250 | 22 / 23 / 21 | 7, 8, 11 | 8, 11 | 8, 11 | 25 / 25 |
| 1500 | 19 / 19 / 19 | 2, 4, 9, 13, 18, 23 | 2, 4, 9, 13, 18, 23 | 2, 4, 9, 13, 18, 23 | 25 / 25 |
| 1750 | 19 / 18 / 23 | 2, 4, 17, 18, 22, 23 | 2, 4, 9, 17, 18, 22, 23 | 4, 17, 18, 22, 23 | 25 / 25 |
| 2000 | 21 / 21 / 18 | 11, 12, 17, 22 | 11, 12, 17, 22 | 12 | 25 / 25 |

RA4 reaches all25 modes at500 and later loses different regions. E22 reaches all25 by750 and preserves them at each remaining saved checkpoint. RA4 therefore has recurrent mode loss, rather than a subset that was never reached. At1250, high emitted precision0.87646 still leaves modes7,8,11 below1% emitted mass, while the served clean table has zero accepted rows in8 and11. At2000, emitted and served clean deficits coincide in11,12,17,22. Their emitted accepted counts are78,35,68,80 and clean counts7,0,5,4. Emission perturbations can supply a few accepted points around an empty clean mode, but do not restore its1% mass.

## Target availability and eligible parent supply

Seven CPU refits use each saved training critic, its unchanged1024-row real FIFO and the checkpoint's existing CPU RNG state. Each produces64 cells and25 groups. Every cell and group has100% oracle-mode purity on its real rows. Positive real targets therefore exist for every lost mode in these reconstructions. The head geometry and last GPU generator/cache are transient and absent from checkpoints; these refits diagnose current accessibility and do not replay preceding GPU actions or discoveries.

Examples of an actual supply limit:

| Update / mode | Accepted training rows | Eligible parents in real-mode cells | Inside eligible parents | Group target | Group vacancy | Inside birth capacity |
|---|---:|---:|---:|---:|---:|---:|
| 750 / 12 | 1 | 2 | 1 | 42 | 38 | 1 |
| 1000 / 7 | 0 | 0 | 0 | 47 | 47 | 0 |
| 1000 / 12 | 0 | 0 | 0 | 29 | 29 | 0 |
| 1250 / 8 | 0 | 0 | 0 | 54 | 54 | 0 |
| 1250 / 11 | 4 | 4 | 4 | 37 | 32 | 4 |
| 1500 / 2 | 0 | 0 | 0 | 40 | 40 | 0 |
| 1500 / 23 | 0 | 0 | 0 | 47 | 47 | 0 |

Zero parent supply makes both ordinary mass and support copying inaccessible despite positive real targets and physical vacancies. Small supply also limits recovery because every parent can be used only once per reaction. At1250, mode8 has zero training rows even assigned to its nearest oracle mode. Other zero accepted modes may still have nearby unsupported rows, which cannot supply the required p>Q copy parents. This is consistent with the intended copy-parent restriction. Count discovery power alone cannot supply a missing eligible parent.

At the final training state, modes2,6,8,12,13,18,20 are below1% accepted mass. Their inside birth capacities are2,0,5,7,0,0,0. Mode13 has zero eligible parents and target46; mode20 has zero eligible parents and target50; mode18 has one eligible parent but zero inside parents and target48. These current supply restrictions affect the training actor even though the served average still covers modes13,18,20.

## Serving averages change the observed missing set

Checkpoint models G/prior store the training iterate. The existing serve_average=4 policy serves ema_G/ema_prior whenever the table tester's last decisive verdict is stationary. RA4 uses this average at every saved checkpoint from750 onward. This behavior is explicit in GANTrainer.state_dict and _serve_apply, and CPU EMA forwards exactly reproduce all saved clean counts. It is not a checkpoint mismatch.

For the final served deficits, training parents are present:

| Mode / centre | Served clean accepted | Training accepted | Eligible / inside parents | Group target / supported | Group vacancy | Inside birth capacity |
|---|---:|---:|---:|---:|---:|---:|
| 11 / (0,-.5) | 7 | 28 | 29 / 21 | 38 / 31 | 7 | 4 |
| 12 / (0,0) | 0 | 10 | 21 / 7 | 43 / 27 | 16 | 7 |
| 17 / (.5,0) | 5 | 24 | 34 / 23 | 30 / 44 | 0 | 0 |
| 22 / (1,0) | 4 | 23 | 21 / 8 | 38 / 23 | 15 | 4 |

Eligible counts are learned-feature membership counts and can include rows outside the oracle acceptance radius. The reaction law acts on the training table; it does not evaluate these oracle mode IDs or directly optimize served counts. Mode17's training group is already above its reference target, so its vacancy ledger permits no additional support births even though the served table has only five accepted centres there.

The average is not uniformly worse: final RA4 training coverage is18 and precision0.48730, versus served clean coverage21 and precision0.82031. At1500, served modes4,9,13 have zero accepted rows while their training counts are38,35,43. At2000, training modes13 and18 have zero accepted rows while served counts are43 and38. Rows and generator parameters are being averaged across changing mode positions. Consequently, the served missing set cannot be explained entirely by zero training parent supply. The final EMA rate is1/256 per update; E22's is1/128 and its saved training table remains at25 modes from1000 onward. These checkpoint comparisons identify the state mismatch in the objective, without isolating its causal contribution from an intervention.

## Scope and result

The remaining limitation has two observed parts: recurrent training regions with zero or scarce eligible parents despite positive real cell/group targets, and a moving training table whose served average has a different set of weak modes. Global count evidence resolves the earlier local finite-bin power gap, but its copy-only actions retain these supply and serving limitations. The final recorded reaction spends all51 ordinary slots (1 mass+6 local+44 global) and rejects isolation for544 flags. This preceding GPU record is not replayed by the CPU refit.

No concrete implementation error was reproduced. No proposal, gate change, training or seed experiment follows from this diagnostic. Receipt source hashes cover both runs' checkpoints/metrics/configs, the RA4 package and learned source freeze, evaluator and original toy stream; all remained unchanged. Exact emitted/served trajectories are authoritative. CPU accessibility is a labeled same-input reconstruction; it cannot establish the precise historical GPU parent reservations or discovered categories.

# Constant learning-rate screen: clean 96-row wave

The first clean constant-rate wave found **no shared-22 candidate**. Of 96 predeclared configurations, 13 passed the full-budget trajectory gate, 10 of those passed residual-student, nine passed stripes, and none passed mode-hold. The remaining six screening hosts, the fresh full-19 replay, and the three 100-Gaussian problems were therefore **skipped**, not counted as failures. No seeds, budgets, host architectures, or thresholds changed.

The reproducible source epoch is commit `a72f4ffb90f14fa6a89a3744d04751eba0c0d27f`; its [driver](constant_lr_screen.py) and 96 configs are bound by manifest SHA-256 `0be07d4eb755dbd49593443f556903f72bc385234190d6166006f73bf861f9dd`. All runs used CPU, one thread per process, six parallel processes, and AVX2 dispatch. The exact configs, archived source, complete episodes, skip receipts, and logs are in `artifacts/toy100-constraints/particlegan-constant-lr-wave2-a72f4ff`; the compact [96-row leaderboard](constant-lr-wave1.json) records values, verdicts, final metrics, and measured case time. All 1,197 RAM evidence files matched the durable copy byte-for-byte (inventory SHA-256 `a53bdc7cbfb02e9f39be09a77a130bf6451b0ed0d75f750782719d7654bb5387`). All 128 attempted episodes independently regraded to the same status after relocation.

Every row removed `network_lr_horizon_cap` and `network_lr_floor`, set `lr_floor=1` and `lr_anneal_start=0`, and thus applied **multiplier 1.0 to G, D, and prior at every update**. Row `c000_winner_rate` changed only those schedule fields and the name from the passing shared recipe. Its 118 archived executable source-file hashes match the earlier production 22/22 replay exactly. The other rows varied only shared LR, D/prior LR multipliers, Adam β₂, regularizer κ, and regularizer coefficient. Sixteen rows were interpretable anchors; 80 were fixed Halton points spanning LR 0.0002–0.005, D multiplier 0.5–2.5, prior multiplier 0.5–5, β₂ about 0.975–0.9998, κ 0.7–1.5, and coefficient 2–10. Adam β₁ stayed floating-point `0.0`. The noise mechanism and all other fields matched the shared recipe.

| Full-budget stage | Attempted | Strict pass | Median case seconds | Outcome |
| --- | ---: | ---: | ---: | --- |
| trajectory | 96 | 13 | 1.18 | 83 stopped |
| residual-student | 13 | 10 | 1.22 | 3 stopped |
| stripes | 10 | 9 | 5.76 | 1 stopped |
| mode-hold | 9 | 0 | 7.14 | all 9 stopped |
| bars, overlap, blobs, intensity, unequal mass, unequal width | 0 | — | — | skipped |
| fresh full-19; native three | 0 | — | — | skipped |

The attempted cases consumed 262.04 measured training/evaluation seconds in aggregate. These are conditional counts: trajectory was observed for every row; later stages were observed only for rows that passed preceding gates. No run was truncated within a host. Every stage was regraded from its compressed episode and checked for the applied shared-noise receipt.

Mode-hold requires eight modes and HQ ≥0.9 throughout its sustained terminal pass. The scheduled production control had all eight modes at the final six checkpoints and HQ from 0.943 to 1.0 over the terminal five. Constant-rate `c000_winner_rate` never reached eight modes and finished at seven modes/HQ 0.668. The most informative constant-rate traces were:

| Candidate | Constant LR and key changes | Best transient 8-mode check | Final modes / HQ |
| --- | --- | --- | --- |
| `c006_lr_0_0025` | LR 0.0025; other optimizer fields equal to shared recipe | step 900, HQ 0.938 | 8 / 0.642 |
| `c036_halton_021` | LR 0.001654, β₂ 0.9813, D ×0.674, prior ×1.145, κ 1.434, coefficient 5.436 | step 1150, HQ 0.984 | 8 / 0.534 |
| `c080_halton_065` | LR 0.001025, β₂ 0.9949, D ×1.615, prior ×0.684, κ 1.460, coefficient 2.098 | never eight; HQ peaked 0.902 at seven modes | 7 / 0.875 |

For `c006`, terminal HQ was 0.806, 0.218, 0.500, 0.659, 0.642 at updates 1000–1200. For `c036`, HQ fell from 0.984 at update 1150 to 0.534 at 1200 while retaining eight detected modes. These changes are much larger than sampling uncertainty from the 4,096-draw checkpoint evaluations. They show non-sustained quality under those constant-rate runs; a curve alone does **not** establish a limit cycle or its cause. The nearest final row, `c080`, still lacked a mode and missed HQ by 0.025.

A bounded next experiment should test a game-stabilizing update with **the same constant actual group rates**, starting from the shared κ=1.176 core and the LR 0.0025 row on mode-hold and trajectory. [Optimistic Adam](https://arxiv.org/abs/1711.00141) or the Adam extragradient method in [Gidel et al.](https://arxiv.org/abs/1802.10551) directly target adversarial update dynamics; the existing optimistic scratch screen used a different optimizer core and does not answer this comparison. An extra-gradient method costs an additional gradient evaluation, so its computation and frozen-budget interpretation must be recorded. Any survivor still needs ten strict screening passes, fresh full-19, all three native problems, and a new common-22 replay. Constant-rate continuation beyond 7,000 updates and a changing-distribution test would be separate sustained-behavior evidence; changing the total budget also changes the existing noise burn-in and would confound that test unless burn-in is held fixed.

An earlier source epoch `406c3e9` used decimal points in 12 anchor names, violating a frozen legacy-host name constraint. Those 12 rows are **harness-invalid**; the other 84 completed screens and all partial evidence remain in `artifacts/toy100-constraints/particlegan-constant-lr-wave1-406c3e9` (1,098 copied files; inventory SHA-256 `767feb7e6cf97ba50d2ffb512cf4cf84c4e4c3bbed517d03397b259dfd584984`). The clean wave redeclared and reran all 96 configurations from the corrected source epoch. A separate 400-step smoke with the corrected low-LR anchor completed and failed numerically, confirming the name fix before that wave.

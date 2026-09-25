# Constant-rate game-update screen: 79 candidates

**No candidate passed sustained mode-hold.** All 79 complete 1,200-step episodes independently regraded to strict FAIL, with no harness errors. Later screening hosts, fresh full-19, native three, common-22, 14,000-step continuation, and distribution-shift checks were **skipped**. This is evidence about these declared methods and ranges, not a proof that a constant-rate shared recipe is impossible.

The [machine-readable leaderboard](constant-game-screen.json), [grid declaration](../../configs/toy100/constant_game_probes.json), and [independent collector](constant_game_collect.py) retain every candidate and terminal curve. Every rate remained constant at every optimizer update, including G, D, and prior. Seeds, frozen host models, budgets, thresholds, live-weight evaluation, and the winner's noise mechanism were unchanged. Six single-thread CPU workers used the AVX2 environment from the earlier constant-rate screen.

## Declared comparisons

| Wave / executable source | Candidate count | Shared core / update | Strict mode-hold passes |
| --- | ---: | --- | ---: |
| `6340eac` | 24 | Winner core; damped Optimistic Adam | 0 |
| `be540fd` | 16 | Simple core; ordinary or optimistic Adam | 0 |
| `be540fd` | 24 | Both cores; ordinary or optimistic AMSGrad | 0 |
| `101dd9f` | 15 | Local ordinary AMSGrad follow-up around `bg016` | 0 |

The winner core uses κ=1.176, regularizer coefficient 6, and prior regularization 0.05. The simple core changes only those three values to 1, 1, and 0. Its schedule-free comparison was declared after the separate scheduled recipe passed the older 19 hosts. All rows removed the network horizon cap/floor and set `lr_anneal_start=0`, `lr_floor=1`. The exact per-row configuration files and hashes are archived alongside each source epoch's manifest.

Wave 1 crossed LR {0.001, 0.0025, 0.00425}, β₂ {0.99, 0.999}, and optimism α {0.125, 0.25, 0.5, 1}. Wave 2 added 16 simple-core rows with LR {0.0005, 0.001, 0.0025, 0.00425}, β₂=0.999, and α {0, 0.125, 0.25, 0.5}; its 24 AMSGrad rows used both cores, LR {0.001, 0.0025, 0.00425}, β₂=0.99, and α {0, 0.25, 0.5, 1}. β₁=0, D multiplier 1, and prior multiplier 2 were common throughout these first 64 rows.

The 15-row extension had a specific numerical justification: `bg016` passed four consecutive terminal checks before failing the final one. The extension crossed LR {0.0008, 0.001, 0.00125, 0.0015}, β₂ {0.9, 0.99}, and prior multiplier {1, 2}, with the exact previously observed point excluded. It retained the winner core, α=0, and AMSGrad throughout. It found no improvement in sustained terminal performance.

## Numerical result

The gate requires eight modes and HQ≥0.9 throughout the terminal suffix. Ranking below is diagnostic; the frozen verdict remains FAIL for every row.

| Candidate | Update / core | LR; β₂; prior multiplier; α | Passing terminal checks / 5 | Final modes / HQ |
| --- | --- | --- | ---: | --- |
| `bg016` | AMSGrad / winner | .001; .99; 2; 0 | 4 | 8 / .8438 |
| `bg008` | Adam / simple | .0025; .999; 2; 0 | 2 | 8 / .9539 |
| `og005` | Optimistic Adam / winner | .001; .999; 2; .25 | 0 | 8 / .7783 |
| `ag011` | AMSGrad / winner | .0015; .9; 1; 0 | 0 | 8 / .8328 |

`bg016` had eight modes at updates 1000, 1050, 1100, 1150, and 1200, with HQ **.9211, 1.0000, .9607, .9216, .8438**. `bg008` had **8/.8188, 8/.8428, 8/.9492, 6/.5503, 8/.9539**. The last endpoint alone would wrongly promote the simple-core Adam row. None of the 15 local follow-ups had even one passing terminal checkpoint. All mode-hold episodes together consumed 1,276.73 measured case seconds, excluding the collector and test suite.

## What changed in the update

[Daskalakis et al., Algorithm 1](https://arxiv.org/html/1711.00141v2) uses twice the current bias-corrected Adam direction minus the previous direction. The tested damped extension was

`θ ← θ − η [(1+α) u_t − α u_(t−1)]`,

where `u_t = m̂_t / (sqrt(v̂_t)+ε)`. α=1 is the paper's rule; α=0 is ordinary Adam; intermediate α is an explicitly introduced tuning coefficient. Both directions use the same current group rate. The adapter preserves ordinary Adam moment updates and uses no extra gradient evaluation or training RNG draw. Numerical tests check the formula, α=0 bitwise equivalence, checkpoint replay, and every optimizer role.

[AMSGrad, Algorithm 2](https://arxiv.org/html/1904.09237v1) keeps the coordinatewise maximum of historical uncorrected second moments. The implementation delegates that maximum to PyTorch's Adam `amsgrad=True`, applies the usual bias correction, and optionally applies the same optimistic direction correction. It has no training-horizon schedule. Its preconditioner is still adaptive: **constant nominal LR is not constant effective per-coordinate LR**. The paper's convex regret theorem does not establish convergence of these GAN experiments, and the optimistic AMSGrad composition has no claimed guarantee.

Wave 2 and 3 receipts include actual gradient RMS, actual parameter-change RMS, and denominator minimum/maximum for every group at every update. For `bg016`, mean RMS movement of G fell from 0.001417 over updates 1–100 to 0.0000767 over updates 1001–1200; prior movement fell from 0.002767 to 0.000206. Nominal G/D/prior rates were exactly .001/.001/.002 throughout. Shrinking moves therefore occurred without a timed rate schedule, but still did not sustain the frozen quality threshold. No continued-learning or shift-responsiveness claim follows from this result.

## Evidence and replay

The raw durable archives are relative to the research worktree:

- `artifacts/toy100-constraints/particlegan-constant-game-wave1-6340eac`
- `artifacts/toy100-constraints/particlegan-constant-game-wave2-be540fd`
- `artifacts/toy100-constraints/particlegan-constant-game-wave3-101dd9f`

All **1,200 original RAM files** matched their durable copies byte-for-byte. Each archive includes the manifest, exact configurations, archived frozen suite sources, optimizer/driver/regrader sources, complete compressed episodes, actual-rate and update receipts, per-candidate logs, a retention proof, and a SHA-256 inventory. Final inventory hashes and manifest hashes are in the leaderboard. All 79 episodes were independently regraded after relocation. The production common gate still rejects every scratch optimizer episode, as verified by the scratch regrader; the evidence does not bypass that audit.

Run `constant_game_collect.py` against the three durable directories to rebuild the leaderboard and regrade receipts. To reproduce training, check out the indicated source epoch and run its `optimistic_screen.py prepare` then `run` against a fresh RAM directory. The screen's validator rejects changed executable sources, configurations, task order, or source commits. Its declared order begins mode-hold, then trajectory and the remaining eight inexpensive hosts, stopping only after a complete failed host.

Validation: **109 tests passed**, covering exact optimistic/AMSGrad formulas, shrink-and-react scalar behavior, checkpoints, G/D/prior application, missing or altered receipt rejection, and production common-gate rejection. The scalar shift test verifies optimizer mechanics; it is not a trained distribution-shift experiment.

## Recommendation

Stop expanding these nearby optimistic/maximum-second-moment grids. The next bounded mechanism should use an actual second gradient evaluation: [Gidel et al., Algorithm 4](https://arxiv.org/html/1802.10551v5) evaluates at a joint lookahead point, updates moments at both halfsteps, and takes the final step from the original weights. Two ordinary alternating Adam sweeps are not that algorithm. The additional gradient and data cost must be counted explicitly, and G/D gradients must refer to the same parameter point.

For a future gradient-proportional particle method, the `N` factor has a derivation rather than requiring an arbitrary huge prior rate: for empirical measure `μ_N=(1/N)Σδ_(z_i)`, the natural particle metric is `(1/N)Σ v_i·w_i`, so its gradient equals `N` times the Euclidean particle gradient. This is an untested mathematical proposal here. Applying it across auxiliary hosts requires binding which tensors represent empirical particles and how their objectives depend on that measure. Any resulting candidate still needs all 22 frozen problems, then a separate continuation with fixed noise burn-in and a shift test without resetting workers or optimizer state.

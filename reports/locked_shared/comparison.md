# Which existing formulation works on these toys?

**Removing the host particle L2 term is the strongest small-change candidate with EMA ring evaluation in this fixed-seed comparison:** all three measured targets pass, trajectory MSE is 0.002835, and the ring holds 8/8 modes at 100% HQ. The RpGAN loss, b_cap, models and training budgets stay unchanged. This is a candidate for these toys, not a universal replacement for particle L2.

## Same-budget comparison

All rows use seed 0. Training budgets remain two-pole 80, trajectory 400, ring 1,200 steps. Default: cover 1.5 (trajectory only), particle L2 0.02, VICReg 0.05 (trajectory/ring only). Only the named term changes unless the row explicitly combines settings. Two-pole has no VICReg or cover training loss, so those ablations leave that host unchanged.

| Formulation | Two-pole slope / result | Trajectory MSE / result | Ring modes; HQ / result | Targets passed |
| --- | --- | --- | --- | --- |
| Remove particle L2 | 0.4190 / PASS | 0.002835 / PASS | 8/8; 100.00% / PASS | **3/3** |
| Remove VICReg | 0.4197 / PASS | 0.002348 / PASS | 8/8; 92.07% / PASS | **3/3** |
| Base regularization on tiny host | 0.4190 / PASS | 0.002012 / PASS | 7/8; 100.00% / PASS | **3/3** |
| Locked / base GAN core | 0.4197 / PASS | 0.233890 / FAIL | 3/8; 49.49% / INCONCLUSIVE | **1/3** |
| R1+R2, coefficient 0.02 | 1.0976 / FAIL | 0.003386 / PASS | 7/8; 100.00% / PASS | **2/3** |
| Eikonal | 0.8506 / PASS | 0.023018 / FAIL | 4/8; 42.31% / INCONCLUSIVE | **1/3** |
| Interpolation cap | 0.5877 / PASS | 0.041957 / FAIL | 8/8; 100.00% / PASS | **2/3** |
| Ra logistic | 0.2803 / PASS | 0.002774 / PASS | 6/8; 82.71% / INCONCLUSIVE | **2/3** |
| Rp hinge | 0.2335 / PASS | 0.002002 / PASS | 5/8; 91.58% / INCONCLUSIVE | **2/3** |
| VICReg weight 1 | 0.4197 / PASS | 0.227468 / FAIL | 8/8; 100.00% / PASS | **2/3** |
| Remove trajectory cover | 0.4197 / PASS | 0.001346 / PASS | Not run | **2/2** |
| Trajectory cover 1.0 | 0.4197 / PASS | 0.002134 / PASS | Not run | **2/2** |
| Cap coefficient 10 | 0.4754 / PASS | 0.353823 / FAIL | 8/8; 91.72% / PASS | **2/3** |
| Remove both prior penalties | 0.4190 / PASS | 0.263834 / FAIL | 7/8; 100.00% / PASS | **2/3** |
| R1+R2, coefficient 0.1 | 0.7342 / PASS | 0.003768 / PASS | 5/8; 65.16% / INCONCLUSIVE | **2/3** |

Two-pole also requires travel ≥ 0.30; every candidate above reaches that travel threshold. Its slope limit is 1.0. Trajectory MSE must be ≤ 0.02. Ring requires ≥ 7 modes and HQ ≥ 90%. INCONCLUSIVE is preserved. Not-run combinations earn no pass.

The base-core row calls `get_recipe('gan').make_loss()` and `.make_gradient_penalty()`. It is numerically identical to locked_shared on these hosts. Base regularization means VICReg 1, particle L2 0, trajectory cover 0; the small cloud, host optimizers and budgets stay fixed in that row.

## Live-model leaderboard

Only the ring evaluation changes here: use the final live generator and live prior instead of their EMA. Two-pole and trajectory already evaluate live weights. Budgets, seed and thresholds are unchanged. Rows are ordered by the number of passed targets; equal counts are ties.

| Formulation | Two-pole | Trajectory | Live ring modes / HQ | Live ring verdict | Targets passed |
| --- | --- | --- | --- | --- | --- |
| R1+R2, coefficient 0.1 | PASS | PASS | 7/8; 100.00% | PASS | **3/3** |
| Base regularization on tiny host | PASS | PASS | 2/8; 24.61% | FAIL | **2/3** |
| Eikonal | PASS | FAIL | 8/8; 100.00% | PASS | **2/3** |
| R1+R2, coefficient 0.02 | FAIL | PASS | 7/8; 91.67% | PASS | **2/3** |
| Ra logistic | PASS | PASS | 4/8; 40.45% | INCONCLUSIVE | **2/3** |
| Remove VICReg | PASS | PASS | 5/8; 66.50% | INCONCLUSIVE | **2/3** |
| Remove particle L2 | PASS | PASS | 8/8; 74.05% | INCONCLUSIVE | **2/3** |
| Rp hinge | PASS | PASS | 2/8; 32.89% | FAIL | **2/3** |
| Cap coefficient 10 | PASS | FAIL | 6/8; 57.50% | INCONCLUSIVE | **1/3** |
| Interpolation cap | PASS | FAIL | 5/8; 65.16% | INCONCLUSIVE | **1/3** |
| Locked / base GAN core | PASS | FAIL | 5/8; 82.30% | INCONCLUSIVE | **1/3** |
| Remove both prior penalties | PASS | FAIL | 4/8; 57.59% | INCONCLUSIVE | **1/3** |
| VICReg weight 1 | PASS | FAIL | 5/8; 66.70% | INCONCLUSIVE | **1/3** |

**No particle L2 passes 2/3 live targets, tied for second by gate count.** Its ring reaches 8/8 modes, but 74.05% HQ misses the 90% requirement. It improves coverage over the original live model (5/8), while lowering HQ (82.30% → 74.05%). The original ring verdict is INCONCLUSIVE; the suite's binary summary counts that as a missed target.

**R1+R2 at coefficient 0.1 is the only recorded same-budget variant passing all three live targets:** two-pole slope 0.7342, trajectory MSE 0.003768, ring 7/8 modes at 100% HQ. That run retains particle L2 0.02. Combining R1+R2 0.1 with no L2 has not been tested. Its EMA ring misses the target, so there is no recorded small-host variant that wins all three targets under both EMA and live evaluation. These are final-step measurements from one fixed seed, not a stability guarantee.

## Stock recipe on the ring host

Here the stock recipe supplies its 20,000-particle prior, VICReg 1, no particle L2, batch 256, learning rate 0.0006, discriminator LR multiplier 1.5, prior multiplier 10, betas (0, 0.999), EMA 0.995 and cosine decay. The original 8-mode ring data, 96-wide host networks/initialization and 4,096-sample evaluation remain. This applies the recipe to the ring; it is not a run of the separate 100-Gaussians trainer.

| Budget | EMA modes / HQ | EMA result | Live modes / HQ | Live result |
| --- | --- | --- | --- | --- |
| 1,200 | 8/8; 83.06% | INCONCLUSIVE | 8/8; 91.50% | PASS |
| 7,000 | 8/8; 99.66% | PASS | 8/8; 99.05% | PASS |

The 7,000-step run uses the stock budget and more particles; it is not an equal-compute win over the 1,200-step tiny-cloud runs.

## What the measurements suggest

1. **No implementation drift was found.** PR #36, PyPI 0.5.0, and the base commit of the original ParticleGAN toy PRs use identical GAN loss, gradient penalty, particle-prior and VICReg source files. The current conceptmod suite reproduces 19 PASS and 2 FAIL after excluding cover-posture columns; the two failures are the same trajectory/ring cases.

2. **The auxiliary objectives interact badly in this small host.** Removing either particle L2 or VICReg clears both missed targets; removing both fails trajectory again (MSE 0.263834). L2 contracts the cloud while VICReg encourages spread. These interventions establish sensitivity to their combination, not a proof that one term is always harmful.

3. **Set coverage does not guarantee paired identity.** In the locked trajectory run only 6/12 predictions are nearest to their own target. Removing cover or reducing it from 1.5 to 1.0 restores 12/12 and passes identity MSE. The set-cover loss can reward the right collection of arcs under the wrong assignment; it competes with the identity objective.

4. **EMA and training stability matter.** The no-L2 ring's reported EMA is 8/8 at 100% HQ, but its live model is 8/8 at 74.05% HQ. The stock 7,000-step run passes with both EMA and live weights. The no-L2 candidate is promising, but its live dynamics are not solved. R1+R2 at 0.02 passes trajectory/ring, yet misses the two-pole slope requirement (1.0976 > 1); increasing it to 0.1 fixes that bound but loses the ring EMA target.

**Recommendation:** keep the current production defaults unchanged in this verification PR. Use the no-particle-L2 arm as the next candidate for hosts where VICReg already supplies the prior regularization; retain the measured identity, diversity and slope tests. Keep the stock recipe as the longer-budget baseline. There is no evidence here that a new GAN formulation is needed, and no candidate has been shown to win every behavioral toy or downstream task. No seed search was performed.

## Reproduce

```bash
python -m benchmarks.locked_shared.investigate --only base_core r1_r2 eikonal interpolation_cap relativistic_average rp_hinge no_vicreg base_prior_weight no_particle_l2 no_cover music_cover stronger_cap
python -m benchmarks.locked_shared.investigate --only base_regularization no_prior_regularization r1_r2_0_1 --output reports/locked_shared/followup.json
python -m benchmarks.locked_shared.base_recipe
python -m benchmarks.locked_shared.summarize
```

Raw metrics and training curves: [initial comparisons](investigation.json), [follow-up controls](followup.json), [stock recipe](base_recipe.json). These are actual trained outcomes, not configuration acceptance checks.

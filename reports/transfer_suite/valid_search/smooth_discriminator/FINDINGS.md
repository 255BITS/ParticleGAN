# Smooth discriminator findings

**Softplus(beta5) solves unequal width under the original recipe and budget.** This is useful architecture evidence for a formulation entry that permits different discriminator choices per toy. It is not an all-six shared-architecture win: both fully evaluated finalists pass three data toys.

| Research D | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained total |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Axis Fourier + Softplus(beta5) | PASS | FAIL | PASS | FAIL | Final only | PASS | 3/6 |
| Axis Fourier + Tanh | PASS | FAIL | FAIL | FAIL | PASS | PASS | 3/6 |

The Softplus5 unequal-width run sustains the last **7/24** observations, confirming at step 1,100 of its original 1,200-step budget. Final live HQ is **98.706%**, mass TV **.05591**, normalized sliced distance **.07169**, component covariance error **.41357**, and minimum eigenvalue ratio **.28064**. The critic keeps the original 4,929 parameters, axis Fourier features, width 64, two hidden layers, initialization order and optimizer; only its activation changes.

On rare mass, Softplus5 passes covariance (**.38232**), minimum eigenvalue (**.45857**), HQ (**98.755%**) and minimum mass ratio (**.69691**). It still fails overall: mass TV is **.166699** (bound .15), normalized sliced distance **.181933** (bound .18), and no observation passes every bound. The dominant 55% component receives 38.33% of generated mass. Density shape is improved, but target occupancy is wrong. Overlap has a numerical final pass with suffix1; it remains a sustained failure. Anisotropic fails distance, mass TV and covariance.

Eight architectures were declared before training: original axis Fourier features with SiLU, Softplus(beta1), Softplus(beta5) or Tanh; fixed random-oriented features with four projections and SiLU; or eight projections with SiLU, Softplus1 or Tanh. The oriented features use declared radial frequency bands and one local seed0, accept no target information, and do not consume global initialization RNG state. Four projections preserve 4,929 parameters; eight explicitly increase input features and parameter count to 5,441. No random-feature variant solves rare mass or unequal width. Oriented8 Tanh passes overlap only in its three-case screen.

Every run retains the original shared recipe: Adam(0,.99), G LR .001, D LR .0015, prior LR .01, 1:1 updates, cosine schedule, Rp logistic, b_cap coefficient3/kappa1.25, prior regularization .05 and no particle L2. G is unchanged; there are 256 particles, batch128, and original budgets of1,200 (spiral1,600). All numerical metric thresholds remain unchanged. Each complete curve has24 fixed live observations and needs a final passing suffix of at least5; EMA is separate.

There are **30 actual GAN episodes**: eight architectures × three hard toys, followed by the remaining three toys for the two best screen cards (sustained pass count, then lower mean normalized final shortfall). They contain **720 live observations** and **193.454 seconds** summed recorded wall time on a shared CPU host. No extra-budget runs, optimizer changes, target-derived features, auxiliary losses or seed searches were used. The [complete matrix](MATRIX.md) retains every failed and untested cell.

Every episode stores `candidate.architecture` and `spec.research_discriminator`. Validation confirms that the effective spec differs from the original only by that explicit research-architecture declaration. The exact new numerical module SHA256 is `9958f52f265c6ecf37de1449afc02d25db916f7c05879017c33025757f81bfcc`. Source is based on commit `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff` plus the archived research module. The archive retains all58 numerical dependency files and the exact driver.

Ten focused reusable-module tests pass. They exercise an active cap's second-order parameter-gradient path, unchanged axis features and linear initialization, random-projection RNG isolation, and state-dict restoration. These static tests are separate from the30 GAN episodes.

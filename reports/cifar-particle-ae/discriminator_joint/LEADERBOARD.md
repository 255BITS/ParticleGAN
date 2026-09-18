# Joint discriminator interventions

3/3 certified. All start from the identical CNN E-only 10k checkpoint (FID50k 19.4482).

| Run | Final FID50k | Test reconstruction MSE | Joint train minutes | Wall minutes |
|---|---:|---:|---:|---:|
| weaker | 19.7848 | 0.14736 | 7.36 | 9.77 |
| control | 20.0119 | 0.14806 | 7.20 | 9.57 |
| warmstart | 25.2598 | 0.14797 | 7.25 | 9.60 |

Warmstart has an additional 2048 D-only updates (48.6 training seconds), original regularization thereafter. Weaker changes only bcap coefficient from1 to0.1, retaining every8 schedule. Control preserves original recipe and all training state. One D update per G update throughout joint training. No seed experiments.

| Run | Step | FID50k |
|---|---:|---:|
| weaker | 15000 | 20.1344 |
| weaker | 20000 | 19.7848 |
| control | 15000 | 19.7875 |
| control | 20000 | 20.0119 |
| warmstart | 15000 | 27.4657 |
| warmstart | 20000 | 25.2598 |

Final changes relative to matched control: weaker -0.2271, warmstart +5.2479.

FID uses EMA G/prior, 50k generated images and the unchanged cached CIFAR train50k reference. Diagnostic D-only AUC uses live weights and held-out test images, so it is a different measurement. A single continuation per intervention does not quantify stochastic run-to-run uncertainty. No automatic long promotion.

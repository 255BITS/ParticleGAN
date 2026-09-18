# Scratch transformer and E-only reconstruction results

All three50k scouts completed and were re-certified against their configs and current source hashes. Pipeline exited0. Both GPUs are idle; no continuation is queued. The target below13 remains unmet.

| Generator and reconstruction routing | Final FID50k ↓ | Test reconstruction MSE ↓ | Training hours |
|---|---:|---:|---:|
| Historical CNN, full reconstruction | **18.9012** | 0.04022 | 0.62 |
| New CNN, E only | 23.1749 | 0.15222 | 0.61 |
| Transformer, full reconstruction | 24.4911 | **0.03357** | 4.72 |
| Transformer, E only | 24.6440 | 0.22069 | 4.54 |

The historical CNN is the existing trajectory reused at user request, not a new concurrent control. All measurements use50k prior samples, EMA G/prior and the same CIFAR train50k Inception reference. Reconstruction uses10k test images. No seed experiments were run.

## What the curves tell us

**Reconstruction gradients on G/prior are not necessary for the plateau.** The fresh E-only CNN already reaches19.4482 at10k, stays around19.7–20.4 through40k, then worsens to23.1749. The transformer E-only starts at20.3919 at10k and never improves below that. Neither architecture unlocks better FID when reconstruction trains only E. This weakens the hypothesis that direct L2 competition is the sole bottleneck; it does not establish that L2 is always beneficial or harmless.

**More generator capacity improves reconstruction without improving generation.** The transformer has18.85M G parameters,29.2× the CNN. With full reconstruction, its test MSE beats the historical CNN by16.5%, but FID worsens by5.59 and training costs about7.6× as much. Within the transformer run, reconstruction improves from0.07970 at10k to0.03357 at50k while FID fluctuates between24.41 and27.26. The extra capacity is usable for reconstruction; it does not translate into better unconditional distribution matching under this recipe. This is architecture plus capacity, not an isolated attention experiment or a faithful reproduction of the full published TransGAN setup.

**The E-only transformer is substantially less stable than its final score suggests.** Its curve is20.39 →22.92 →25.47 →71.87 →24.64. The40k sample grid repeats a small set of aircraft/animal/object appearances, visibly supporting a temporary loss of diversity. The50k grid is more varied again. Its final score being only0.15 worse than full reconstruction conceals that excursion. Feature covariance trace ratio remains0.97 at40k, showing that this scalar alone misses the distribution problem; do not describe all feature variance as collapsed.

The CNN E-only final regression is also important: it is20.0023 at40k before jumping3.17 points at50k. Consequently, the+4.27 final difference against historical full reconstruction should not be read as a stable, isolated estimate of reconstruction's benefit. The robust observation across the curves is that removing reconstruction from generation did not produce a lower plateau. No new observed minimum beats the historical50k CNN checkpoint.

## Remaining hypotheses and recommendation

The evidence shifts attention toward mechanisms shared by the generators: discriminator feedback, its regularization/optimization, and the particle prior/sampling setup. It does not uniquely identify one. Logged D/G adversarial losses in all three runs mostly sit near0.693 at evaluation points, despite mediocre FID. That is consistent with the previously observed weak-feedback issue, but loss values alone cannot distinguish a useful equilibrium from an ineffective discriminator. No new gradient probes were run in this results review.

**Do not promote either transformer endpoint into a long continuation.** Its cost is much higher, its final FID is worse, and its curve does not show a sustained improvement toward the target. The E-only CNN also offers no convincing improvement to promote.

Recommended next controlled diagnostic: use a saved inexpensive CNN checkpoint near10k, freeze G and the prior, and test whether D can learn to separate fixed-generator samples from real images under the current regularization. Evaluate on held-out draws rather than only training batches, and measure image gradients as well as discrimination. This would test a suspected shared limitation before another costly generator change. It is a proposal only; no D-only run or further training was launched.

All50k endpoints and intermediate10k checkpoints remain available. `LEADERBOARD.md` contains the full curves and costs; `leaderboard.json` contains certified summaries. `PLAN.md` and `PREFLIGHT.md` document the architecture adaptation and the preflight saturation fixes.

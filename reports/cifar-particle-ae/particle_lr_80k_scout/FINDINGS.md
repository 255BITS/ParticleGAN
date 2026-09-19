# Lower learning rates do not break the FID plateau

Both80k-to100k training forks and both endpoint probes completed and passed source/config certification. Best/final checkpoint hashes verified. No further training queued. All scores use the historicalFID50k protocol; coverage/density use10k real/fake features.

| Model | Step | FID50k | Density | Coverage | Latent component confusion | Median nearest-center distance |
|---|---:|---:|---:|---:|---:|---:|
| Original parent | 80k | 15.7527 | .64686 | 64.26% | .0092% | 2.1637 |
| Unchanged continuation | 100k | 16.4609 | .59324 | 63.36% | .0244% | 2.3309 |
| Half G LR | 100k | 16.1118 | .63288 | 64.19% | .0946% | 2.3422 |
| Half all LRs | 100k | 19.2482 | .63452 | 61.01% | 3.5858% | 1.2143 |

Half-G improves the100k endpoint by0.3491FID and0.83percentage points coverage versus unchanged. This is mitigation of deterioration, not a new best: it remains0.3591FID worse than the80k parent. Its85/90/95/100k curve is15.9338/16.3924/16.3796/16.1118. The final downswing is encouraging but does not establish sustained improvement. Best resumed point15.9338 at85k; preserve the better original80k checkpoint.

Half-all worsens monotonically across sampled85/90/95/100k:17.2142/17.5723/18.6474/19.2482. It finishes2.7873FID worse than unchanged, with2.35percentage points less coverage. Its density remains above the unchanged control: density alone is insufficient. These results weaken a simple explanation that every learning rate is too high late in training; residual-specific causality remains untested. E reconstruction is encoder-only, so changing E LR does not directly update G or the generative prior.

## Generator diagnostics

On256 fixed latent inputs from the parent EMA prior,100k EMA output drift is0.3843RMS unchanged,0.3445 half-G and0.3434 half-all (pixels in[-1,1]). Both reduced-rate arms move less, but almost equal drift accompanies very differentFID. Output movement magnitude alone does not explain quality.

No obvious conditioning-scale or activation explosion appears on these fixed inputs. Across the original80k snapshot and sampled checkpoints, conditioning gain RMS remains roughly1.47–2.03 and block activation RMS roughly0.96–2.25. Output saturation stays around1–2%. These aggregate observations do not bound worst-case behavior or prove a healthy generator Jacobian.

The last learned1x1 skip projection (`blocks.2.skip.weight`) does stand out in relative Adam updates. Averaged over the last10 logged updates, its update/weight RMS is1.034% in half-G and1.083% in half-all, versus roughly0.16–0.19% for neighboring conv/conditioning weights. In half-G, the skip weight RMS is0.01408 and update RMS0.0001456; conv1 weight RMS0.08605 and update RMS0.0001431. Thus the high ratio largely reflects smaller learned weights receiving similar absolute updates. This is a lead about per-layer parameter scaling, not evidence the skip path dominates the output or causes the plateau. It occurs in both arms, so it does not by itself explain their difference. Equalized fan-in scaling is not guaranteed to solve a mismatch in learned weight magnitudes.

## Prior geometry and next test

Half-all brings substantial local clumping forward to100k: median nearest-center distance1.2143 versus2.3422 for half-G, and1175/32768 latent confusion errors versus31/32768. Every observed error stays within the same original clone family. Its confusion3.59% already exceeds the historical unchanged160k2.67%. This is a component-distinguishability diagnostic, not a semantic-collapse measurement.

The prior LR and D LR changed together between half-G and half-all. Therefore the new result cannot distinguish weaker discriminator adaptation from altered prior dynamics, nor prove clumping causes theFID regression. The previous frozen-center result already showed that increasing overlap is not necessary for regression. Halving nominal rates also is not equivalent to running half as many steps, given Adam state, fixed EMA, regularizer cadence and coupled updates.

Recommendation: preserve the80k parent, do not extend half-all, and do not automatically promote half-G merely because the last point fell. The cleanest next pair of checkpoint forks would retain half-G and separately halve onlyD or onlyprior LR, leaving other rates at their original settings. This separates the remaining changes responsible for the strong difference between the two completed arms. This is a recommendation only; neither job is launched. If prioritizing the user's residual-layer hypothesis instead, measure actual skip/residual branch contributions before testing a lower LR only on learned skip projections. Do not replace architecture or EqualLinear parameterization and rates simultaneously.

One trajectory per intervention; no seed-only replicates. Source-preservation and observer-state audit passed. Complete training is not bitwise reproducible because the historical CUDA pooling backward lacks a deterministic implementation; seeVALIDATION.md. FID differences are descriptive, not statistical significance claims. SeeLEADERBOARD.md for curves,curves.png for plots,DIAGNOSTIC_REVIEW.json for tensor metrics, andCHECKPOINTS.json for preserved artifacts.

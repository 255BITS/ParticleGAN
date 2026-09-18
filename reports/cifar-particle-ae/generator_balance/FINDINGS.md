# Generator-only LR test

Halving G learning rate did not improve the plateau. It lost to the matched control at both FID50k evaluations. Neither continuation improved on the common 10k parent (19.4482). No endpoint is promoted.

| Arm | FID50k at 15k | FID50k at 20k | Test MSE at 20k |
|---|---:|---:|---:|
| Control | 19.5589 | 20.3044 | 0.14846 |
| Half G LR | 20.7412 | 20.6996 | 0.14310 |

The final 0.3952-point disadvantage is small, and a single continuation does not estimate stochastic uncertainty. The observation is absence of a demonstrated gain, not proof that every lower G rate is harmful. MSE improves 3.6% while FID does not, another example of reconstruction quality not predicting unconditional generation quality.

## What was controlled

Both restore the same scratch CNN E-only 10k checkpoint, complete G/D/E/prior/EMA/Adam/RNG state. Only G LR changes from 0.0003 to 0.00015; E 0.0003, prior 0.003, D 0.00045 remain fixed. One D update, original bcap coefficient 1 every 8 with multiplier 8, no seed experiments. Actual per-group rates, intervention manifest and matched RNG consumption audited. Frozen features, sigma and parent hashes verified. Both source certificates passed.

A new standalone trainer preserves all historical certificates. Two deterministic CUDA tests passed: exact full-state unchanged continuation, and actual first-update isolation of G LR with identical E/prior/D parameters and all Adam moments. Both actual-parent eight-update smokes passed. Training takes 7.20/7.34 minutes; wall 9.59/9.73 minutes. No material cost advantage or penalty beyond normal execution variation.

## Interpretation

The proposed simple update-balance fix is unsupported at this setting. Existing D can learn fixed-target separation, but D warmup, weaker bcap and now lower G LR have not yielded useful sustained FID gains. This does not prove pretrained features are adequate or robust as training feedback. Classification AUC, gradient norm, and ability to fit a frozen generator distribution are different from providing useful directions in joint training.

## Endpoint feedback probes

Both read-only endpoint probes passed certificates and frozen-state checks (held-out 2048 images plus independent generated draws).

| Arm | Test AUC | Fake image gradient | G adversarial gradient |
|---|---:|---:|---:|
| Control | 0.4743 | 0.1558 | 0.1435 |
| Half G LR | 0.5575 | 0.4921 | 0.5724 |

Half-G has roughly 4x the G gradient norm and better real/fake ranking, yet worse FID. Thus this intervention changes measured feedback strength without producing the desired image-distribution improvement. Endpoint ranking depends on phase in the game; neither AUC nor gradient magnitude establishes feedback quality or a unique cause. Pixel/feature gradient cosine remains near zero.

All three read-only support probes completed. Removing noise worsens FID to 42–43; doubling noise worsens it to 21.7–23.5. Grouped samples retain similar object/pose/layout within each inspected particle, with about 5% within-particle pixel variation and 32–33% Inception-feature variation across the balanced diagnostic. This motivates testing more trainable particle centers, not claiming that a hard support ceiling has been established. See ../particle_support/FINDINGS.md. No training is queued.

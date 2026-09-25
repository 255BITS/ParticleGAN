# Network-only H1600 affine rotated100 probe

This scratch probe used [the archived configuration](accuracy-network-horizon/declared_config.json), the same generic uniform-square prior initialization on `[-5,5]^2`, identity-initialized trainable affine generator, Fourier-3 discriminator, seed 1234, and 7,000-step budget. Only the learning-rate time policy differed: G and D followed the original cosine with horizon capped at 1,600 updates, while the learned prior followed its original 7,000-update cosine. This is **not common-gate evidence** or a public `Recipe` change.

The [complete evidence](accuracy-network-horizon/rotated100) passes the individual original and strict accuracy gates. All five terminal live checks pass, and the separate 100,000-draw holdout passes. The probe saved 7,000 actual G/prior/D rate receipts and asserted each against its declared role schedule; all were verified. G and D reached the 5% floor (`0.0002125`) by update 1,600. The prior remained at `0.0085` through update 4,200, then decayed to `0.0004250` by the final update. Recorded source hashes matched the working source at completion.

| Step | Modes | HQ | Mass TV | Center RMS / σ | Covariance trace bias | Radial KS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6,000 | 100 | 0.9741 | 0.04725 | 0.1563 | +0.0674 | 0.0331 |
| 6,250 | 100 | 0.9789 | 0.04725 | 0.1444 | +0.0203 | 0.0185 |
| 6,500 | 100 | 0.9811 | 0.04710 | 0.1151 | −0.0114 | 0.0054 |
| 6,750 | 100 | 0.9822 | 0.04710 | 0.1190 | −0.0274 | 0.0096 |
| 7,000 | 100 | 0.9828 | 0.04715 | 0.1475 | −0.0374 | 0.0146 |

The live 100k holdout has precision 0.98311, mass TV 0.03784, center RMS 0.1209σ, covariance trace bias −0.0367, and radial KS 0.0136. Earlier, while the prior still used its full rate, live HQ lagged EMA markedly (0.9232 versus 0.9721 at step 3,000). Live HQ rose through 0.9680 at step 5,750 and first passed at step 6,000. The time ordering is consistent with late prior settling, but this single trajectory does not establish causation. Qualification would require the same declared policy on all three 100-mode problems and the 19 frozen transfer hosts.

## Same-configuration replay on all three

The matching grid100 and staggered100 runs also pass all five terminal live checks and independent 100k holdouts. [All three evidence sets and the combined GIF](accuracy-network-horizon/README.md) are archived together. The shared config, model options, exact probe source, and all 7,000 optimizer-rate rows are byte-identical. This establishes experimental 3/3 accuracy, with the shared 22-case gate still outstanding.

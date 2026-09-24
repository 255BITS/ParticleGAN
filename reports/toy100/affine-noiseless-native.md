# Noiseless affine native controls

The public default optimizer and loss core, with zero input/output noise, passes a fresh older-19 replay. These two native grid100 controls change only the declared discriminator Fourier resolution (5 or 7); both use the same affine generator, 20,000 uniformly initialized particles, seed 1234, 7,000 updates, and shared G/D horizon 1,600. They do not pass strict native accuracy.

| Fourier bands | Final quality modes | Passing terminal checks | Holdout precision | Mass TV | Center RMS / σ | Covariance trace bias | Radial KS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 100/100 | 0/5 | 0.94631 | 0.04394 | 0.49046 | -0.20372 | 0.04502 |
| 7 | 91/100 | 0/5 | 0.95693 | 0.09243 | 0.42121 | -0.12648 | 0.06397 |

Fourier-5 covers all 100 modes and passes the mass-balance limit, but component centers and shapes remain inaccurate. Fourier-7 also loses adequate quality coverage and mass balance. The corresponding target holdout passes the oracle checks. No threshold or budget is changed to rescue either result.

Source was frozen at `1c1a086`. The local archive `artifacts/toy100-accuracy/affine-noiseless/native-f5-f7-v1/` retains the predeclared manifest, both exact configs, all final-window samples and independent 100k holdouts, executable source, and failed regrades. Every file was SHA-256 verified during relocation. The [fresh older-19 result](affine-noiseless-f5-transfer.md) is separate evidence, not a combined 22-case pass.

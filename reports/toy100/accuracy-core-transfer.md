# Older-toy tuning still needs a matching 100-mode accuracy pass

The shared G/D-horizon experiment passes all three strict 100-mode gates with
β₂=.999, prior LR multiplier 2, output σ=.029, and input noise ending at 10%
of the budget. Two bounded counterchecks substitute configurations selected
from older-toy results. The affine/square initialization, network horizon
1,600, Fourier-3 critic, batch 2,048, 7,000 updates, fixed seed, and gates are
unchanged. These are explicit scratch probes, not production common-gate rows.

| Older-toy configuration | Native countercheck | Final coverage | Strict native gate | Holdout mass TV | Center RMS / σ |
|---|---|---:|---|---:|---:|
| Best eight-host noise screen: β₂=.99, prior2, σ=.0275, input end .125 | grid100 | 100/100 | FAIL | .09791 | .21400 |
| Best full19: β₂=.999, prior3, σ=.029, input end .2 | rotated100 | 100/100 | FAIL | .03338 | .11100 |

The first configuration passes 7/8 older bottleneck hosts but fails native
mass balance and centering. At the final 20k draw, mass TV is .1019 (strict
limit .06), center RMS .23755σ (limit .20), and the largest within-mode
covariance eigenvalue ratio is 1.7266 (original limit 1.70). Its mean width
bias −.0609 and radial KS .0153 pass; narrow output noise is not the observed
reason for rejection. All five terminal checks and the holdout fail.

The second configuration passes 16/19 older hosts. Its rotated100 holdout
passes every criterion, but the first of five terminal live checks, update
6,000, has precision .9648 below .97. Updates 6,250–7,000 pass, giving only
four terminal successes. The final 20k precision is .9786; holdout precision
is .97787, covariance bias −.03039, and radial KS .01113. This remains a
strict failure rather than a promoted near-pass.

Complete local evidence is retained under
`artifacts/toy100-accuracy/affine-network-best-old-grid` and
`artifacts/toy100-accuracy/affine-network-best16-rotated`. Each contains the
exact declared config, extra model/schedule options, probe source, 7,000
actual optimizer-rate rows, events, terminal clouds, holdout, and regraded
verdicts. Neither result can be combined with a different recipe's passes.

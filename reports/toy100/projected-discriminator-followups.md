# Projected discriminator follow-ups

Two predeclared noiseless grid100 probes followed the [four-direction F5 comparison](projected-discriminator-f5.md). Both kept the ordinary trainable affine generator, the public-default optimizer/loss core, seed 1234, 20,000 particles, and the complete 7,000-update budget. Neither passes the original coverage gate or strict accuracy; all five terminal checks fail.

| Single change from four-direction F5/H1600 | Final modes | Final precision | 100k precision | 100k mass TV | 100k center RMS / σ | 100k trace bias | 100k radial KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight evenly spaced unit projection directions | 96 | .96740 | .96748 | .08901 | .19163 | −.08830 | .03338 |
| Shared G/D learning-rate horizon 3200 | 100 | .96480 | .96597 | .04933 | .25232 | −.06587 | .01667 |

Eight directions preserve adequate average center/width statistics but lose mass balance and sufficient coverage. Their final conditional covariance eigenvalue ratios span .2567–2.1597. Extending the shared network horizon keeps all modes but fails precision, centering, and component shape; final eigenvalue ratios span .1789–2.0566. The required eigenvalue interval is .4–1.7. These changes do not repair the component-level failures hidden by favorable average shape statistics.

The eight-direction source is `b26a2c2`, with exact original config SHA `51f6bd91a60e523708b51a36bd2f38907a28bc138a1d0d1b366e6a6a200586b0`. The horizon source is `4a0bf2c`, with config SHA `c4683fe886b056d42ebc7c17bdf62226a6831d14b116733951cf0050bcca5984`. Each source passed three focused construction tests before training. Both models explicitly declare scratch/common-gate ineligibility.

Local evidence is retained in `artifacts/toy100-accuracy/affine-noiseless/projected-d8-f5-v1-b26a2c2/` and `artifacts/toy100-accuracy/affine-noiseless/projected-f5-h3200-v1-4a0bf2c/`. Each archive's 58 files matched the RAM originals by SHA-256. Both relocated source archives verified, and independent coverage/accuracy regrading reproduced FAIL/FAIL from the retained terminal and holdout samples.

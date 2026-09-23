# Noiseless affine grid: Fourier 5, 6, and 7

All three fixed-seed native `grid100` runs **failed both frozen gates at all five terminal checks**. The F5 recipe separately passed all 19 older hosts; that older-host result does not rescue its native 100-mode failure. F6 was a preregistered architecture control: its exact config differs from F5 only in `name` and `fourier: 5 → 6`. F7 likewise differs from F5 only in those two fields. Source stayed at clean `1c1a0865fe605c9f212d832c06596c12510f937e`; each native run used 7,000 steps, five final 20k checks, and a separate 100k holdout.

| Run | Config SHA-256 prefix | Final frozen modes / precision | Final TV / center RMS / covariance trace bias / radial KS | 100k holdout TV / center RMS / covariance trace bias / radial KS |
| --- | --- | --- | --- | --- |
| F5 | `51f6bd91` | 100; .9467 | .0526 / .499σ / −.203 / .0437 | .0439 / .490σ / −.204 / .0450 |
| F6 | `9e35b6e7` | 100; .9480 | .0479 / .392σ / −.110 / .0194 | .0402 / .383σ / −.112 / .0228 |
| F7 | `2e335962` | 91; .9583 | .0999 / undefined¹ / undefined¹ / undefined¹ | .0924 / .421σ / −.126 / .0640 |
| Target oracle | same saved target draws | 100; .9894 | .0248 / .101σ / −.0086 / .0069 | .0131 / .0468σ / −.0040 / .0019 |

¹ F7 had only one in-radius point in mode 0 among its final 20k draws, so the frozen conditional-moment scorer correctly declined to compute center, covariance, and radial statistics. Its 100k holdout had nine points in that mode versus 989 target points. Modes 10 and 99 had only 50 and 59 holdout points. The three runs' saved target arrays were bit-identical.

The 100k clouds expose the conditional error hidden by pooled metrics. Each mode was assigned to its nearest target center; center and covariance statistics use only points within the frozen 3σ radius. Covariance ratios below are relative to the target Gaussian's analytic variance after that truncation. Per-mode radial KS is an additional diagnostic, not a new gate.

| Holdout live cloud | Median / 90th percentile center bias | Median covariance trace ratio | Median per-mode radial KS | Lowest per-mode precision |
| --- | --- | ---: | ---: | ---: |
| F5 | .387σ / .772σ | .758 | .255 | .723 |
| F6 | .286σ / .643σ | .908 | .232 | .826 |
| F7 | .330σ / .659σ | .851 | .229 | .627 |
| Target oracle | .041σ / .067σ | .997 | .029 | .980 |

F6 improves the pooled radial shape and mass balance, yet its center RMS remains almost twice the frozen .20σ accuracy limit; its covariance trace bias exceeds the .10 limit, and precision .948 misses the original .97 requirement. Its conditional covariance is still uneven: the narrowest mode has only .308 of target trace variance on the holdout, while the final 20k frozen covariance eigenvalue range is .142–2.026 versus the required .4–1.7. F5 has an even narrower holdout mode with trace ratio .014. These deviations are much larger than oracle sampling variation: even the 20k target's median per-mode radial KS is .058, versus F6's .231 at 20k. Finite particle support alone therefore does not explain the observed conditional-shape error.

The per-mode center offsets are mostly local rather than a single smooth affine warp. After fitting a weighted affine trend across the 100 centers, residual center RMS is .485σ for F5, .379σ for F6, and .416σ for F7, close to their total holdout center RMS values .490σ, .383σ, and .421σ. This is a numerical description of the saved clouds, not an attribution to a particular optimizer component.

The F5/F7 evidence (`artifacts/toy100-accuracy/affine-noiseless/native-f5-f7-v1/`) and F6 evidence plus diagnostic script and JSON (`artifacts/toy100-accuracy/affine-noiseless/native-f6-v1-1c1a086/`) are retained. F6's relocated V2 source archive verified; independent coverage and accuracy regrading reproduced valid `FAIL 0/1`, with no integrity error. No F6 older-host replay was needed after the native failure.

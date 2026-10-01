# Final RA10 saved mean-law decomposition

One descriptive CPU chart; original grid validity VALID and quality FAIL remain unchanged.

Chart: K=128, rank=8, groups=100.
Current joint cohort: L=19772, U=228.

| Population | Feature residual energy | Raw group-mean gap | Raw covariance trace |
|---|---:|---:|---:|
| anchors_A | 0.5012675269752216 | 0.0068738187 | 0.0006438542 |
| anchors_L | 0.5254331351927515 | 0.0069031386 | 0.00027845008 |
| anchors_U | None | 0.062469866 | 0.010639833 |
| saved_clean_C | 0.5027375575696992 | 0.0069748317 | 0.00062707118 |
| saved_noisy_Y_clean_frame | 0.2733283335238303 | 0.0071436676 | 0.0023039854 |
| saved_noisy_Y_natural | 0.2733283335238303 | 0.0071436676 | 0.0023039854 |
| saved_real_target | 0.09590821368751415 | 0.0045353216 | 0.0018009461 |

## Fixed comparisons

- Paired feature-noise increment versus anchor target residual: dot=0.43952929, cosine=0.7715159348156404, covered even mass=1.
- Paired feature-noise increment versus clean target residual: dot=0.43843709, cosine=0.768472778910112, covered even mass=1.
- Unpaired A-to-C distribution delta versus anchor target residual: dot=6.9400955e-05, cosine=0.0024438539310711015, covered even mass=1.
- All-anchor residual versus L inside-even target residual: dot=0.43733363, cosine=0.9386251314297783, covered even mass=1.

Natural clean-to-noisy group transitions: 0.
Paired raw mean increment norm: 0.0011751345.

## Limits

C/Y use g(C) for both binning and clipping frame in the paired increment. Natural g(Y) energy is separate.
A-to-C is unpaired. These are final-state empirical comparisons, with trained D/shared FIFO, assignment, clipping and finite-sample limits.
No historical causal attribution, equality certificate, scorer result or production repair is established.
No actions, new samples, data/latent/output-noise draws, training updates, constructors, CUDA contexts or scoring calls occurred.
The one fitted chart uses the declared private random projection from a clone of saved CPU RNG.

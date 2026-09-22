# Smooth discriminator architecture screen

Research architecture variants only. All cards retain the original vector recipe: Rp logistic, b_cap3/kappa1.25, prior regularization .05, Adam(0,.99), G LR .001/D LR .0015/prior LR .01, cosine and 1:1 updates. Original G,256 particles,batch128 and budgets are unchanged (1,200; spiral1,600). Seed0 only. Live PASS needs every metric passing for the final five of24 observations. EMA is separate.

| Research D | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained / attempted | Mean final shortfall | Seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| axis_softplus5 | PASS (17/24) | FAIL (0/24) | PASS (7/24) | FAIL (0/24) | Final only (1/24) | PASS (22/24) | 3/6 | 0.0423 | 42.33 |
| axis_tanh | PASS (17/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | PASS (6/24) | PASS (15/24) | 3/6 | 0.1776 | 37.75 |
| oriented8_tanh | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (5/24) | Not run | 1/3 | 0.4962 | 19.24 |
| axis_silu | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (4/24) | Not run | 0/3 | 0.0974 | 19.94 |
| oriented4_silu | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.2864 | 18.34 |
| oriented8_silu | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.3094 | 17.73 |
| axis_softplus1 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.4304 | 18.72 |
| oriented8_softplus1 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.5474 | 19.40 |

## Architecture declarations

All critics retain width64 and two hidden layers. Axis features exactly retain the original 10-dimensional input (raw2 + sin/cos at pi and2pi on each axis). Random-oriented4 features also yield10 dimensions. Random-oriented8 yields18 dimensions and modestly increases D parameters. Every orientation uses one fixed local seed0, no target data and no seed selection.

| D | Features | Activation | Radial frequencies / pi | Parameters |
| --- | --- | --- | --- | ---: |
| axis_silu | axis | silu | axis1,2 | 4929 |
| axis_softplus1 | axis | softplus beta1.0 | axis1,2 | 4929 |
| axis_softplus5 | axis | softplus beta5.0 | axis1,2 | 4929 |
| axis_tanh | axis | tanh | axis1,2 | 4929 |
| oriented4_silu | oriented | silu | [1.0, 1.0, 2.0, 2.0] | 4929 |
| oriented8_silu | oriented | silu | [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0] | 5441 |
| oriented8_softplus1 | oriented | softplus beta1.0 | [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0] | 5441 |
| oriented8_tanh | oriented | tanh | [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0] | 5441 |

Actual GAN episodes: **30**; fixed live observations: **720**; summed recorded wall time: **193.45s** on a shared CPU host.

Full cards/original and effective specs/verdicts and links to complete live+EMA curves/actions/update counts: [index.json](index.json.gz). All failures retained. Architecture derivative/reproducibility checks are static tests, not GAN training episodes.

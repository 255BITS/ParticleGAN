# Fixed-sigma particle routing scout

Rank: modes covered descending, then high-quality fraction descending, then sample SW1 ascending. Final online weights; no best-checkpoint selection.

| Rank | Arm | Steps | Modes /100 | HQ % ↑ | Width /real ≈1 | Balance TV ↓ | SW1 ↓ | Recon MSE ↓ | Used /400 | Offset RMS ≈1 | Train sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_bounded | 6000 | 92 | 82.23 | 0.694 | 0.2631 | 0.3911 | 0.002775 | 347 | 0.1772 | 38.99 |
| 2 | route_offset | 6000 | 91 | 76.66 | 0.7223 | 0.3168 | 0.483 | 0.005378 | 274 | 0.2033 | 37.86 |
| 3 | route_noise | 6000 | 89 | 76.63 | 0.7798 | 0.3267 | 0.5109 | 0.004865 | 277 | 1.002 | 38.16 |
| 4 | route_balanced | 6000 | 85 | 64.29 | 0.752 | 0.3331 | 0.4327 | 0.005167 | 238 | 1.171 | 40.46 |
| 5 | gan | 6000 | 80 | 59.09 | 0.7466 | 0.36 | 0.2426 | — | — | — | 28.91 |
| 6 | route_grad100 | 6000 | 77 | 65.55 | 0.7524 | 0.3973 | 0.4437 | 0.008043 | 287 | 0.2102 | 37.78 |
| 7 | route_zero | 6000 | 71 | 46.21 | 0.7485 | 0.4268 | 0.3834 | 0.008368 | 304 | 0 | 38.08 |

Width is the existing per-mode core-radius metric divided by a fresh real-data reference; near 1 is desirable. Inspect it alongside HQ and balance, not rank alone.
Reconstruction MSE averages both coordinates. HQ means within 0.09 of a true center; coverage requires ≥10 HQ samples per mode. TV measures imbalance among HQ samples.
One shared seed, distinct mechanisms. This is a scout, not a statistical superiority claim. Sigma is fixed; offsets and means may learn. No KL is used. Only route_balanced adds an aggregate particle-usage loss; no arm matches offset distributions.

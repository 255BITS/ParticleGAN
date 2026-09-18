# Generator learning-rate scout

2/2 certified. Same CNN E-only 10k parent: FID50k 19.4482.

| Arm | FID at 15k | FID at 20k | Test MSE | Training minutes | Wall minutes |
|---|---:|---:|---:|---:|---:|
| control | 19.5589 | 20.3044 | 0.14846 | 7.20 | 9.59 |
| half_g | 20.7412 | 20.6996 | 0.14310 | 7.34 | 9.73 |

Final FID changes versus matched control: half_g +0.3952.

Only G learning rate changes (0.0003 to 0.00015). E 0.0003, prior 0.003, D 0.00045; one D step, coefficient 1, lazy bcap every 8. Full Adam/EMA/RNG restored. Actual optimizer rates and matched RNG consumption audited. FID uses 50k EMA samples and the unchanged CIFAR train reference. No seed replicates or automatic long promotion.

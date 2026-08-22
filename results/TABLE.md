# Regularizer study -- aggregate results

185 runs loaded from disk. Skipped: 0 without summary.json, 0 with unreadable/incomplete summary.json, 0 with unparseable run names, 0 loaded but missing/empty metrics.jsonl (final-only; their convergence-speed entries are censored).


## Main grid (lr_mult 1.0, every-step penalty)

| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_r1r2 | 1 | 5 | 0.1834 ± 0.0217 | 0.3625 ± 0.0298 | 1.000 ± 0.000 | 100.0 | 0.598 | 14.832 ± 0.717 | 0.00 (0.00) | >5900 ± 1377 (3 censored) | 1.000 ± 0.000 (max 1.000) | 0.0004 | 0.229 | 0.045 / 0.078 | 111 |
| a_r1r2 | 0.3 | 5 | 0.2294 ± 0.0432 | 0.4708 ± 0.0548 | 1.000 ± 0.000 | 100.0 | 0.868 | 7.818 ± 1.253 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.000) | 0.0002 | 0.600 | 0.095 / 0.173 | 143 |
| a_r1r2 | 0.1 | 5 | 0.3381 ± 0.0495 | 0.6077 ± 0.0565 | 0.984 ± 0.010 | 100.0 | 0.943 | 11.403 ± 2.989 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.000) | 0.0002 | 1.918 | 0.186 / 0.312 | 110 |
| b_cap | 1 | 5 | 0.3675 ± 0.0282 | 0.6581 ± 0.0285 | 0.976 ± 0.005 | 100.0 | 0.967 | 27.021 ± 7.591 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0003 | 39.362 | 0.675 / 0.822 | 108 |
| d_asym | 1 | 5 | 0.4139 ± 0.0304 | 0.7201 ± 0.0387 | 0.966 ± 0.014 | 100.0 | 0.962 | 60.984 ± 17.525 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0002 | 86.517 | 0.938 / 0.958 | 104 |
| b_cap | 0.1 | 5 | 0.3944 ± 0.0462 | 0.7017 ± 0.0547 | 0.954 ± 0.016 | 100.0 | 0.950 | 62.064 ± 20.939 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0002 | 39.602 | 0.878 / 1.163 | 107 |
| c_eikonal | 0.1 | 5 | 0.4522 ± 0.0659 | 0.7626 ± 0.0618 | 0.952 ± 0.026 | 100.0 | 0.962 | 72.685 ± 13.015 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0004 | 72.609 | 1.037 / 1.201 | 102 |
| e_interp | 1 | 5 | 0.4564 ± 0.0675 | 0.7923 ± 0.0685 | 0.960 ± 0.011 | 100.0 | 0.954 | 110.584 ± 28.025 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0004 | 112.493 | 0.972 / 1.058 | 113 |
| d_asym | 0.1 | 5 | 0.4581 ± 0.1049 | 0.7723 ± 0.0967 | 0.964 ± 0.015 | 100.0 | 0.955 | 83.236 ± 22.281 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0010 | 46.384 | 0.908 / 1.096 | 106 |
| e_interp | 0.1 | 5 | 0.4652 ± 0.0438 | 0.8125 ± 0.0448 | 0.952 ± 0.023 | 100.0 | 0.934 | 135.691 ± 19.780 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0005 | 132.406 | 2.031 / 1.797 | 98 |
| c_eikonal | 1 | 5 | 0.4570 ± 0.0784 | 0.7620 ± 0.0811 | 0.948 ± 0.026 | 100.0 | 0.958 | 59.635 ± 29.564 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.001 (max 1.003) | 0.0006 | 76.636 | 0.993 / 1.016 | 100 |
| e_interp | 0.02 | 5 | 0.5010 ± 0.0723 | 0.8878 ± 0.0481 | 0.954 ± 0.019 | 100.0 | 0.877 | 234.299 ± 26.659 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0007 | 646.045 | 5.545 / 4.182 | 86 |
| a_r1r2 | 0.02 | 5 | 0.6305 ± 0.1508 | 0.8896 ± 0.1198 | 0.858 ± 0.058 | 98.2 | 0.940 | 50.465 ± 23.144 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0002 | 36.284 | 0.559 / 1.028 | 108 |
| b_cap | 0.02 | 5 | 0.5764 ± 0.0721 | 0.8980 ± 0.0604 | 0.934 ± 0.033 | 100.0 | 0.924 | 130.535 ± 13.712 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0005 | 23.505 | 1.237 / 1.636 | 110 |
| e_interp | 0.005 | 5 | 0.6187 ± 0.0468 | 1.0218 ± 0.0340 | 0.940 ± 0.024 | 100.0 | 0.815 | 342.193 ± 55.059 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.000 (max 1.002) | 0.0006 | 2277.770 | 11.860 / 8.657 | 87 |
| d_asym | 0.02 | 5 | 1.1245 ± 1.1318 | 1.5004 ± 1.2091 | 0.802 ± 0.277 | 85.0 | 0.925 | 155.583 ± 21.000 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0005 | 23.347 | 1.217 / 1.556 | 105 |
| c_eikonal | 0.02 | 5 | 0.7474 ± 0.1692 | 1.0578 ± 0.1432 | 0.892 ± 0.050 | 99.8 | 0.923 | 153.475 ± 29.404 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0006 | 34.395 | 1.307 / 1.626 | 103 |
| a_r1r2 | 0.005 | 5 | 1.4886 ± 1.0045 | 1.8738 ± 1.0690 | 0.712 ± 0.256 | 82.6 | 0.888 | 249.242 ± 59.534 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0004 | 7.469 | 1.336 / 1.901 | 107 |
| c_eikonal | 0.005 | 5 | 1.2968 ± 0.5060 | 1.7218 ± 0.5685 | 0.744 ± 0.175 | 83.6 | 0.874 | 316.605 ± 29.882 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.002) | 0.0005 | 19.718 | 2.174 / 3.031 | 104 |
| d_asym | 0.005 | 5 | 1.7853 ± 0.8849 | 2.2238 ± 0.9760 | 0.594 ± 0.227 | 68.0 | 0.851 | 231.009 ± 19.630 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.002) | 0.0043 | 67.119 | 1.929 / 2.832 | 103 |
| b_cap | 0.005 | 5 | 1.7945 ± 0.8902 | 2.2202 ± 0.9823 | 0.596 ± 0.261 | 69.0 | 0.891 | 224.082 ± 86.267 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.001 (max 1.003) | 0.0004 | 8.909 | 1.764 / 2.396 | 113 |
| f_none | 0 | 5 | 2.0983 ± 0.6230 | 2.5683 ± 0.6664 | 0.344 ± 0.234 | 47.4 | 0.899 | 223.000 ± 242.714 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.055 ± 0.052 (max 1.151) | 0.0002 | 36.275 | 18.943 / 43.944 | 81 |

## Lazy variant: penalty every 16 steps at the nominal coeff (16x internal rescale)

| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b_cap | 1 | 5 | 0.4425 ± 0.0679 | 0.7280 ± 0.0589 | 0.948 ± 0.033 | 100.0 | 0.960 | 39.806 ± 9.285 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0003 | 40.271 | 0.577 / 0.666 | 113 |
| c_eikonal | 0.1 | 5 | 0.4459 ± 0.0290 | 0.7573 ± 0.0286 | 0.938 ± 0.020 | 100.0 | 0.960 | 79.800 ± 26.193 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0004 | 120.942 | 0.917 / 0.968 | 167 |

## Lazy control: penalty every step at coeff/16 (same nominal coeff in the label)

| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b_cap | 1 | 5 | 0.4722 ± 0.0989 | 0.7754 ± 0.0782 | 0.948 ± 0.012 | 100.0 | 0.946 | 79.841 ± 6.801 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0002 | 36.747 | 0.899 / 1.237 | 161 |
| c_eikonal | 0.1 | 5 | 1.0127 ± 0.4255 | 1.3550 ± 0.3646 | 0.848 ± 0.076 | 99.2 | 0.872 | 347.652 ± 88.340 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.000 (max 1.002) | 0.0007 | 44.522 | 2.252 / 2.742 | 276 |

## LR sensitivity: lr_mult = 0.5

| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_r1r2 | 1 | 5 | 0.2137 ± 0.0259 | 0.3895 ± 0.0382 | 1.000 ± 0.000 | 100.0 | 0.562 | 24.049 ± 0.843 | 0.00 (0.00) | >6840 ± 320 (4 censored) | 1.000 ± 0.000 (max 1.000) | 0.0004 | 0.485 | 0.043 / 0.061 | 211 |
| b_cap | 1 | 5 | 0.4034 ± 0.0486 | 0.7015 ± 0.0521 | 0.970 ± 0.024 | 100.0 | 0.931 | 59.240 ± 12.660 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.001) | 0.0002 | 5.188 | 0.549 / 0.519 | 258 |
| a_r1r2 | 0.1 | 5 | 0.4402 ± 0.0550 | 0.7081 ± 0.0483 | 0.936 ± 0.017 | 100.0 | 0.909 | 19.250 ± 4.249 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.000) | 0.0002 | 3.185 | 0.228 / 0.301 | 198 |
| d_asym | 1 | 5 | 0.4685 ± 0.0208 | 0.7862 ± 0.0501 | 0.942 ± 0.007 | 100.0 | 0.914 | 98.233 ± 44.993 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.001) | 0.0003 | 42.813 | 0.949 / 0.930 | 191 |
| c_eikonal | 0.1 | 5 | 0.5061 ± 0.0731 | 0.8269 ± 0.0735 | 0.912 ± 0.035 | 99.8 | 0.930 | 90.134 ± 25.705 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.000 ± 0.000 (max 1.000) | 0.0002 | 39.957 | 1.026 / 1.019 | 197 |

## LR sensitivity: lr_mult = 2.0

| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_r1r2 | 0.3 | 5 | 0.1899 ± 0.0360 | 0.4197 ± 0.0431 | 1.000 ± 0.000 | 100.0 | 0.841 | 8.425 ± 1.085 | 0.00 (0.00) | >6580 ± 840 (4 censored) | 1.000 ± 0.000 (max 1.001) | 0.0002 | 1.027 | 0.118 / 0.263 | 165 |
| a_r1r2 | 1 | 5 | 0.1991 ± 0.0202 | 0.3822 ± 0.0286 | 1.000 ± 0.000 | 100.0 | 0.538 | 18.712 ± 1.789 | 0.00 (0.00) | >6660 ± 680 (4 censored) | 1.000 ± 0.000 (max 1.000) | 0.0005 | 1.381 | 0.042 / 0.092 | 192 |
| a_r1r2 | 0.1 | 5 | 0.2608 ± 0.0405 | 0.5282 ± 0.0409 | 1.000 ± 0.000 | 100.0 | 0.943 | 14.345 ± 2.216 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.000 (max 1.001) | 0.0001 | 4.746 | 0.272 / 0.525 | 200 |
| b_cap | 1 | 5 | 0.2901 ± 0.0240 | 0.5734 ± 0.0183 | 0.998 ± 0.004 | 100.0 | 0.986 | 17.726 ± 5.024 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.001 (max 1.005) | 0.0004 | 315.205 | 0.807 / 0.953 | 307 |
| d_asym | 1 | 5 | 0.3377 ± 0.0328 | 0.6271 ± 0.0288 | 0.996 ± 0.005 | 100.0 | 0.986 | 21.872 ± 3.839 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.001 ± 0.001 (max 1.002) | 0.0011 | 345.089 | 0.919 / 1.002 | 168 |
| c_eikonal | 0.1 | 5 | 0.4143 ± 0.0409 | 0.7000 ± 0.0410 | 0.984 ± 0.016 | 100.0 | 0.953 | 42.822 ± 6.977 | 0.00 (0.00) | >7000 ± 0 (5 censored) | 1.002 ± 0.001 (max 1.002) | 0.0013 | 318.352 | 1.137 / 1.379 | 189 |

## Footnotes

1. **Rows are sorted by median final exact W1 within each variant group** (best first). `± ` is the population std over seeds.

2. **steps→thresh is measured on the *sliced* W1 series, against a sliced threshold.** The headline quality number is the exact W1, but metrics.jsonl only carries `sliced_w1` per eval step -- there is no exact-W1 trace to threshold. Rather than fitting a per-run sliced→exact calibration, the speed statistic is defined entirely on the sliced scale: `sliced_threshold = 2 x min_coeff(mean final w1_sliced of a_r1r2) = 0.06783`. The corresponding exact-W1 threshold (`2 x min_coeff(mean final w1_exact of a_r1r2)` = 0.36682) is reported for reference but is NOT used for the timeseries. The two are on different scales and must not be compared to each other.

3. Seeds that never reach the threshold are counted as `total_steps` and the cell is prefixed with `>`; the count of such censored seeds is shown in parentheses. Means over censored groups are lower bounds.

4. `collapse rate` is the fraction of seeds with `collapse_events > 0`; the parenthesized number is the mean event count per seed.

5. `dom. modulus` is the modulus of the dominant eigenvalue of the local game Jacobian; > 1 indicates a locally divergent (non-contractive) equilibrium. Runs whose `spectral` block is missing or an error record are omitted from that column only.

6. `med_nr / med_nf @mid` are the median discriminator gradient norms at real and fake samples, taken at the mid-training checkpoint recorded in `summary.mid`.


# Stress solvability search

All attempts: fixed seed0, original thresholds,24 live checks, final passing suffix>=5. EMA separate. Seen cadence is development evidence; this search is not fresh transfer.

| Card | Task | Resources | Final | Stable suffix | SW1 | Mass TV | HQ | Cov error | Min eigen | Seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| budget2 | stress_fast_critic | more updates | PASS | 10/24 | 0.0473 | 0.0566 | 0.9846 | 0.5392 | 0.3045 | 15.03 |
| budget3 | stress_fast_critic | more updates | PASS | 7/24 | 0.0411 | 0.0566 | 0.9902 | 0.3032 | 0.5610 | 21.19 |
| budget3 | stress_large_critic | more updates | PASS | 8/24 | 0.0798 | 0.1321 | 0.9895 | 0.3803 | 0.3877 | 22.22 |
| budget3 | stress_long_horizon | more updates | PASS | 6/24 | 0.0771 | 0.0840 | 0.9919 | 0.4580 | 0.1627 | 34.34 |
| budget3 | stress_small_batch | more updates | PASS | 11/24 | 0.0539 | 0.1050 | 0.9883 | 0.2828 | 0.2684 | 16.11 |
| prior_lr_30 | stress_fast_critic | same architecture/data/update budget | PASS | 6/24 | 0.0536 | 0.0891 | 0.9922 | 0.4542 | 0.2585 | 7.31 |
| slow_g_every2_budget5 | stress_slow_critic | changed update cadence + more outer steps (D6000/G3000) | PASS | 11/24 | 0.0448 | 0.1392 | 1.0000 | 0.2384 | 0.6403 | 24.14 |
| stockish_extended | stress_fast_critic | more updates + changed capacity | PASS | 19/24 | 0.0309 | 0.0266 | 0.9861 | 0.3743 | 0.3527 | 30.25 |
| stockish_extended | stress_r1_r2 | more updates + changed capacity | PASS | 16/24 | 0.0530 | 0.1113 | 0.9941 | 0.1158 | 0.7558 | 21.52 |
| budget2 | stress_long_horizon | more updates | PASS | 4/24 | 0.0799 | 0.0889 | 0.9763 | 0.4508 | 0.4326 | 21.17 |
| slow_g_every2_budget3 | stress_slow_critic | changed update cadence + more outer steps (D3600/G1800) | PASS | 2/24 | 0.0453 | 0.1392 | 0.9868 | 0.8343 | 0.5477 | 13.58 |
| stockish_extended | stress_long_horizon | more updates + changed capacity | FAIL | 0/24 | 0.0414 | 0.0442 | 0.9922 | 0.4618 | 0.1364 | 50.56 |
| prior_reg1_lr1 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0355 | 0.0493 | 0.9370 | 1.0031 | 0.3163 | 7.29 |
| beta2_0p99 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0702 | 0.1877 | 0.9812 | 0.3515 | 0.4641 | 7.78 |
| stockish_extended | stress_small_batch | more updates + changed capacity | FAIL | 0/24 | 0.0784 | 0.0840 | 0.9958 | 0.6687 | 0.1032 | 20.54 |
| stockish_extended | stress_large_critic | more updates + changed capacity | FAIL | 0/24 | 0.0478 | 0.0647 | 0.9971 | 0.6553 | 0.0906 | 36.52 |
| stockish_extended | reserved_alternating_critic_updates | more updates + changed capacity | FAIL | 0/24 | 0.0584 | 0.1384 | 0.9792 | 1.6086 | 0.1526 | 31.15 |
| budget2 | stress_large_critic | more updates | FAIL | 0/24 | 0.0891 | 0.1460 | 0.9812 | 2.0631 | 0.6615 | 12.83 |
| lr1p5 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0670 | 0.1523 | 0.9844 | 2.2467 | 0.2166 | 8.23 |
| budget3 | stress_r1_r2 | more updates | FAIL | 0/24 | 0.0612 | 0.1785 | 0.9668 | 2.2998 | 0.5410 | 16.86 |
| prior_reg_zero | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0851 | 0.1287 | 0.9717 | 2.7054 | 0.6385 | 7.42 |
| cap2_k1 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0609 | 0.1182 | 0.9504 | 3.1914 | 0.3799 | 6.92 |
| prior_lr_30 | stress_large_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0535 | 0.1099 | 0.9814 | 3.3633 | 0.7466 | 12.76 |
| baseline | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0457 | 0.0566 | 0.9805 | 3.3686 | 0.3632 | 8.35 |
| prior_reg_0p2 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0667 | 0.1375 | 0.9851 | 3.4727 | 0.4901 | 7.62 |
| stockish_extended | stress_slow_critic | more updates + changed capacity | FAIL | 0/24 | 0.0685 | 0.1055 | 0.9487 | 3.7251 | 0.7546 | 28.55 |
| particles1024 | stress_fast_critic | changed capacity | FAIL | 0/24 | 0.0584 | 0.0952 | 0.9524 | 3.8329 | 0.3725 | 6.64 |
| cap1_k1 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0528 | 0.0867 | 0.9602 | 4.0673 | 0.4905 | 7.84 |
| prior_reg_3 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0495 | 0.0801 | 0.9685 | 3.2955 | 0.0072 | 6.56 |
| budget2 | stress_small_batch | more updates | FAIL | 0/24 | 0.0591 | 0.1050 | 0.9756 | 4.1970 | 0.4059 | 10.06 |
| budget3 | reserved_alternating_critic_updates | more updates | FAIL | 0/24 | 0.0492 | 0.0813 | 0.9675 | 4.2128 | 0.1908 | 23.02 |
| budget3 | stress_slow_critic | more updates | FAIL | 0/24 | 0.0791 | 0.1694 | 0.9717 | 4.1042 | 0.6093 | 17.06 |
| prior_lr_30 | stress_long_horizon | same architecture/data/update budget | FAIL | 0/24 | 0.1018 | 0.1980 | 0.9524 | 4.4633 | 0.5009 | 18.14 |
| prior_lr_3 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.1005 | 0.1367 | 0.9438 | 4.7599 | 0.2830 | 6.30 |
| budget2 | reserved_alternating_critic_updates | more updates | FAIL | 0/24 | 0.0611 | 0.1116 | 0.9695 | 5.1059 | 0.5735 | 17.77 |
| beta2_0p9 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0631 | 0.1265 | 0.9624 | 5.6697 | 0.4225 | 7.57 |
| particles4096 | stress_fast_critic | changed capacity | FAIL | 0/24 | 0.0439 | 0.0525 | 0.8979 | 5.6767 | 0.5224 | 7.17 |
| prior_lr_1 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0405 | 0.0715 | 0.8638 | 6.0115 | 0.5840 | 6.11 |
| beta2_0p99 | stress_long_horizon | same architecture/data/update budget | FAIL | 0/24 | 0.0831 | 0.0752 | 0.9614 | 6.4060 | 0.3236 | 12.41 |
| cap10 | stress_long_horizon | same architecture/data/update budget | FAIL | 0/24 | 0.0540 | 0.0850 | 0.9351 | 7.1641 | 0.5159 | 11.17 |
| density4096_slow | stress_slow_critic | more particles + larger batch + more updates | FAIL | 0/24 | 0.0787 | 0.1099 | 0.9287 | 8.3375 | 0.6087 | 35.45 |
| fourier3 | stress_fast_critic | changed capacity | FAIL | 0/24 | 0.0750 | 0.1467 | 0.9519 | 8.4411 | 0.4802 | 7.02 |
| budget2 | stress_r1_r2 | more updates | FAIL | 0/24 | 0.0705 | 0.1865 | 0.9290 | 8.2354 | 0.9223 | 11.41 |
| cap5_k1p25 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0680 | 0.0815 | 0.8989 | 9.7893 | 0.2623 | 7.07 |
| prior_reg_1 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0811 | 0.1423 | 0.9197 | 10.0555 | 0.8081 | 6.41 |
| slow_prior_lr3_budget3 | stress_slow_critic | more updates | FAIL | 0/24 | 0.0837 | 0.1111 | 0.9553 | 10.4415 | 0.6225 | 17.22 |
| cap10 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0929 | 0.1365 | 0.8806 | 11.3326 | 0.0637 | 6.11 |
| budget2 | stress_slow_critic | more updates | FAIL | 0/24 | 0.0868 | 0.1768 | 0.9275 | 12.0413 | 0.6909 | 10.87 |
| prior_lr_30 | stress_r1_r2 | same architecture/data/update budget | FAIL | 0/24 | 0.0620 | 0.1418 | 0.9187 | 12.8011 | 0.3778 | 9.52 |
| prior_reg1_lr3 | stress_fast_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0821 | 0.1631 | 0.8684 | 14.6499 | 0.6844 | 6.66 |
| prior_lr_30 | reserved_alternating_critic_updates | same architecture/data/update budget | FAIL | 0/24 | 0.1314 | 0.1499 | 0.8792 | 16.0684 | 0.4917 | 13.21 |
| fourier3_particles1024 | stress_fast_critic | changed capacity | FAIL | 0/24 | 0.0492 | 0.0811 | 0.8625 | 16.8263 | 1.1931 | 6.88 |
| beta2_0p99 | stress_r1_r2 | same architecture/data/update budget | FAIL | 0/24 | 0.0855 | 0.2346 | 0.8518 | 16.8094 | 0.7625 | 6.29 |
| beta2_0p99 | stress_large_critic | same architecture/data/update budget | FAIL | 0/24 | 0.1283 | 0.1960 | 0.9280 | 17.3153 | 0.6676 | 7.02 |
| beta2_0p99 | stress_small_batch | same architecture/data/update budget | FAIL | 0/24 | 0.0745 | 0.1011 | 0.8584 | 19.1648 | 0.5984 | 5.66 |
| beta2_0p99 | reserved_alternating_critic_updates | same architecture/data/update budget | FAIL | 0/24 | 0.0856 | 0.1809 | 0.8372 | 20.8833 | 0.5437 | 8.19 |
| prior_lr_30 | stress_small_batch | same architecture/data/update budget | FAIL | 0/24 | 0.1096 | 0.2351 | 0.8489 | 20.9116 | 0.2663 | 10.39 |
| cap10 | stress_large_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0738 | 0.0981 | 0.8162 | 22.9977 | 0.3308 | 6.71 |
| cap10 | stress_r1_r2 | same architecture/data/update budget | FAIL | 0/24 | 0.0842 | 0.1892 | 0.8398 | 23.1169 | 1.0302 | 5.64 |
| prior_lr_30 | stress_slow_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0629 | 0.0730 | 0.8064 | 28.0275 | 0.8208 | 13.80 |
| beta2_0p99 | stress_slow_critic | same architecture/data/update budget | FAIL | 0/24 | 0.0958 | 0.0872 | 0.7480 | 33.4492 | 0.8830 | 7.29 |
| cap10 | reserved_alternating_critic_updates | same architecture/data/update budget | FAIL | 0/24 | 0.0956 | 0.1809 | 0.7852 | 33.5195 | 0.4525 | 7.88 |
| cap10 | stress_small_batch | same architecture/data/update budget | FAIL | 0/24 | 0.1482 | 0.1050 | 0.7283 | 37.0022 | 0.2507 | 5.48 |
| fourier4 | stress_fast_critic | changed capacity | FAIL | 0/24 | 0.1430 | 0.1558 | 0.4939 | 39.6812 | 1.0175 | 6.45 |
| cap10 | stress_slow_critic | same architecture/data/update budget | FAIL | 0/24 | 0.1567 | 0.1277 | 0.5430 | 74.8021 | 0.9749 | 6.33 |
| slow_prior_lr3 | stress_slow_critic | same architecture/data/update budget | FAIL | 0/24 | 0.1916 | 0.1304 | 0.5015 | 86.3941 | 1.4794 | 5.85 |

# Configuration search ledger

Every completed attempt is listed, including failures. The seed and gate thresholds were fixed. Times include evaluation and are device-specific. A row passing one problem does not certify the suite.

[Configs, provenance, and complete evaluation curves](trials.json)

| Trial | Problem | Device | Budget | Modes | HQ | Mass TV | First 100 | Stable from | Gate | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| direct-disk7 | grid100 | cpu | 4000 | 85 | 0.8672 | 0.1867 | — | — | FAIL | 34.9 |
| direct-uniform5-f2 | grid100 | cpu | 4000 | 100 | 0.9787 | 0.0425 | 500 | — | FAIL | 38.0 |
| direct-uniform5-f3 | grid100 | cpu | 4000 | 100 | 0.9793 | 0.0478 | 750 | 3000 | PASS | 36.4 |
| direct-uniform5-noisy | grid100 | cpu | 4000 | 100 | 0.9637 | 0.0534 | 3000 | — | FAIL | 35.3 |
| direct-uniform5-small-lr | grid100 | cpu | 3000 | 100 | 0.9773 | 0.0424 | 1000 | — | FAIL | 26.5 |
| instance-noise/control | grid100 | cuda:0 | 7000 | 97 | 0.9873 | 0.1206 | — | — | FAIL | 72.6 |
| instance-noise/sigma05 | grid100 | cuda:0 | 7000 | 100 | 0.9900 | 0.0665 | 5500 | 6000 | PASS | 67.7 |
| instance-noise/sigma05 | rotated100 | cuda:0 | 7000 | 100 | 0.9878 | 0.0577 | 5500 | 6000 | PASS | 79.3 |
| instance-noise/sigma05 | staggered100 | cuda:0 | 7000 | 99 | 0.9807 | 0.1052 | — | — | FAIL | 91.4 |
| instance-noise-cpu/sigma05 | grid100 | cpu | 7000 | 100 | 0.9812 | 0.0834 | 5750 | 6000 | PASS | 122.7 |
| instance-noise-cpu/sigma05 | rotated100 | cpu | 7000 | 100 | 0.9892 | 0.0546 | 5250 | 6000 | PASS | 122.7 |
| instance-noise-cpu/sigma05 | staggered100 | cpu | 7000 | 99 | 0.9765 | 0.0952 | — | — | FAIL | 123.6 |
| instance-noise-repair/sigma05_prior1 | staggered100 | cuda:0 | 7000 | 96 | 0.9775 | 0.1020 | — | — | FAIL | 69.5 |
| instance-noise-repair/sigma075_long_prior005 | staggered100 | cuda:0 | 7000 | 99 | 0.9765 | 0.0910 | — | — | FAIL | 73.0 |
| instance-noise-repair/sigma10_prior005 | staggered100 | cuda:0 | 7000 | 95 | 0.9691 | 0.1247 | — | — | FAIL | 72.0 |
| noisy-allocation/batch1024 | grid100 | cuda:0 | 7000 | 99 | 0.9882 | 0.0934 | — | — | FAIL | 81.6 |
| noisy-allocation/latent16 | grid100 | cuda:0 | 7000 | 99 | 0.9923 | 0.1062 | — | — | FAIL | 77.1 |
| noisy-allocation/particles40000 | grid100 | cuda:0 | 7000 | 99 | 0.9912 | 0.1103 | — | — | FAIL | 76.6 |
| noisy-allocation/prior_reg1 | grid100 | cuda:0 | 7000 | 100 | 0.9872 | 0.0896 | 5750 | — | FAIL | 71.7 |
| noisy-mlp | grid100 | cpu | 7000 | 95 | 0.9933 | 0.1139 | — | — | FAIL | 107.4 |
| recommended | grid100 | cpu | 7000 | 100 | 0.9812 | 0.0834 | 5750 | 6000 | PASS | 98.8 |
| recommended | rotated100 | cpu | 7000 | 100 | 0.9892 | 0.0546 | 5250 | 6000 | PASS | 99.9 |
| recommended | staggered100 | cpu | 7000 | 100 | 0.9826 | 0.0750 | 5500 | 6000 | PASS | 176.1 |
| search-architecture/hist_f3_w128_d3 | grid100 | cuda:0 | 7000 | 98 | 0.9690 | 0.1221 | — | — | FAIL | 70.1 |
| search-architecture/hist_f4_w128_d3 | grid100 | cuda:0 | 7000 | 92 | 0.9840 | 0.1567 | — | — | FAIL | 69.9 |
| search-architecture/hist_f4_w128_d4 | grid100 | cuda:0 | 7000 | 93 | 0.9742 | 0.1623 | — | — | FAIL | 83.1 |
| search-architecture/hist_f4_w192_d3 | grid100 | cuda:0 | 7000 | 79 | 0.8667 | 0.1743 | — | — | FAIL | 72.3 |
| search-architecture/hist_f6_w128_d3 | grid100 | cuda:0 | 7000 | 73 | 0.8807 | 0.2060 | — | — | FAIL | 71.8 |
| search-recipe/historical_beta099 | grid100 | cuda:0 | 7000 | 97 | 0.9792 | 0.1313 | — | — | FAIL | 66.6 |
| search-recipe/historical_cap6 | grid100 | cuda:0 | 7000 | 80 | 0.9326 | 0.1900 | — | — | FAIL | 70.3 |
| search-recipe/historical_d1 | grid100 | cuda:0 | 7000 | 92 | 0.9876 | 0.1595 | — | — | FAIL | 66.6 |
| search-recipe/historical_lr0006 | grid100 | cuda:0 | 7000 | 98 | 0.9867 | 0.1199 | — | — | FAIL | 69.6 |
| search-recipe/historical_lr0012 | grid100 | cuda:0 | 7000 | 93 | 0.9841 | 0.1504 | — | — | FAIL | 68.6 |
| search-recipe/historical_prior2 | grid100 | cuda:0 | 7000 | 97 | 0.9865 | 0.1297 | — | — | FAIL | 68.7 |
| search-recipe/historical_reg005 | grid100 | cuda:0 | 7000 | 95 | 0.9761 | 0.1538 | — | — | FAIL | 72.2 |
| search-schedule/direct_disk65_grid | grid100 | cpu | 4000 | 97 | 0.8187 | 0.1811 | — | — | FAIL | 35.5 |
| search-schedule/direct_f3_rotated | rotated100 | cpu | 4000 | 88 | 0.8162 | 0.1490 | — | — | FAIL | 35.8 |
| search-schedule/direct_f3_staggered | staggered100 | cpu | 4000 | 100 | 0.9313 | 0.1061 | 1000 | — | FAIL | 37.9 |
| search-schedule/direct_fastaffine_rotated | rotated100 | cpu | 4000 | 88 | 0.8215 | 0.1462 | — | — | FAIL | 36.1 |
| search-schedule/grid7k/batch_1024 | grid100 | cpu | 7000 | 100 | 0.9819 | 0.1164 | 6000 | — | FAIL | 177.1 |
| search-schedule/grid7k/batch_128 | grid100 | cpu | 7000 | 98 | 0.9731 | 0.1338 | — | — | FAIL | 48.7 |
| search-schedule/grid7k/batch_512 | grid100 | cpu | 7000 | 99 | 0.9817 | 0.1146 | — | — | FAIL | 102.4 |
| search-schedule/grid7k/constant_lr | grid100 | cpu | 7000 | 2 | 0.2288 | 0.1092 | — | — | FAIL | 65.5 |
| search-schedule/grid7k/schedule_000 | grid100 | cpu | 7000 | 95 | 0.9632 | 0.1225 | — | — | FAIL | 68.2 |
| search-schedule/grid7k/schedule_025 | grid100 | cpu | 7000 | 99 | 0.9690 | 0.1182 | — | — | FAIL | 67.4 |
| search-schedule/grid7k/schedule_040 | grid100 | cpu | 7000 | 97 | 0.9716 | 0.1156 | — | — | FAIL | 66.4 |
| search-schedule/instance-noise-batch1024/sigma05 | staggered100 | cpu | 7000 | 100 | 0.9826 | 0.0750 | 5500 | 6000 | PASS | 182.5 |
| search-schedule/noisy_mlp_8k | grid100 | cpu | 8000 | 96 | 0.9952 | 0.1081 | — | — | FAIL | 115.2 |
| search-schedule/noisy_mlp_anneal025 | grid100 | cpu | 7000 | 96 | 0.9911 | 0.1089 | — | — | FAIL | 120.9 |
| search-schedule/promoted/batch_1024 | rotated100 | cpu | 7000 | 99 | 0.9727 | 0.0902 | — | — | FAIL | 180.1 |
| search-schedule/promoted/batch_1024 | staggered100 | cpu | 7000 | 99 | 0.9817 | 0.1080 | — | — | FAIL | 192.8 |
| staggered-repairs-cpu/fourier3/sigma05 | staggered100 | cpu | 7000 | 98 | 0.9740 | 0.1188 | — | — | FAIL | 114.2 |
| staggered-repairs-cpu/less_input_noise/sigma05 | staggered100 | cpu | 7000 | 100 | 0.9805 | 0.0875 | 5750 | — | FAIL | 115.7 |
| staggered-repairs-cpu/longer_coarse_phase/sigma05 | staggered100 | cpu | 7000 | 100 | 0.9889 | 0.0599 | 5750 | — | FAIL | 116.6 |

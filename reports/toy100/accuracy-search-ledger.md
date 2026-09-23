# Accuracy search ledger

Every discovered 100-mode trial is retained, including failures and interrupted runs. The final fidelity column scores one saved 20,000-sample draw; it does **not** certify five-checkpoint/100,000-sample accuracy or the common 22-toy gate. All trials use the frozen training seed; no seed sweep is performed.

| Run | Steps | Coverage gate | Final modes | HQ | Mass TV | Center / σ | Width bias | Radial KS | Final fidelity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `affine-normal/grid100` | 7000 | FAIL | 96 | 0.9817 | 0.1183 | 0.1891 | -0.0183 | 0.0088 | FAIL |
| `affine-square-rotated-beta999/rotated100` | 7000 | FAIL | 5 | 0.0874 | 0.2984 | — | — | — | FAIL |
| `ci-original-coverage/artifacts/toy100-ci/grid100` | 7000 | PASS | 100 | 0.9812 | 0.0835 | 0.6638 | -0.1794 | 0.0258 | FAIL |
| `ci-original-coverage/artifacts/toy100-ci/rotated100` | 7000 | PASS | 100 | 0.9892 | 0.0545 | 0.1749 | -0.1521 | 0.0634 | FAIL |
| `ci-original-coverage/artifacts/toy100-ci/staggered100` | 7000 | PASS | 100 | 0.9826 | 0.0750 | 0.2397 | -0.1963 | 0.0740 | FAIL |
| `fanin-equalized/g-and-d/grid100` | 7000 | FAIL | 100 | 0.9863 | 0.0946 | 0.1679 | 0.0004 | 0.0079 | FAIL |
| `fanin-equalized/g-and-d-beta999-prior3-end02/grid100` | 7000 | FAIL | 98 | 0.9832 | 0.1264 | 0.1401 | 0.0048 | 0.0104 | FAIL |
| `fanin-equalized/g-only/grid100` | 7000 | FAIL | 92 | 0.9861 | 0.1637 | 0.1895 | 0.0014 | 0.0092 | FAIL |
| `root-search/refine/fourier3_batch2048_8k/grid100` | 8000 | PASS | 100 | 0.9885 | 0.0666 | 0.1368 | -0.0442 | 0.0169 | FAIL |
| `root-search/refine/fourier3_batch2048_floor002/grid100` | 7000 | PASS | 100 | 0.9885 | 0.0651 | 0.1428 | -0.0400 | 0.0144 | FAIL |
| `root-search/screen/cap2/grid100` | 7000 | FAIL | 100 | 0.9880 | 0.0628 | 0.1371 | 0.0136 | 0.0117 | FAIL |
| `root-search/screen/fourier3/grid100` | 7000 | FAIL | 100 | 0.9860 | 0.0657 | 0.1448 | 0.0122 | 0.0107 | FAIL |
| `root-search/screen/fourier4/grid100` | 7000 | FAIL | 100 | 0.9844 | 0.0670 | 0.1571 | 0.0034 | 0.0089 | FAIL |
| `root-search/screen/longer_input_noise/grid100` | 7000 | FAIL | 100 | 0.9877 | 0.0605 | 0.1432 | 0.0097 | 0.0109 | FAIL |
| `search-agent/batch1024/noise029_batch1024_floor005/grid100` | 7000 | FAIL | 100 | 0.9881 | 0.0612 | 0.1298 | 0.0121 | 0.0113 | FAIL |
| `search-agent/batch2048/noise029_batch2048_floor005/grid100` | 7000 | PASS | 100 | 0.9896 | 0.0525 | 0.1333 | -0.0159 | 0.0052 | PASS |
| `search-agent/cap2-batch2048/noise029_batch2048_cap2_floor002/grid100` | 7000 | PASS | 100 | 0.9907 | 0.0654 | 0.1245 | -0.0390 | 0.0141 | FAIL |
| `search-agent/cap2-batch512/noise029_batch512_cap2_floor002/grid100` | 7000 | FAIL | 100 | 0.9885 | 0.0544 | 0.1111 | 0.0009 | 0.0063 | PASS |
| `search-agent/grid-screen/noise029_anneal030/grid100` | 7000 | FAIL | 100 | 0.9845 | 0.0737 | 0.1515 | 0.0428 | 0.0236 | FAIL |
| `search-agent/grid-screen/noise029_batch1024/grid100` | 0 | INCOMPLETE | — | — | — | — | — | — | running |
| `search-agent/grid-screen/noise029_batch512/grid100` | 7000 | FAIL | 100 | 0.9840 | 0.0720 | 0.1541 | 0.0464 | 0.0252 | FAIL |
| `search-agent/grid-screen/noise029_floor002/grid100` | 7000 | FAIL | 100 | 0.9870 | 0.0681 | 0.1298 | -0.0002 | 0.0071 | FAIL |
| `search-agent/grid-screen/noise030_batch512/grid100` | 7000 | FAIL | 100 | 0.9799 | 0.0723 | 0.1616 | 0.0999 | 0.0489 | FAIL |
| `search-agent/residual-architecture/alpha001/grid100` | 7000 | FAIL | 68 | 0.8573 | 0.2555 | — | — | — | FAIL |
| `search-agent/residual-architecture/alpha010/grid100` | 7000 | FAIL | 88 | 0.9714 | 0.1647 | 0.2520 | -0.0064 | 0.0070 | FAIL |
| `search-agent/residual-architecture/alpha030_noiseend010/grid100` | 7000 | FAIL | 80 | 0.9413 | 0.1763 | — | — | — | FAIL |
| `search-agent/residual-architecture/alpha030_noiseend010_anneal040/grid100` | 7000 | FAIL | 84 | 0.9766 | 0.1833 | — | — | — | FAIL |
| `search-agent/residual-architecture/alpha030_noiseend010_cap2/grid100` | 7000 | FAIL | 94 | 0.9794 | 0.1116 | 0.1491 | -0.0061 | 0.0061 | FAIL |
| `search-agent/residual-architecture/alpha050_noiseend010/grid100` | 7000 | FAIL | 88 | 0.9585 | 0.1683 | — | — | — | FAIL |
| `search-agent/warmup-screen/warm_b2048_f2_beta0999/warm_b2048_f2_beta0999/grid100` | 7000 | FAIL | 54 | 0.8909 | 0.3824 | 0.7265 | -0.0509 | 0.0810 | FAIL |
| `search-agent/warmup-screen/warm_b2048_f2_beta0999_anneal060/warm_b2048_f2_beta0999_anneal060/grid100` | 7000 | FAIL | 59 | 0.9194 | 0.3687 | 0.6321 | -0.0416 | 0.0506 | FAIL |
| `search-agent/warmup-screen/warm_b2048_f2_g128_d3/warm_b2048_f2_g128_d3/grid100` | 7000 | FAIL | 85 | 0.9827 | 0.2010 | 0.2729 | 0.0063 | 0.0174 | FAIL |
| `shared-v3/screen/v3_noise/grid100` | 7000 | FAIL | 80 | 0.8780 | 0.2147 | — | — | — | FAIL |
| `shared-v3/screen/v3_noise_anneal04/grid100` | 7000 | FAIL | 84 | 0.9631 | 0.1774 | 0.4298 | 0.0916 | 0.0674 | FAIL |
| `shared-v3/screen/v3_noise_cap2/grid100` | 7000 | FAIL | 93 | 0.9752 | 0.1531 | 0.2520 | 0.0363 | 0.0292 | FAIL |
| `shared-v3/screen/v3_noise_floor005/grid100` | 7000 | FAIL | 69 | 0.9453 | 0.2808 | — | — | — | FAIL |
| `shared-v3/small-g/v3_small_g16_f3_end01/grid100` | 7000 | FAIL | 83 | 0.9790 | 0.2141 | 0.1881 | 0.0285 | 0.0221 | FAIL |
| `shared-v3/small-g/v3_small_g32_f3_end01/grid100` | 7000 | FAIL | 82 | 0.9680 | 0.2246 | 0.2232 | 0.0395 | 0.0246 | FAIL |
| `shared-v3/small-g/v3_small_g64_f3_end01/grid100` | 7000 | FAIL | 94 | 0.9755 | 0.1456 | 0.1922 | 0.0432 | 0.0207 | FAIL |

# Source-bound secant counters

Every row below comes from a certified final checkpoint verified by file SHA, byte count and recursive state SHA. [Machine-readable proof](secant-counters.json) includes checkpoint identities, each parameter shape, nominal rate, last effective fraction, owned-row counts and exact visit clocks. Baseline checkpoints contain no secant state. The incomplete word attempt has no certified final counter packet; its counters remain unknown.

Fractions average parameter blocks or owned prior rows, so row observations must not be combined with network blocks as equal tasks. Motion ratios divide sums of Euclidean block lengths; they are not a joint parameter norm. Gaussian stability includes its restored 1,000-update producer prefix. All fourteen complete candidate checkpoints had zero zero-proposal counts.

| Task | Role | Observations | Mean fraction | Floor hits | Damped | Applied/proposed lengths |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rotated100 | generator | 56000 | 0.077603 | 54698/56000 (97.67%) | 98.64% | 0.079047 |
| rotated100 | prior | 13625515 | 0.082895 | 9930281/13625515 (72.88%) | 99.53% | 0.080362 |
| rotated100 | critic | 56000 | 0.074524 | 53879/56000 (96.21%) | 99.22% | 0.073750 |
| vector_unequal_width | generator | 7200 | 0.094642 | 6400/7200 (88.89%) | 97.97% | 0.093448 |
| vector_unequal_width | prior | 121074 | 0.138534 | 68075/121074 (56.23%) | 97.58% | 0.154351 |
| vector_unequal_width | critic | 7200 | 0.089105 | 6266/7200 (87.03%) | 98.40% | 0.084770 |
| gaussian1d_smoke | generator | 6000 | 0.237188 | 3310/6000 (55.17%) | 87.00% | 0.211064 |
| gaussian1d_smoke | prior | 100934 | 0.128389 | 55174/100934 (54.66%) | 98.05% | 0.139515 |
| gaussian1d_smoke | critic | 6000 | 0.214866 | 3728/6000 (62.13%) | 89.72% | 0.180533 |
| vector_unequal_mass | generator | 7200 | 0.110227 | 6002/7200 (83.36%) | 96.06% | 0.100540 |
| vector_unequal_mass | prior | 121074 | 0.181925 | 54799/121074 (45.26%) | 95.66% | 0.192313 |
| vector_unequal_mass | critic | 7200 | 0.103894 | 6247/7200 (86.76%) | 97.11% | 0.095838 |
| residual_student | generator | 2400 | 0.213996 | 1281/2400 (53.37%) | 91.12% | 0.251681 |
| residual_student | prior | 4800 | 0.320458 | 1964/4800 (40.92%) | 80.31% | 0.347354 |
| residual_student | critic | 2400 | 0.169728 | 1657/2400 (69.04%) | 93.79% | 0.157557 |
| ring16_acquisition | generator | 9600 | 0.094196 | 8428/9600 (87.79%) | 98.33% | 0.086235 |
| ring16_acquisition | prior | 161390 | 0.073568 | 131592/161390 (81.54%) | 99.71% | 0.071447 |
| ring16_acquisition | critic | 9600 | 0.094227 | 8185/9600 (85.26%) | 98.53% | 0.080746 |
| vector_anisotropic | generator | 7200 | 0.149206 | 4809/7200 (66.79%) | 94.58% | 0.124648 |
| vector_anisotropic | prior | 121074 | 0.185712 | 52492/121074 (43.36%) | 95.54% | 0.201463 |
| vector_anisotropic | critic | 7200 | 0.123995 | 4822/7200 (66.97%) | 97.08% | 0.116627 |
| staggered100 | generator | 56000 | 0.074568 | 54943/56000 (98.11%) | 98.93% | 0.074615 |
| staggered100 | prior | 13625515 | 0.083529 | 9967358/13625515 (73.15%) | 99.44% | 0.082428 |
| staggered100 | critic | 56000 | 0.073212 | 53994/56000 (96.42%) | 99.31% | 0.072352 |
| grid100 | generator | 56000 | 0.072962 | 54951/56000 (98.13%) | 99.04% | 0.074080 |
| grid100 | prior | 13625515 | 0.082396 | 10262300/13625515 (75.32%) | 99.38% | 0.082014 |
| grid100 | critic | 56000 | 0.076142 | 53733/56000 (95.95%) | 98.98% | 0.072304 |
| unused_token_hold | generator | 400 | 0.830160 | 17/400 (4.25%) | 28.25% | 0.849595 |
| unused_token_hold | critic | 800 | 0.542724 | 111/800 (13.88%) | 63.00% | 0.519948 |
| trajectory | generator | 2400 | 0.205447 | 1431/2400 (59.62%) | 89.29% | 0.234349 |
| trajectory | prior | 4800 | 0.333208 | 2264/4800 (47.17%) | 78.17% | 0.369941 |
| trajectory | critic | 2400 | 0.177547 | 1666/2400 (69.42%) | 92.92% | 0.183636 |
| two_pole | generator | 80 | 1.000000 | 0/80 (0.00%) | 0.00% | 1.000000 |
| two_pole | critic | 320 | 0.803118 | 13/320 (4.06%) | 33.75% | 0.853722 |
| gaussian1d_stability | generator | 36000 | 0.258880 | 15926/36000 (44.24%) | 88.34% | 0.219362 |
| gaussian1d_stability | prior | 605154 | 0.103012 | 362350/605154 (59.88%) | 99.22% | 0.110625 |
| gaussian1d_stability | critic | 36000 | 0.222530 | 17572/36000 (48.81%) | 92.31% | 0.198125 |
| ae_gan_hold | generator | 2000 | 0.159822 | 1438/2000 (71.90%) | 92.45% | 0.162894 |
| ae_gan_hold | prior | 2983 | 0.252103 | 872/2983 (29.23%) | 90.31% | 0.252383 |
| ae_gan_hold | critic | 1000 | 0.178074 | 600/1000 (60.00%) | 93.40% | 0.167229 |

At ring completion, all six generator blocks are at fraction .0625; five critic blocks are .0625 and the sixth .0883883. Its 256 owned prior rows have clocks 576–684 and last fractions .0625–.211550. Grid completion has all eight generator and eight critic blocks at .0625; all 20,000 prior rows have clocks 583–790 and fractions .0625–.651527. Gaussian continuation reaches network clocks 6,000 and prior clocks 2,252–2,475, retaining different last fractions across blocks. The saved packet preserves both clock dtypes and validity masks. Exact sparse untouched-row replay and zero/no-change behavior are established by the already completed software checks, not inferred solely from cumulative counters.

This shows a live, heterogeneous intervention, but ring and native network motion is dominated by the fixed 1/16 floor. Final aggregates cannot establish when damping preceded quality loss, identify the curvature versus growth restriction, or separate minibatch/opponent changes from local curvature.

# Shared network-floor bracket

Three globally shared network learning-rate floors were declared before
training: 0.002, 0.01, and 0.02. Every other field in
`accuracy_network_floor.json` was identical, including κ=1.0, prior LR floor
0.05, the 1600-step network horizon, β₂=0.999, output noise 0.029 with 20%
warmup, and input noise 0.5 ending at 10% of the host budget. The frozen
manifest `artifacts/toy100-accuracy/compatibility/network-floor-bracket-v1/manifest.json`
has SHA-256 `4e94048198d8125edc6a067f8fb545153ecc5aadf013654e304c08d51c1d0ee0`.
All transfer runs used source commit `a3be165`, seed 0, one CPU thread,
original host budgets and thresholds, and the same nine canonical bottlenecks.

| Network floor | Strict nine-host result | Overlap suffix | Stripes suffix | Bars suffix |
| ---: | ---: | ---: | ---: | ---: |
| 0.002 | 7/9; overlap and bars fail | 1 | 8 | 2 |
| 0.010 | **9/9** | 7 | 8 | 7 |
| 0.020 | 8/9; overlap fails | 4 | 8 | 6 |

At 0.002, overlap passes at step 1000 but its SW1 rises from 0.061 there to
0.156 at step 1100; it finishes at 0.116 without a sustained terminal run.
Bars first has all four modes at step 575, leaving only two passing checks.
At 0.010, overlap, stripes, and bars all sustain at least five terminal
checks; bars finishes with four modes and HQ 0.96875. At 0.020, bars remains
stable, while overlap finishes with covariance error 0.408 and SW1 0.138,
breaking its terminal streak. Stripes passes all three rows with suffix 8.

The predeclared rule promoted the first 9/9 row, 0.010, to all 19 frozen
transfer hosts while the final bracket row continued. Strict regrading gives
**18/19**, with only `residual_student` failing. Its 400-step final identity
MSE is 0.04915 against 0.02, success rate is 0.5833 against 1.0, and
wrong-pad rate is 0.4167 against 0.0; it has no passing observations. Thus
the nine-host screen was insufficient, and subsequent bottleneck screens
must include `residual_student`. This candidate does not establish a
shared 22-toy pass.

Four further matched, predeclared `residual_student` controls isolated the
cause under the same source, seed, host budget, noise, and gates. The
control manifest SHA-256 is
`f985d02054afbe3670711bfc799a3a43ed1bbfa64d49c3d4478fdd1eeb7b8897`.

| Global κ | Network floor | Residual-student gate | Final identity MSE | Wrong-pad rate |
| ---: | ---: | --- | ---: | ---: |
| 1.25 | recipe floor 0.05 | PASS; suffix 9 | 0.000934 | 0 |
| 1.00 | recipe floor 0.05 | FAIL; suffix 0 | 0.04883 | 0.4167 |
| 1.00 | 0.005 | FAIL; suffix 0 | 0.04919 | 0.4167 |
| 1.00 | 0.020 | FAIL; suffix 0 | 0.04904 | 0.4167 |

The κ change is the observed cause of the wrong-pad regression in these
matched runs; network floor values from 0.005 through 0.020 do not repair it.
The 0.010 full19 episode is the fifth reference, not a rerun.

Raw local evidence, including all compressed episodes, source snapshots,
configs, action/noise receipts, and strict regrades, is retained at
`artifacts/toy100-accuracy/compatibility/network-floor-bracket-v1/` and
`artifacts/toy100-accuracy/compatibility/network-floor-residual-controls-v1/`.
Their RAM originals remain under `/dev/shm/particlegan-toy-floor-bracket-v1-a3be165/`
and `/dev/shm/particlegan-toy-floor-residual-controls-v1-a3be165/`.
After copying, all 85 bracket and 44 control files matched SHA-256, and
isolated-source regrading returned the same outcomes from the relocated
evidence. The respective file-hash manifest digests are
`251c8370c0b0eaaee589b371da2ae1eb08193ad6b849166c5aae649c89e8b0fa`
and `1d0f0a0b8ccae2d78754247c8e3d3c952079f65bfe4462eadc476c32bbfec7b8`.
The `artifacts/` paths are local workspace evidence, not GitHub links.

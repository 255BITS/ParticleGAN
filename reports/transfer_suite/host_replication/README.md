# Second-host replication of the b_cap3 19/19 claim

**Not replicated: 8/9 required + 8/10 practical = 16/19 on a second CPU host.** The archived 19/19 evidence rebuilds and verifies unchanged, and it replays exactly on its original host. This replay keeps identical source hashes, torch 2.13.0, seed 0, specs, policies, architectures and thresholds. Every one of the 46 replayed episodes already differs from its archive at its first measurement (the rare-mode winner by 3.7e-6 at step 50); differences then amplify. 39/46 archived passing trials still pass. Cases without a surviving supporting architecture: **mode_hold, vector_unequal_mass, img_bars4**.

This is verification of the Codex result, not a new formulation or leaderboard entry. The formulation leaderboard still counts the archived live evidence. This host is a robustness audit: only archived passing trials were replayed, so archived failures cannot be promoted by a lucky host.

| Case | Archived | This host | Archived passing architectures still passing | Replayed final passing checks |
| --- | --- | --- | ---: | --- |
| two_pole | PASS | PASS | 1/1 | original required host 7 |
| trajectory | PASS | PASS | 1/1 | original required host 8 |
| residual_student | PASS | PASS | 1/1 | original required host 19 |
| unipolar | PASS | PASS | 1/1 | original required host 18 |
| ae_gan_hold | PASS | PASS | 1/1 | original required host 19 |
| cover_leftover | PASS | PASS | 1/1 | original required host 14 |
| unused_token_hold | PASS | PASS | 1/1 | original required host 14 |
| mid_scale_identity | PASS | PASS | 1/1 | original required host 17 |
| mode_hold | PASS | **FAIL** | 0/1 | original required host 0 |
| vector_two_broad | PASS | PASS | 9/9 | original architecture 19, d128_l3_f3 21, d128_l3_f4 22, d256_l3_f3 16, axis_softplus5 17, axis_tanh 17, softplus10_d64_l2_f2 16, softplus5_d128_l3_f2 24, linear_skip_d96_beta5 18 |
| vector_unequal_mass | PASS | **FAIL** | 0/1 | linear_skip_d96_beta5 0 |
| vector_unequal_width | PASS | PASS | 1/2 | axis_softplus5 4, softplus10_d64_l2_f2 5 |
| vector_anisotropic | PASS | PASS | 3/3 | original architecture 16, d128_l3_f3 16, d128_l3_f4 7 |
| vector_overlap | PASS | PASS | 4/6 | d128_l3_f3 0, d128_l3_f4 8, d256_l3_f3 0, d64_l2_f5 14, axis_tanh 5, oriented8_tanh 5 |
| vector_spiral | PASS | PASS | 9/9 | original architecture 23, d128_l3_f3 24, d128_l3_f4 22, d256_l3_f3 20, axis_softplus5 22, axis_tanh 16, softplus10_d64_l2_f2 22, softplus5_d128_l3_f2 23, linear_skip_d96_beta5 23 |
| img_stripes2 | PASS | PASS | 2/2 | residual16 20, residual12 7 |
| img_bars4 | PASS | **FAIL** | 0/1 | residual16 4 |
| img_blobs4 | PASS | PASS | 1/2 | baseline 0, residual16 12 |
| img_intensity2 | PASS | PASS | 2/2 | residual16 6, residual12 6 |

Required tests list the original host. Passing needs every live metric at each of the final five of 24 checks; EMA is not used.

## Archived passing trials that fail here

| Case / architecture | Archived final passing checks | Replayed final passing checks | Replayed final failing metrics |
| --- | ---: | ---: | --- |
| mode_hold / original required host | 8 | 0 | modes, hq |
| vector_unequal_mass / linear_skip_d96_beta5 | 6 | 0 | component_covariance_error, component_min_eigen_ratio |
| vector_unequal_width / axis_softplus5 | 7 | 4 | All final bounds pass; too few final passing checks |
| vector_overlap / d128_l3_f3 | 5 | 0 | mean_error, covariance_error |
| vector_overlap / d256_l3_f3 | 7 | 0 | mean_error |
| img_bars4 / residual16 | 7 | 4 | All final bounds pass; too few final passing checks |
| img_blobs4 / baseline | 8 | 0 | modes, hq |

The required eight-mode ring collapses to 5/8 modes with 74.6% good samples. The rare-mode winner ends at component covariance error 2.624 (≤.85) and minimum spread .092 (≥.15). img_bars4 passes its final four checks but needs five; residual16 is its only archived supporting architecture.

## Rare-mode winner: documented command

`python -u -m benchmarks.transfer_suite.run_linear_skip --tasks vector_unequal_mass`, unchanged, under two torch 2.13.0 builds and three CPU instruction pins:

| Torch build | Pinned instruction set | Exact vs archive | First divergence | Live result | Final passing checks | Final covariance error | Final spread |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| CPU wheel | none | No | step 50, 3.7e-06 | FAIL | 0 | 2.624 | 0.092 |
| CPU wheel | ATen AVX2 | No | step 50, 3.7e-06 | FAIL | 0 | 2.624 | 0.092 |
| CPU wheel | ATen+MKL+oneDNN AVX2 | No | step 50, 1.9e-06 | FAIL | 0 | 0.307 | 0.030 |
| cu126 wheel (archive build) | none | No | step 50, 3.7e-06 | FAIL | 0 | 2.624 | 0.092 |
| cu126 wheel (archive build) | ATen AVX2 | No | step 50, 3.7e-06 | FAIL | 0 | 2.624 | 0.092 |
| cu126 wheel (archive build) | ATen+MKL+oneDNN AVX2 | No | step 50, 1.9e-06 | FAIL | 0 | 0.307 | 0.030 |

The two torch builds give bit-identical results on this host, so the archive/host difference is the CPU numerical path, not the wheel. Pinning MKL/oneDNN to AVX2 changes the trajectory but does not recover the archived one. The replication driver reproduces the official CLI bit-for-bit on this host, and repeated runs are deterministic. The CLI plan declared the three CPU-wheel runs; the cu126 runs and the six-toy run were added after those failed, to isolate the torch build and record the documented full profile.

The winner's full six-data CLI profile here is also 3/6, with a different set: two_broad PASS (18), unequal_mass FAIL (0), unequal_width FAIL (0), anisotropic PASS (17), overlap FAIL (2), spiral PASS (23). Archived: rare mass, broad and spiral pass; anisotropic fails.

## Interpretation

These fixed-seed results are sensitive to floating-point rounding. Cases with several passing architectures or long passing streaks replicate (broad, spiral, anisotropic, stripes, intensity; blobs via residual16; overlap via d128_l3_f4, d64_l2_f5 and both tanh critics). Cases resting on one selected architecture do not reliably replicate: the rare 2% mode (1 winner of 58 candidates) and bars4 (residual16 only). The required eight-mode ring also collapses here. A 19/19 claim should cite host identity alongside seed 0; a host-robust claim would need passes on more than one CPU host. No threshold, target, seed or scoring rule changed.

Host: torch 2.13.0+cpu, CPU capability AVX512, Linux-6.12.94+-x86_64-with-glibc2.39, one thread per episode, 4 concurrent episodes. Archived host: torch 2.13.0+cu126, AVX2.

## Reproduction

```bash
python -u -m reports.transfer_suite.host_replication.run --output /tmp/host-replication > /tmp/host-replication.log 2>&1
grep -E "^(DONE|COMPLETE)" /tmp/host-replication.log   # or tail -f
python -m reports.transfer_suite.host_replication.build   # rebuild from archived episodes, no training
```

[Machine-readable results](leaderboard.json) · [Replay log](run.log) · [Declared jobs](plan.json.gz) · [Protocol and source hashes](protocol.json.gz) · [CLI replay plan](cli/plan.json.gz) · [Archive hashes](archive_manifest.json) · [Tests](tests.log).

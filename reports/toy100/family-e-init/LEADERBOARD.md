# Family E CPU screen

Init-only search. Learning rate, schedules, loss coefficients, clips, and every other K3P recipe knob are unchanged. Each row is a deterministic orthogonal construction, a bias scheme, and the Roberts R2 particle prior.

## Build

| field | value |
|---|---|
| torch | `2.14.0+cu130` |
| torch CUDA build | `13.0` |
| torch git | `08187d9e0fba026dc8217405802ab5381dc88d90` |
| `torch.cuda.is_available()` | false |

This is the same cu130-on-CPU situation as the earlier cloud C screen. Absolute pass rates disagree with A6000 history. The calibration row below is the comparison point for this build.

Screen log (local, not committed): `/tmp/family-e-init/screen.log`.

## What was screened

33 variants: 11 families × biases `bz` (zero), `pb` (golden pattern at the declared RMS), `wq` (Weyl at 1/4 of the default bound). Every variant uses the R2 prior (`_pq`). Per-layer scale is the declared Kaiming element RMS (`std`).

The Frobenius match (`*_frob`) is the same bits on these Linear hosts. The mode_hold init hash of `giv_bz_pq` and `giv_bz_pq_frob` is `ec31f0892adc8467…`. Frobenius was not screened a second time.

| prefix | construction |
|---|---|
| `giv` | product of Givens rotations at golden-angle steps |
| `giw` | same product at Weyl (`sqrt(2)`) angles |
| `cay` | Cayley transform of a deterministic skew matrix |
| `rft` | real Fourier orthogonal |
| `cir` | real circulant orthogonal (unimodular Thue-Morse spectrum) |
| `haar` | Haar butterfly (`pi/4` pairings) |
| `walsh` | sequency-ordered Walsh |
| `sob` | QR of a Sobol matrix through the inverse normal |
| `lat` | QR of a rank-1 lattice matrix |
| `but` | FFT-style butterfly of golden-angle Givens |
| `exp` | scaling-and-squaring exponential of a tridiagonal skew matrix |

Patches: `reports/toy100/family-e-init/variants/<name>.patch` against `master`. `--init <name>` selects the construction. `--init default` installs that patch's `PATCH_DEFAULT`.

## Determinism

Init-only mode_hold, two processes at offset 0 and one at offset 101: **33/33** identical `all_params_sha256` and identical `initial-values.pt` tensor payloads (`/tmp/family-e-init/determinism.json`).

The two full repo-seed ring trainings (`s0-mode_hold` and `prio-probe-mode_hold`) also match on parameter tensors for both finalists:

| variant | `initial-values.pt` | `final-state.pt` |
|---|---|---|
| `cay_pb_pq` | `507402fc963a5084` | `b0c59491759633fe` |
| `rft_bz_pq` | `4b567f833f6f9064` | `9a83f817e775e42f` |

`result.json` still differs in wall-clock fields and in the `.pt` container hash (zip metadata). The tensor payloads match.

## Calibration and ranked screen

Offsets `0, 101, 202, 303, 404, 505, 606, 707`. Each cell pair is ring (`mode_hold`) then `vector_unequal_mass`. `P8` is a ring pass with 8 modes. `F8` is 8 modes with a failing suffix. Rank is ring passes, then unequal passes. All 528 training jobs exited 0.

| variant | ring | unequal | cells |
|---|---:|---:|---|
| `cay_pb_pq` | 7/8 | 6/8 | P8 P P8 P P8 F P8 P F0 F P8 P P8 P P8 P |
| `rft_bz_pq` | 6/8 | 6/8 | P8 P P8 F P8 F P8 P P8 P F6 P P8 P F7 P |
| `cay_bz_pq` | 6/8 | 3/8 | P8 P P8 P P8 F P8 F F7 F F6 F P8 F P8 P |
| `giw_bz_pq` | 6/8 | 2/8 | P8 F P8 F P8 P P8 F P8 P P8 F F6 F F6 F |
| `giw_pb_pq` | 6/8 | 2/8 | P8 F F7 F P8 F P8 F P8 P F2 P P8 F P8 F |
| `haar_bz_pq` | 5/8 | 6/8 | F4 P P8 P P8 P P8 P P8 P F8 F F3 F P8 P |
| `exp_wq_pq` | 5/8 | 4/8 | P8 F P8 P P8 F F7 P P8 F F0 P F7 P P8 F |
| `cir_bz_pq` | 5/8 | 3/8 | F7 F P8 F F1 F P8 P P8 P F8 P P8 F P8 F |
| `lat_pb_pq` | 5/8 | 3/8 | F8 F P8 F F7 P P8 P P8 F F7 F P8 F P8 P |
| `lat_wq_pq` | 5/8 | 3/8 | F7 F F7 F P8 P P8 P P8 P F8 F P8 F P8 F |
| `walsh_bz_pq` | 5/8 | 3/8 | P8 F P8 P F8 P P8 F F7 P P8 F P8 F F8 F |
| `k3p` | 5/8 | 2/8 | P8 P F7 F F5 F P8 F F8 F P8 F P8 F P8 P |
| `lat_bz_pq` | 5/8 | 2/8 | P8 F P8 P F8 F F5 F P8 F F7 F P8 P P8 F |
| `giw_wq_pq` | 5/8 | 1/8 | P8 F F7 F F8 F P8 F P8 P P8 F F3 F P8 F |
| `haar_pb_pq` | 4/8 | 5/8 | P8 P F2 F P8 P P8 P F0 P F5 P P8 F F8 F |
| `giv_wq_pq` | 4/8 | 4/8 | F8 P F5 F P8 P P8 F P8 P P8 P F7 F F0 F |
| `haar_wq_pq` | 4/8 | 4/8 | F7 F F6 F P8 F P8 P F6 P P8 F P8 P F7 P |
| `walsh_pb_pq` | 4/8 | 4/8 | F8 F P8 P F0 P P8 P P8 F F8 P P8 F F8 F |
| `cir_wq_pq` | 4/8 | 3/8 | F8 F P8 F P8 F F6 P F8 F F7 F P8 P P8 P |
| `sob_bz_pq` | 4/8 | 3/8 | P8 F P8 P P8 F F7 P F7 F F7 F P8 F F8 P |
| `rft_pb_pq` | 4/8 | 2/8 | P8 P P8 F F7 F P8 F P8 F F6 F F7 F F7 P |
| `cir_pb_pq` | 4/8 | 1/8 | P8 F F6 F F8 P F1 F P8 F P8 F P8 F F0 F |
| `giv_pb_pq` | 4/8 | 1/8 | P8 F F8 F P8 P F8 F F2 F F4 F P8 F P8 F |
| `sob_pb_pq` | 3/8 | 6/8 | P8 P F7 P F8 F P8 P F8 P F8 P P8 F F6 P |
| `walsh_wq_pq` | 3/8 | 5/8 | F8 P F7 F P8 F P8 P F7 F F7 P P8 P F8 P |
| `exp_pb_pq` | 3/8 | 4/8 | P8 P F7 P F8 F F1 P P8 F F6 F P8 F F1 P |
| `rft_wq_pq` | 3/8 | 4/8 | F7 F F8 P P8 P P8 P F7 P F8 F F7 F P8 F |
| `giv_bz_pq` | 3/8 | 3/8 | P8 F P8 P F8 F F1 P P8 P F5 F F1 F F5 F |
| `cay_wq_pq` | 3/8 | 2/8 | P8 F F7 F F7 P P8 F F8 F F8 F P8 F F7 P |
| `exp_bz_pq` | 3/8 | 2/8 | F2 F F5 F P8 P F7 F P8 F P8 F F0 P F7 F |
| `sob_wq_pq` | 3/8 | 1/8 | F8 F F7 F P8 F P8 F F7 P P8 F F7 F F1 F |
| `but_wq_pq` | 3/8 | 0/8 | F2 F P8 F F8 F F0 F P8 F P8 F F7 F F6 F |
| `but_pb_pq` | 2/8 | 0/8 | F7 F P8 F F6 F F8 F F7 F F7 F P8 F F6 F |
| `but_bz_pq` | 1/8 | 1/8 | P8 F F7 F F7 P F8 F F0 F F0 F F2 F F0 F |

`k3p` is the random-init calibration on this build: ring 5/8 (offsets 0, 303, 505, 606, 707) and unequal 2/8 (offsets 0, 707). Offset 404 is `F8` (8 modes, suffix fail). Repo-seed ring: PASS, 8 modes, hq 0.999755859375, suffix 7, 15.8s.

## Priority gates at the repo seed

CPU-runnable gates only. `native100.py` raises `CUDA with CUBLAS_WORKSPACE_CONFIG=:4096:8 required` before training, so `grid100` and `rotated100` were not run.

| gate | `cay_pb_pq` | `rft_bz_pq` |
|---|---|---|
| toy `mode_hold` | PASS, 8 modes, hq 0.9998, suffix 7 | PASS, 8 modes, hq 1.0, suffix 7 |
| hold `mode_hold` | PASS, converged at 1400, 1200 hold checks, last live step 2900 modes 8 hq 0.9995 | POST_CONVERGENCE_FAIL, converged at 1400, first failure at 1934 (534 checks) |
| toy `vector_unequal_mass` | PASS, mmr 0.981, eigen 0.696, cov 0.220, suffix 13 | PASS, mmr 0.891, eigen 0.332, cov 0.352, suffix 11 |
| toy `img_stripes2` | PASS, 2 modes, hq 1.0, suffix 23 | PASS, 2 modes, hq 0.9375, suffix 16 |
| toy `img_blobs4` | FAIL, 4 modes, hq 0.875 | FAIL, 3 modes, hq 0.84375 |
| native `grid100` | not run (CUDA required) | not run (CUDA required) |
| native `rotated100` | not run (CUDA required) | not run (CUDA required) |
| shift `mode_hold` | FAIL. Pre-shift hold 120/120, min hq 0.913. Recovery 0/120, modes 0 | FAIL. Pre-shift 102/120, min hq 0.818. Recovery 0/120, modes 0 |

CPU score on the six gates that ran: `cay_pb_pq` 4/6, `rft_bz_pq` 3/6.

## Reading

`cay_pb_pq` (Cayley + pattern bias + R2) is the only construction that beats this build's K3P calibration on both early gates at once, 7/8 ring and 6/8 unequal. Its single ring miss is offset 404, a hard collapse to 0 modes, and unequal also fails on that offset. Unequal also fails at offset 202.

`rft_bz_pq` (real Fourier + zero bias + R2) ties the best unequal rate and is one ring pass behind. Ring misses are near-misses: offset 505 has 6 modes, offset 707 has 7. Unequal misses are offsets 101 and 202. The long hold converges and then breaks.

Bias is family-specific. Cayley wants the pattern bias (`pb` 7/8 and 6/8, `bz` 6/8 and 3/8, `wq` 3/8 and 2/8). Real Fourier wants zero bias (`bz` 6/8 and 6/8, `pb` 4/8 and 2/8, `wq` 3/8 and 4/8). Weyl-Givens (`giw_bz_pq`) is strong on ring (6/8) and stays at the K3P unequal rate (2/8). Haar with zero bias matches the calibration ring rate and reaches 6/8 unequal, with hard ring misses (4, unstable-8, and 3 modes). Butterfly orthogonal is the weak family: best ring 3/8, unequal 0/8 or 1/8.

## A6000 recommendation

Confirm `cay_pb_pq` first: determinism, the eight repo-seed priority gates including native `grid100` and `rotated100`, the 22-suite, then the eight sample-seed offsets. Watch offset 404. Run `rft_bz_pq` second. CPU passes are a screen against this build's calibration. They are not the qualification.

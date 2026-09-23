# Why the archived 19/19 does not replay on other CPUs

**The archived rare-mode PASS is one outcome of a chaotic trajectory, pinned to its original CPU numerical path.** Two MKL kernels round differently on different CPUs. Seed-0 GAN training amplifies those one-ULP differences until the late checks are effectively independent draws. No fresh run on any tested numerical path passes the rare toy, so 19/19 is not host-portable.

## 1. Where the first difference appears

Native default path vs MKL CNR AVX2 on the same machine, same process-level seed and code. During the first forward passes only the narrow output layers differ:

| Tensor at update 1 | Shape | Differing elements | Max abs difference |
| --- | --- | ---: | ---: |
| g.net.0 | [128, 64] | 0 | 0.00e+00 |
| g.net.2 | [128, 64] | 0 | 0.00e+00 |
| g.net.4 | [128, 2] | 142 | 5.96e-08 |
| d.main.net.0 | [128, 96] | 0 | 0.00e+00 |
| d.main.net.2 | [128, 96] | 0 | 0.00e+00 |
| d.main.net.4 | [128, 1] | 70 | 8.94e-08 |
| d_grad_before_first_update | [10467] | 4468 | 1.34e-07 |
| d_after_first_update | [10467] | 209 | 1.78e-06 |

Wide hidden layers match bit-for-bit. G 64→2 and D 96→1 differ by about one float32 ULP, so MKL chooses CPU-specific kernels for these narrow matrix products. Under MKL CNR COMPATIBLE, native AVX512 and emulated Haswell agree on every forward output and on all 10,467 D gradients. The first Adam step still changes 104 D parameters by 1.5e-08.

That second source is `torch.sqrt` in the Adam denominator. Its float32 results depend on MKL settings: MKL AVX2 instructions make every sqrt correctly rounded, while the default AVX512 path misses 73. An initial isolated check found identical lerp, addcmul and addcdiv results across these CPUs, while sqrt differed:

| Numerical path | Step-50 fingerprint | Matches archive | sqrt results not correctly rounded / 10,467 | Adam step digest |
| --- | --- | --- | ---: | --- |
| native default | `a6d2934c35a8e183` | no | 73 | `73c1cdf7f5db` |
| native ATen AVX2 | `a6d2934c35a8e183` | no | 73 | `73c1cdf7f5db` |
| native ATen scalar | `c2ba7fd042b8fba6` | no | 80 | `b1fb8bfd26c8` |
| native MKL instructions AVX2 | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| native MKL CNR AVX2 | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| native MKL CNR AVX | `f2f40c3f35b37ef6` | no | 1735 | `bcbe822f24eb` |
| native MKL CNR COMPATIBLE | `6f99c137c7a399d1` | no | 1735 | `bcbe822f24eb` |
| native all AVX2 pins | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| emulated Haswell-v4 | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| emulated Skylake-Client-v4 | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| emulated EPYC-Rome-v4 | `ded0a0f1b26004a2` | no | 1581 | `f6c51898bf02` |
| emulated EPYC-Milan-v2 | `ded0a0f1b26004a2` | no | 1581 | `f6c51898bf02` |
| emulated Haswell-v4 MKL CNR AVX2 | `2a5fafcb6626038c` | no | 0 | `73c1cdf7f5db` |
| emulated Haswell-v4 MKL CNR COMPATIBLE | `1e78d5a02996b3e4` | no | 1581 | `f6c51898bf02` |
| emulated EPYC-Rome-v4 MKL CNR AVX2 | `717ea16909ad9682` | no | 1581 | `f6c51898bf02` |
| emulated EPYC-Rome-v4 MKL CNR COMPATIBLE | `1e78d5a02996b3e4` | no | 1581 | `f6c51898bf02` |

The archive's step-50 fingerprint is `b2da3ce2c10fdc29`. There are 8 distinct fingerprints across 16 tested paths, and none matches. The archive host reported AVX2; emulated Intel AVX2 and AMD Zen 2/3 CPUs still miss it. Exact replay therefore needs that machine or its exact MKL kernel path.

## 2. How fast the difference grows

Relative parameter difference, G / particles / D, between the two native paths:

| G update | Relative difference |
| ---: | --- |
| 1 | 2.2e-08 / 6.6e-09 / 2.2e-07 |
| 2 | 2.4e-08 / 1.3e-08 / 2.2e-07 |
| 5 | 3.0e-08 / 3.5e-08 / 2.2e-07 |
| 10 | 4.4e-08 / 6.1e-08 / 2.2e-07 |
| 20 | 7.7e-08 / 1.2e-07 / 2.6e-07 |
| 50 | 2.0e-06 / 6.5e-06 / 6.2e-06 |
| 100 | 1.5e-05 / 3.2e-05 / 1.6e-05 |
| 150 | 2.7e-04 / 6.3e-04 / 1.7e-04 |
| 200 | 2.4e-03 / 8.3e-03 / 9.2e-03 |
| 250 | 1.7e-02 / 2.4e-02 / 1.6e-02 |
| 300 | 3.5e-02 / 7.9e-02 / 6.8e-02 |
| 400 | 3.5e-02 / 7.6e-02 / 6.6e-02 |
| 600 | 5.8e-02 / 1.1e-01 / 9.4e-02 |
| 800 | 5.7e-02 / 1.4e-01 / 1.1e-01 |
| 1000 | 7.0e-02 / 1.5e-01 / 1.4e-01 |
| 1200 | 6.9e-02 / 1.6e-01 / 1.4e-01 |

Adam uses beta1=0, so the first update is nearly sign-like. Parameters with tiny gradients turn ULP gradient changes into parameter changes of about 1e-6. The GAN then grows the difference by roughly e every 20 updates until it saturates near update 300 of 1,200. The five scored checks at updates 1,000–1,200 are no longer tied to the archived trajectory. At this rate float64 would delay saturation by only several hundred updates. That estimate is extrapolated, not run.

## 3. What makes fresh runs portable

With `MKL_CBWR=AVX2`, the full 1,200-update rare-mode run is **bit-identical** between native Intel AVX512 and an emulated Intel Haswell (AVX2) CPU, excluding timing. That run is **FAIL** with 0 final passing checks. The same setting does not cover AMD: emulated Zen 2 gives another fingerprint. CNR COMPATIBLE unifies emulated Intel and AMD but not native AVX512, because the MKL vector sqrt still differs. A cross-vendor exact mode needs code-level control of these two kernels, not only environment variables.

| Fresh numerical path | Required | Practical | Total | Passing archived trials | Cases without support |
| --- | ---: | ---: | ---: | ---: | --- |
| Archived host | 9/9 | 10/10 | 19/19 | 46/46 | — |
| This host, default | 8/9 | 8/10 | 16/19 | 39/46 | mode_hold, vector_unequal_mass, img_bars4 |
| This host, MKL CNR AVX2 (Intel-portable) | 8/9 | 9/10 | 17/19 | 38/46 | mode_hold, vector_unequal_mass |

The required eight-mode ring and the rare 2% mode fail on both fresh paths. bars4 passes under CNR AVX2 but not default. Every other case retains a passing architecture on all three paths. The passing architecture can change; for example, both anisotropic D128 variants fail under CNR AVX2 while the original D passes.

## Recommendations

- Report the numerical path with seed 0. For future evidence, run with `MKL_CBWR=AVX2` so any Intel AVX2/AVX512 host reproduces it exactly.
- Do not call a cell host-portable unless it passes on at least two numerical paths, for example default and CNR AVX2. The rare mode and ring currently fail that test.
- For cross-vendor exact replay, make Adam's sqrt correctly rounded and give narrow output layers a fixed-order reduction in a declared evaluation mode, then rerun all 19 cases. This changes archived numerics, so it would require new evidence.

## Reproduce

```bash
python3 -m reports.transfer_suite.host_replication.portability.paths /tmp/paths     # needs qemu-user for emulated CPUs
python3 -m reports.transfer_suite.host_replication.portability.trace /tmp/a.pt
MKL_CBWR=AVX2 python3 -m reports.transfer_suite.host_replication.portability.trace /tmp/b.pt
python3 -m reports.transfer_suite.host_replication.portability.analyze /tmp/a.pt /tmp/b.pt /tmp/growth.json
MKL_CBWR=AVX2 python -u -m reports.transfer_suite.host_replication.run --output /tmp/cnr-avx2
python3 -m reports.transfer_suite.host_replication.portability.build
```

[Summary](summary.json) · [Numerical paths](paths.jsonl) · [Growth and localization](default_vs_cnr_avx2.json) · [COMPATIBLE native vs emulated](compatible_native_vs_emulated_haswell.json) · [Cross-CPU full runs](cross_cpu/) · [CNR AVX2 replication](cnr_avx2_replication/run.log).

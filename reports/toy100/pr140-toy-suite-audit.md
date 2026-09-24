# PR #140 toy suite audit

Fresh 22-toy gate on the delayed-arm branch (`cursor/delayed-arm-g-adam-lr-e08e`, `ef4084a4`). Diagnostic only. The arm, ×0.5 scale, and 1200 threshold were not retuned. The three known hardware fails were not edited.

**19/22. Same pattern as the #118 / #107 baseline. No new failures.**

This is not 22/22 and not a production claim.

## Environment

| Setting | Value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cpu |
| CPU capability | AVX2 (`torch.backends.cpu.get_cpu_capability()`; `ATEN_CPU_CAPABILITY=avx2`) |
| Threads | OMP=1, MKL=1, OPENBLAS=1, `CUDA_VISIBLE_DEVICES=''` |
| Config | `configs/toy100/constraints_simple_regularization.json` |
| Config SHA-256 | `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7` |
| Gate | `toy-suite-common22-v1` |
| Wall clock | 2026-09-24 19:36:49Z–19:51:50Z |
| Shipped arm | `stall_reach_delayed_arm_g_adam_lr_half`, arm at update ≥ 1200 with modes ≥ 8 and HQ ≥ 0.9, G Adam ×0.5 |

The suite runner is `python -u -m benchmarks.toy_suite run` (same harness as #118). It trains `GANTrainer`. It does not enter `pr84_delayed_arm_g_lr`. Suite logs contain no `g_lr_arm` or `g_lr_apply` events. The arm stays enabled for the continuous-learning probe that #140 ships; this gate does not call that probe.

Independent `regrade` of the same directory returned the same 19/22. The installed-wheel public-default control was not run (`MISSING`), matching #118.

## Score

| Block | This run | #118 / #107 baseline |
| --- | ---: | ---: |
| 100-mode | 2/3 | 2/3 |
| Canonical transfer | 17/19 | 17/19 |
| Combined | **19/22 FAIL** | **19/22 FAIL** |

Delta vs baseline: **none**. The three fails are the pre-existing hardware set.

## Per problem

| Case | Result | vs #118 |
| --- | --- | --- |
| `grid100` | FAIL | same |
| `rotated100` | PASS | same |
| `staggered100` | PASS | same |
| `two_pole` | PASS | same |
| `trajectory` | PASS | same |
| `residual_student` | PASS | same |
| `unipolar` | PASS | same |
| `ae_gan_hold` | PASS | same |
| `cover_leftover` | PASS | same |
| `unused_token_hold` | PASS | same |
| `mid_scale_identity` | PASS | same |
| `mode_hold` | PASS | same |
| `vector_two_broad` | PASS | same |
| `vector_unequal_mass` | FAIL | same |
| `vector_unequal_width` | FAIL | same |
| `vector_anisotropic` | PASS | same |
| `vector_overlap` | PASS | same |
| `vector_spiral` | PASS | same |
| `img_stripes2` | PASS | same |
| `img_bars4` | PASS | same |
| `img_blobs4` | PASS | same |
| `img_intensity2` | PASS | same |

### The three fails (same receipts as #118)

- `grid100`: live 97/100 modes, HQ 0.910, 0/5 terminal checks. Radial ratio range 0.903–1.704. Holdout (100k): precision 0.90984, mass TV 0.07254, center RMS/σ 0.609, radial KS 0.0541.
- `vector_unequal_mass`: suffix 2. Mass TV 0.0650, min mass ratio 0.537. Smallest component mass 0.0107 vs target 0.020.
- `vector_unequal_width`: suffix 0. Component covariance error up to 0.697, min eigen ratio 0.142.

`rotated100` and `staggered100` first hit 100 modes at step 750.

## Recommendation

**Keep #140 on the continuous-learning track.** This suite does not show a toy regression against #107. It also does not exercise the delayed arm, so it does not add evidence for or against the stay-gate claim (114/120). Do not merge on this audit. Do not call the suite solved.

## Reproduce

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
python -u -m benchmarks.toy_suite run \
  --config configs/toy100/constraints_simple_regularization.json \
  --output artifacts/toy-suite/pr140-avx2
python -m benchmarks.toy_suite regrade --output artifacts/toy-suite/pr140-avx2
```

Driver log: `artifacts/toy-suite/pr140-avx2-driver.log` (gitignored). Graded report: `artifacts/toy-suite/pr140-avx2/compatibility.md`.

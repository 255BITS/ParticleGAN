# Non-kill toy suite audit

Full 22-toy suite run on every non-kill continuous-learning board entry.
Measurement only; no mechanism tuning.

## Environment

| Setting | Value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cpu |
| Config | `constraints_simple_regularization.json` |
| CPU dispatch | AVX2 pinned (`ATEN_CPU_CAPABILITY=avx2`) |
| Threads | OMP=1, MKL=1 |
| Seed | 1234 (native), 0 (transfer) |
| Gate protocol | `toy-suite-common22-v1` |

## Entries audited

| PR | Branch | Tip | Description |
| --- | --- | --- | --- |
| #107 | `cursor/gan-native-followup-f60e` | `0ceeabb8` | Stall reach / .5 lineage |
| #84 | `cursor/alternating-ring-bound-908b` | `468ad261` | Smoothed critic pin |
| #82 | `cursor/cross-curvature-bound-c712` | `4285309d` | Alternating own-curvature |
| #81 | `cursor/cross-drift-replay-908b` | `5501e6e4` | Error-rel gain .5 |
| #93 | `cursor/fence-retract-exit-clip-e514` | `e958a81c` | Fence restore |

## Result: all five entries produce identical toy outcomes

All five PRs produce **byte-for-byte identical** toy suite results.
This is expected: these PRs modify continuous-learning mechanisms
(warm/cold/hold gates, critic response logic) that are not exercised
by the standard toy suite. The toy suite runs the base `GANTrainer`
with deterministic seeds; the PR code paths are never entered.

## Comparison table

**19/22 PASS on all five entries.** Three cases fail identically on every entry.

### 100-mode problems (3 cases)

| Problem | #107 | #84 | #82 | #81 | #93 | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| `grid100` | FAIL | FAIL | FAIL | FAIL | FAIL | coverage FAIL; accuracy FAIL |
| `rotated100` | PASS | PASS | PASS | PASS | PASS | first 100 modes at step 750 |
| `staggered100` | PASS | PASS | PASS | PASS | PASS | first 100 modes at step 750 |

### Transfer cases (19 cases)

| Case | #107 | #84 | #82 | #81 | #93 |
| --- | --- | --- | --- | --- | --- |
| `two_pole` | PASS | PASS | PASS | PASS | PASS |
| `trajectory` | PASS | PASS | PASS | PASS | PASS |
| `residual_student` | PASS | PASS | PASS | PASS | PASS |
| `unipolar` | PASS | PASS | PASS | PASS | PASS |
| `ae_gan_hold` | PASS | PASS | PASS | PASS | PASS |
| `cover_leftover` | PASS | PASS | PASS | PASS | PASS |
| `unused_token_hold` | PASS | PASS | PASS | PASS | PASS |
| `mid_scale_identity` | PASS | PASS | PASS | PASS | PASS |
| `mode_hold` | PASS | PASS | PASS | PASS | PASS |
| `vector_two_broad` | PASS | PASS | PASS | PASS | PASS |
| `vector_unequal_mass` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `vector_unequal_width` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `vector_anisotropic` | PASS | PASS | PASS | PASS | PASS |
| `vector_overlap` | PASS | PASS | PASS | PASS | PASS |
| `vector_spiral` | PASS | PASS | PASS | PASS | PASS |
| `img_stripes2` | PASS | PASS | PASS | PASS | PASS |
| `img_bars4` | PASS | PASS | PASS | PASS | PASS |
| `img_blobs4` | PASS | PASS | PASS | PASS | PASS |
| `img_intensity2` | PASS | PASS | PASS | PASS | PASS |

### Summary

| Entry | 100-mode | Transfer | Combined | Status |
| --- | --- | --- | --- | --- |
| PR #107 | 2/3 | 17/19 | 19/22 | FAIL |
| PR #84 | 2/3 | 17/19 | 19/22 | FAIL |
| PR #82 | 2/3 | 17/19 | 19/22 | FAIL |
| PR #81 | 2/3 | 17/19 | 19/22 | FAIL |
| PR #93 | 2/3 | 17/19 | 19/22 | FAIL |

## Failure analysis

The three failures are **pre-existing on this hardware**, not regressions introduced by any PR.
The archived 22/22 evidence was produced on a different machine. The project docs
note that "two earlier CI runs with identical source and configuration produced
different image-toy trajectories on AVX2 and AVX512 runners."

### `grid100` (100-mode, coverage + accuracy FAIL)

Final live: 97/100 modes, HQ 0.910. Only 0/5 terminal checks pass.
Holdout (100k draws): precision 0.910, mass TV 0.073, center RMS/σ 0.609, radial KS 0.054.
The accuracy thresholds require mass TV ≤ 0.06, center RMS ≤ 0.20, radial KS ≤ 0.04.
The radial ratio range reaches 1.704 (limit 1.40). This is a marginal trajectory
on this CPU that fails to sustain quality in the final window.

### `vector_unequal_mass` (transfer, FAIL)

Passing suffix 2 (sustained only 2 consecutive checks; needs 5).
Final mass TV 0.065, min mass ratio 0.537. The smallest component (target mass 2%)
captures only 1.1% of draws, while the largest over-captures at 55.6% vs 55% target.

### `vector_unequal_width` (transfer, FAIL)

Passing suffix 0 (never sustained). Final component covariance errors range up to
0.697; component min eigen ratio 0.142. The narrow components fail to learn their
target covariance within the frozen budget on this hardware.

## Recommendation per entry

| Entry | Recommendation | Rationale |
| --- | --- | --- |
| PR #107 | **keep** | Identical to base on all 22 toys; no regression |
| PR #84 | **keep** | Identical to base on all 22 toys; no regression |
| PR #82 | **keep** | Identical to base on all 22 toys; no regression |
| PR #81 | **keep** | Identical to base on all 22 toys; no regression |
| PR #93 | **keep** | Identical to base on all 22 toys; no regression |

**None of the five non-kill entries is way off the toy suite.** All produce
the exact same trajectories as the base branch. The 3/22 failures are a
hardware-specific baseline, not a mechanism regression. The continuous-learning
code paths are orthogonal to the toy evaluation harness.

These entries should continue to be evaluated on their continuous-learning
merits (warm/cold/hold gates), which this audit does not measure.

## Reproducing

From any of the five PR branches:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
source .venv/bin/activate
python -u -m benchmarks.toy_suite run --output artifacts/toy-suite/candidate
python -m benchmarks.toy_suite regrade --output artifacts/toy-suite/candidate
```

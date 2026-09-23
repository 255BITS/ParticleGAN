# Three wide-gap islands: equilateral, local (unscaled) width

Does **not** claim a new formulation champion. Tip `gan_v3` / shared_c6 with the
published batchfeat distance-head witness still owns the native-scale unadjusted board.

## Principal

Classic wrong-lengthscale / mode-drop stress: three isotropic islands on an equilateral
triangle with pairwise distance matching `wide_gap_islands_x8` (`d=16`) while **local**
width stays moderate (`σ=1.0`). Distinct from:

- `wide_gap_islands_x8` (only two islands on a line)
- `triangle_gauge_x24` (isotropic scale of means *and* covariances)
- `eight_ring_wide_r8` (eight close circular neighbors; tip passed)

- Winner: published absolute kernels `[0.1, 0.25, 0.5, 1.0]` + `init_std=0.5`
- Control: lengthscale-free SimpleMLP with matched `init_std=4.0`

## Result (tip `510e005`, MKL_CBWR=AVX2, OMP/MKL threads=1)

| Arm | D | init_std | Live | Suffix | mass | sw1 |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Winner (published absolute) | batchfeat distance-head | 0.5 | **FAIL** | 0 | ~53/0/47 drop | 0.351 |
| Control (MLP + matched init) | SimpleMLP default | 4.0 | **PASS** | 16 | ~37/30/33 | 0.052 |

![scatter](three_island_scatter.png)

## Reproduce

```bash
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. python -u reports/transfer_suite/three_island_wide_d16/reproduce_arms.py
```

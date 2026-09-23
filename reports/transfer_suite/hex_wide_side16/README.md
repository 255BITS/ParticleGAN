# Hex-wide lattice: distant modes, local (unscaled) width

Does **not** claim a new formulation champion. Tip `gan_v3` / shared_c6 with the
published batchfeat distance-head witness still owns the native-scale unadjusted board.

## Principal

Classic wrong-lengthscale / mode-covering stress on the next lattice after the
2/3/4 wide-gap hits: six isotropic islands on a regular hexagon with
`side = circumradius = 16` (neighbor gap matching `wide_gap_islands_x8`) and
**unscaled** local width `σ=1.0`.

- Winner: published absolute kernels `[0.1, 0.25, 0.5, 1.0]` + `init_std=0.5`
- Control: lengthscale-free SimpleMLP with matched `init_std=8.0`

Distinct from `eight_ring_wide_r8` (8 modes on r=8; tip already PASSes),
`four_corner_wide_g8` (4 square), and `three_island_wide_d16` (3 equilateral).

## Result (tip `510e005`, MKL_CBWR=AVX2, OMP/MKL threads=1)

| Arm | D | init_std | Live | Suffix | mass% | sw1 |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Winner (published absolute) | batchfeat distance-head | 0.5 | **FAIL** | 0 | ~9/13/24/33/13/8 | 0.229 |
| Control (MLP + matched init) | SimpleMLP default | 8.0 | **PASS** | 8 | ~20/13/16/22/15/13 | 0.051 |

![scatter](hex_wide_scatter.png)

## Reproduce

```bash
MKL_CBWR=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. python -u reports/transfer_suite/hex_wide_side16/reproduce_arms.py
```

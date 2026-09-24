# Frozen PR84 around the delayed failure

The archived stationary hold fails at updates 1390, 1540–1570, 1910, 2060, and 2150. This replay uses the same extracted stencil, bounds .25/3, and constant nominal rates. It does **not** reproduce that 1390 event. On torch 2.14.0+cpu the first failure is the dense warm window, updates 1129–1132.

| Check | Archived hold | This replay |
| --- | --- | --- |
| Updates 1001–1200 | 200/200 | **196/200**. Fail 1129–1132. Minimum HQ .8662, 8 modes |
| Update 1390 | 8 modes, HQ .8787 | 8 modes, HQ 1.0 |
| Update 1540 | 7 modes, HQ .7886 | 8 modes, HQ .9988 |
| Update 1560 | 7 modes, HQ .9-class fail in that spell | 8 modes, HQ .9338 |

Clean support is the 12 prior particles. The graded HQ is the host's large evaluation sample, which is why a clean HQ of 1 can still grade below .9.

## What the steps do

Clean margin is the distance from the nearest particle to the HQ boundary (negative means already outside). `factor` is the G curvature scale. The cap is open whenever rho < .25.

The graded dip is a shrink, then one open step:

| Update | Eval HQ | Clean margin before | Applied max move | rho | factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1122 | .9917 | .062 | .044 | .265 | .94 |
| 1128 | .9521 | .030 | .037 | .258 | .97 |
| **1129** | **.8987** | **.010** | **.080** | **.171** | **1** |
| 1130 | .8662 | .000 | .026 | .421 | .59 |
| 1133 | .9182 | back inside | .029 | .240 | 1 |

From 1122 to 1128 the closest particle walks from .062 to .030 inside the ball. Update 1129 is the large step in that window (.080 versus about .03 on either side). Rho is .171, so the .25 cap does not scale it. The particle lands on the boundary. The evaluation sample is already below .9. The next update, which the cap does scale, is what leaves one clean particle outside. It is back inside at 1133.

The same pattern shows up later, still with the cap open, and the every-10 grade misses it:

| Update | Clean before → after | Margin before | Move | rho | factor |
| --- | --- | ---: | ---: | ---: | ---: |
| 1374 | 1.00 → .917, 8 modes | .006 | .047 | .131 | 1 |
| 1390 | 1.00 → 1.00 | .118 | .018 | .223 | 1 |
| 1532 | 1.00 → .917 | .013 | .043 | .201 | 1 |
| 1540 | 1.00 → 1.00 | .051 | .082 | .144 | 1 |
| 1558 | 1.00 → .917, **7 modes** | .032 | .050 | .100 | 1 |

1390 itself is a small step with margin .118. 1540 moves as far as 1129 and stays inside because the margin was .051. 1558 drops a clean mode for updates 1558–1561. The every-10 sample at 1560 still grades 8 modes and HQ .934.

So the destructive event is not a unique huge proposal. A particle is already within about one unbound step of the boundary, and the smoothed-field rho is below .25, so the cap stays open. A .05–.08 step then spends the margin. Neighbors with the same size step do not fail when the margin is larger.

## Rejected gadgets, not continued

These were fit before the continuation priority. Neither is a rest damper. Neither advanced.

| Attempt | Warm | Why it stopped |
| --- | --- | --- |
| Occupied slope floor on the stencil (`d_asym` flatness half, floor 1) | 15/200, min HQ .273, final 6 modes / .672 | The floor pushes particles out of modes |
| Curvature bound on `plain − smoothed` only | 199/200, fail 1172, HQ .887 | One warm miss. Not a fix for the open-cap step above, and not run out to 2400 |

Rest-damping G stays closed. No cold acquisition and no 2400 hold were run for these two.

Rows are in [motion.json](continuous-evidence/pr84-failure-1390/motion.json).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pg-env/bin/python -u reports/toy100/pr84_failure_1390.py --until 1200 --output /tmp/fail1129
/tmp/pg-env/bin/python -u reports/toy100/pr84_failure_1390.py --until 1570 --output /tmp/fail1390
```

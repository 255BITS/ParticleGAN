# Path-crossing G direction: warm failure, cold not run

The new rule fails the dense warm fork: **14/200**, minimum **4 modes / HQ .6670**, final **5 / .7351**. Cold trajectory, ring, and the 2400 hold were not run. The selected partial candidate remains the original PR84 G-only stencil.

This is one fixed ray set, not a radius, width, or seed sweep. Rates stay G/D .00425 and prior .0085. Curvature bounds stay .25 / 3. The five-point stencil is unchanged. No mode center, rest damper, or gate change is used.

## Rule

On G's critic input only, eight fixed angles are scored at radii 0.5, 1, 1.5, 2, 2.5, and 3. A sample is redirected when some nearer point on a ray scores below the sample and a farther point scores above it, that farther point is more than 1 away from every clean particle, and the sample belongs to the support particle nearest the hit. The output loss gradient is rotated onto that ray and its norm is kept, so Adam's direction changes and a zero local gradient stays at rest. Other samples keep the smoothed local gradient.

## Same-fork warm results

| Variant | Passing checks | Minimum | Final |
| --- | ---: | --- | --- |
| Identity | 200/200 | 8 / .9900 | 8 / .9990 |
| Constant Adam | 4/200 | 0 / 0 | 8 / .6453 |
| PR84 stencil, path off | 196/200 | 8 / .8662 | 8 / .9993 |
| Path crossing | **14/200** | **4 / .6670** | **5 / .7351** |

Identity matches the uninterrupted cold control. The path-off stencil misses updates 1129–1132 and otherwise holds 8 modes. That 196/200 warm dip matches PR84's reported fork, not the later 200/200 replay. A separate from-scratch ring on this interpreter reached 8 modes / HQ .9988, so the archived 7-mode cold ring did not replay here. The path arm is still worse than its own path-off control: failures start at update 1015 and modes fall to 4. Redirect counts on the path arm were nonzero (24–85 samples on the logged steps). The ray fires on a covered ring, not only on an empty missing mode, and the resulting direction change drops modes. No coefficient was retuned after that.

Receipts: [summary](continuous-evidence/path-crossing-warm/summary.json) and gzip forks under `continuous-evidence/path-crossing-warm/`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
python3 -u reports/toy100/path_projection_probe.py --phase warm --output NEW_WARM
```

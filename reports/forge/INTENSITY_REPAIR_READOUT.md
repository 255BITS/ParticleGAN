# Corrected image-data diagnostic

The corrected `img_intensity2` run still failed: **0/2 modes, HQ 0, and no
passing observations**. All 600 updates and 24 observations completed. The
reference `[0,1]` real-data clamp is now present; this single paired run shows
that its omission was not sufficient to explain the failed acquisition.

| Measurement | Corrected run |
| --- | ---: |
| Final mean RMSE | 0.175000027 |
| Final mass TV | 0.5 |
| Passing observations / required observations | 0 / 24 |
| Unintended RNG deviations | 0 |
| Paid supervised seconds | 16.837452506 |
| Optimizer-update phase seconds | 7.178471830 |
| Peak PyTorch allocated bytes | 67,791,872 |
| Peak PyTorch reserved bytes | 69,206,016 |

[Registration](calibration-lanes/develop-20260929-intensity-repair-v3/registration.json)
limits this diagnostic to one task and 1,800 seconds. It used fixed screening
seed 0, the same initializer and recipe, live weights, and the explicitly
sigma-zero image prior. Version 1 sampling receipts confirm clean enumeration;
training output noise remains enabled under its unchanged recipe. A2 recorded
zero actual applications, separate from its synthetic activation probe.

The [attempt](attempts/65265b43758f4dbd85b77ef2ed46ac4b/result.json) belongs to
request `495a30143da81785a2d9756d`, K3P revision
`ef34dc1682c082151c202c4549af6343fa8b309ac5cf71c8ad29098dc55e77e4`, and source
`fc012b47b3714f8b7f67e21ce4d1bf3ff2809b3ff5aa001cff00b36663d1d39d`.
GPU 0 was free of external training before this run. Its queue reservations
returned to zero.

The earlier [three-task baseline](DEVELOP_QUICK_SCREEN_READOUT.md) remains
unchanged under its original source. The corrected task does not import those
results or qualify the complete smoke screen. All independent references remain
unknown; [v3 calibration](calibration/develop-20260929-quick-v3.md) is blocked.
Do not expand the scientific matrix on this result. Compare the effective public
recipe and initialization with a bound solvable reference before selecting a
further substantive diagnostic. The separately registered physical GPU pilot
addresses queue behavior and cannot establish scientific adoption.

The [source audit](HOST_PROVENANCE_AUDIT.md) found that the published passing
intensity profile uses residual upsampling at width 16; this task records
the raw transpose12 host. Both have the same K3P rate constants. Several
vector and native reference architectures also differ from the historical
positive. A next architecture transfer probe must declare that difference and
its source explicitly; no historical pass can be copied into this cohort.

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-develop-20260929-intensity-repair-v3/progress.jsonl
```

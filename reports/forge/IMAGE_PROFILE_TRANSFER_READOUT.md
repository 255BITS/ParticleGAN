# Residual16 intensity transfer readout

The explicit published residual16 host **passed** with the current shared K3P
formulation. It recovered both intensity modes and passed its final eight checks
for **11.198816130 paid seconds**. This gives a current positive for one task;
full-reference calibration remains unknown.

| Measurement | Result |
| --- | ---: |
| Updates / observations | 600 / 24 |
| First and stable pass | step 425 |
| Five-check confirmation | step 525 |
| Final passing suffix | 8 checks |
| Final modes | 2 / 2 |
| Final HQ | 0.90625 |
| Final mode mass | 0.5 / 0.5 |
| Final mass TV | 0 |
| Final mean RMSE | 0.025580958 |
| Optimizer-update phase seconds | 5.573194696 |
| Peak allocated / reserved PyTorch bytes | 70,475,264 / 90,177,536 |
| Unintended RNG deviations | 0 |

The [task](../../configs/forge/tasks/img_intensity2_residual16.json) explicitly
selects residual upsampling at width 16 through the shared image-profile resolver.
The [registration](calibration-lanes/image-profile-intensity-v1/registration.json)
selected only this 600-update cell, with an 1,800-second maximum. GPU 0 was idle
before execution; another workload on GPU 1 was left untouched.

Rates, seed 0, named initializer policy, learned finite-cloud exception, clean
live scoring and sustained/terminal gates were retained. Architecture and width
are explicitly different from the earlier [transpose12 failure](INTENSITY_REPAIR_READOUT.md).
Same initialization policy does not mean identical weights across different
shapes. The measured result supports this architecture transfer; it is not an
exact replay of the historical fixture, a general causal estimate, or a robustness
claim. A2 was enabled but had zero eligible/applied updates, so this run provides
no measured A2 training benefit.

The adapter records the resolved card, actual G/D parameter counts (5,361/4,929),
all parameter shapes and initial state hashes. The
[immutable attempt](attempts/e13557902b3544a081defbcac092c136/result.json) and
[concluded readout](records/readout-7650981f9a48a164aebbd631.json) bind:

- Request `07a1b804b5321d5bc74d92bb`.
- Candidate `2cde93ebff750cc551fcfdae18d82b4edfc60f9d0aca49713968faa5823d3591`.
- Source `79501bcf6ed352dd594f02ea55a356fa70db45f05e8e1d90d1c0e960d5a0e551`.
- Origin code `b1145e70`; registration published at `68754bac` before execution.

The [new matrix](calibration/image-profile-transfer-v1.md) preserves all 16
independent reference tasks, the other two smoke tasks, three substantive
lineages and unchanged acceptance criteria. Only this one cell is measured;
the other 56 cells are unknown. Adoption is blocked, with no receipt issues.

Next, bind the published vector/native architecture and initialization policies
explicitly through shared public components. The [host audit](HOST_PROVENANCE_AUDIT.md)
identifies those differences. Select further diagnostics narrowly under their
own frozen identities; do not fill a matrix whose intended host contracts are
still changing.

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-image-profile-intensity-v1/progress.jsonl
```

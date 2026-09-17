# Held-out particle usage

100,000 real examples per arm; hard selections audited against original reconstruction metrics. Lower TV is better; higher effective usage is more uniform.

| Arm | Used /400 | Effective /400 | Hard usage TV | Soft usage TV | Hard/soft TV gap | Hard chi-square |
|---|---:|---:|---:|---:|---:|---:|
| route_balanced | 275 | 156.2 | 0.5886 | 0.1040 | 0.5693 | 1.9761 |
| route_bounded | 387 | 227.7 | 0.4413 | 0.1185 | 0.4082 | 1.1839 |
| route_grad100 | 336 | 176.1 | 0.5463 | 0.1276 | 0.5054 | 1.7349 |
| route_local | 289 | 150.7 | 0.6044 | 0.1860 | 0.5569 | 2.0818 |
| route_local_balanced | 165 | 119.3 | 0.6872 | 0.1015 | 0.6740 | 2.5768 |
| route_noise | 331 | 174.6 | 0.5447 | 0.1429 | 0.5044 | 1.7671 |
| route_offset | 324 | 167.8 | 0.5643 | 0.1418 | 0.5267 | 1.8792 |
| route_zero | 343 | 193.0 | 0.5039 | 0.1295 | 0.4652 | 1.5497 |

Hard/soft TV gap is TV between the two aggregate routing distributions, not the difference of their distances to uniform. It does not measure gradient accuracy.

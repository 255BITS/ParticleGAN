# Held-out particle usage

100,000 real examples per arm; hard selections audited against original reconstruction metrics. Lower TV is better; higher effective usage is more uniform.

| Arm | Used /400 | Effective /400 | Hard usage TV | Soft usage TV | Hard chi-square |
|---|---:|---:|---:|---:|---:|
| route_balanced | 275 | 156.2 | 0.5886 | 0.1040 | 1.9761 |
| route_bounded | 387 | 227.7 | 0.4413 | 0.1185 | 1.1839 |
| route_grad100 | 336 | 176.1 | 0.5463 | 0.1276 | 1.7349 |
| route_noise | 331 | 174.6 | 0.5447 | 0.1429 | 1.7671 |
| route_offset | 324 | 167.8 | 0.5643 | 0.1418 | 1.8792 |
| route_zero | 343 | 193.0 | 0.5039 | 0.1295 | 1.5497 |

# Round 4: recent observations inside D memory

Six2k scouts following the failed one-write feedback round. Still one D-owned
memory tensor, no private G recurrence, fixed particle per episode, no expert at
runtime. D writes both a learned GRU component and a shift register of the most
recent2 or4 observed points. G only reads this tensor. Learned writer parameters
are optimized by D alone. Memory has32 scalar entries total:28 learned +4 raw
for2 points;24 learned +8 raw for4 points. This holds total memory size fixed,
not learned parameter count.

The residual reader outputs last_point + 3*tanh(network_output/3). From zero M,
last_point is zero. This smooth output parameterization bounds each coordinate
increment and prevents exponential numerical overflow during long evaluation.
It is not gradient clipping. Its range includes the toy's first-point support.
The absolute reader uses3*tanh(network_output/3). A GRU-only control uses the same
absolute output bound, separating this parameterization from the explicit slots.
This is a generic local-motion bias; no circle equation, fitted geometry, true
velocity/center/radius, or extra supervision enters training.

| Scout | Recent points | Learned GRU entries | G output | Feedback |
|---|---:|---:|---|---|
| recent_bound_control | 0 | 32 | bounded absolute | none |
| recent4_absolute | 4 | 24 | bounded absolute | none |
| recent4_delta | 4 | 24 | latest + bounded increment | none |
| recent2_delta | 2 | 28 | latest + bounded increment | none |
| recent4_delta_fb25 | 4 | 24 | latest + bounded increment | .25 probability |
| recent4_delta_fb50_mature | 4 | 24 | latest + bounded increment | .5 probability, prefix>=4 |

All other settings match prior2k scouts:10k schedule, batch128 x4 points,
max real prefix63, API exact B-cap defaults, no clipping, no EMA, no seed sweep.
No auxiliary losses. Feedback rows have one detached generated write / two G
calls per phase; others have one G call. No full generated training rollout.
All256/1024 rollouts are post-training evaluation only.

Whole-memory norm/saturation diagnostics now include raw coordinate slots and
are not directly comparable to all-GRU saturation. Circle/fidelity metrics remain
unchanged.57 tests passed, including explicit slot ordering, gradient ownership,
active B-cap through the combined writer, causal feedback, bounded call counts,
residual base point, output bound, and exact resume. Full GPU smoke passed.

Queue `runs/memory_path/recent_round4`, completion-only notifications and automatic
completed-result reporting. Shared tail:

```sh
tail -F runs/memory_path/core_round1/train.log
```

# Completed follow-up: retain the2k candidate

The exact proposal_mixed_pair25 continuation2k->5k and matched no-adapter2k scout
both completed with zero failures. No further runs are selected. All full cold
and warm passes remain0/128 at256/1024 and prefixes8/32; late stopping remains0.

| Model | Updates | Q prefix8 /32 | Late Q prefix8 /32 | Radial prefix8 /32 | Early position error prefix32 |
|---|---:|---:|---:|---:|---:|
| proposal_clean_s25 | 2000 | 0.006190 / 0.006426 | 0.005767 / 0.006069 | 1.497 / 1.487 | 1.448 |
| proposal_mixed_pair25 | 2000 | 0.008180 / 0.008274 | 0.006662 / 0.006773 | 1.136 / 1.159 | 1.528 |
| plain_mixed_pair25 | 2000 | 0.005702 / 0.005679 | 0.004442 / 0.004921 | 1.708 / 1.707 | 1.608 |
| proposal_mixed_pair25_5k | 5000 | 0.006748 / 0.006976 | 0.005327 / 0.005331 | 1.559 / 1.565 | 1.252 |

The5k continuation regresses from its2k checkpoint on Q, late Q, radial error,
and longest correct arc. Early position accuracy improves. Local next-point
MSE at prefix32 improves.005055->.004269; following-point MSE after a generated
write improves.017637->.012765. These are evaluation metrics only. Improving
local prediction therefore does not imply improved autonomous orbit fidelity.

The matched no-adapter control is worse on Q and radial error, supporting repair
as part of the current recipe. The comparison also changes runtime/training
reader compute and adapter capacity; it does not isolate sample conditioning
from all capacity effects. It fails the original extension gates versus the old
candidate, so no second continuation is warranted.

At5k, matched radius/speed history interventions still have near-zero median
late process response. Only6.25% get both original and reversed directions right
by late mean angular motion, versus10.9% at2k. These small fixed-panel differences
are descriptive, not significance claims. The probes show weak control of late
outputs by intended process parameters; they do not distinguish information loss
in D memory from G failing to decode/use retained information.

Recommendation: keep proposal_mixed_pair25 at2k as the new comparison baseline.
Keep cold generic-circle quality, warm original-process fidelity, and phase error
separate. Next investigate where process identity stops influencing the loop,
using evaluation interventions before adding more training losses or updates.
No full training rollout or MSE objective is needed for that investigation.

Extra training:261.29s for the3k continuation,168.76s for the matched2k scout.
Original2k checkpoint is preserved. Source hashes match all16 initial scouts;
resume changed only name, steps and checkpoint path, with unchanged10k schedule.

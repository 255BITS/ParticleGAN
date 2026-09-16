# Next discussion after round12

Completed14 runs (12scouts+2exactextensions). Nothing running. This round is
included in the user-requested commit. Push was not requested. After compaction,
discuss the next experiments before selecting or queuing another round.

New leading recipe: match_shuffle25, based on old proposal_mixed_pair25 plus
D-only mismatched real-continuation ranking with the SAME point score head used
by G. Weight.25, loss/penalty normalized by1.25, default exact B-cap. No new
persistent state, MSE training objective or generated trajectory training.
Nominal reference is5k by fixed minimum-Q ranking and improved radial/early
errors;2k has better lateQ and essentially equal whole-horizonQ. Keep both.

Paths:
- runs/memory_path/principles_round12/runs/match_shuffle25
- runs/memory_path/principles_round12_followup/runs/match_shuffle25_5k
- reports/memory-handoff/principles_round12/assessment.md
- reports/memory-handoff/principles_round12/followup/assessment.md

Results: warm Q~34% above old baseline, radial~19–21% lower at5k, but all
cold/warm full passes still0/128. Correct arcs remain short. D now recognizes
wrong continuations much better (nearest-history ranking89% versus67% at
prefix32), yet radius/speed retention remains weak. Shuffled negatives beat
nearest negatives. Nearest25 improved at2k but regressed at5k despite better
local prediction. Prefix observation-noise recovery, direct future-query GAN,
and combinations did not win. Future architecture control is available.

Potential next hypothesis, NOT selected or queued:
The new mismatch objective currently uses ONLY real-prefix memories. Test the
same continuation discrimination after a bounded generated write, with real
continuations providing the identity reference. This would teach D's writer
and point head to retain useful distinctions under generated observations,
using the existing one-write budget. Compare clean/explored/mixed mismatch
contexts on top of the shuffled25 recipe. The winning signal currently improves
real-history discrimination; this asks whether it can improve state maintenance.
Do not confuse it with existing mixed point-GAN judging or pretend generated
feedback has never been tried. No guarantee it solves repeated dynamics.

Late restoration improves the immediate output but does not sustain fidelity.
D-gradient alignment is diagnostic only and is confounded by phase drift when
compared with a timed target. No sole-cause diagnosis or proof of erased state.
No further experiment is selected or queued; this records a proposal for discussion.

Constraints persist: no seed sweeps, no MSE training, no generated full rollout,
no circle-specific cursor/labels in training, API B-cap defaults, no clipping/EMA,
fixed particle, D-owned M central, runtime no expert. Metrics for decisions,
completed-only inspection, existing two-GPU pipeline. Stable tail remains:
tail -F runs/memory_path/core_round1/train.log

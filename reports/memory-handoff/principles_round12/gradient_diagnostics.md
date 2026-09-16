# Sample-gradient diagnostic

Added after the first completed scouts; evaluation only, no change to selection.

| Model | Real-prefix32 positive alignment | Late autonomous alignment | Restored real-state alignment |
|---|---:|---:|---:|
| proposal_mixed_pair25 | 60.2% | 25.0% | 62.5% |
| match_nearest25 | 48.4% | 21.1% | 62.5% |
| match_shuffle25 | 70.3% | 32.8% | 67.2% |

Each alignment is the cosine between ascending the point-head score at G output
and target-minus-output. The real-prefix test uses a noisy next sample; late tests
use the original clean next sample. This is a local diagnostic, not a recovery
guarantee. In particular, autonomous phase drift can make movement toward the
original timed target differ from movement that would recover the orbit. Do not
interpret a negative cosine as proof that every useful correction is discouraged.
No parameter gradients accumulate, no optimizer runs, and this probe never
enters training. Restoration also jumps implied position/phase.

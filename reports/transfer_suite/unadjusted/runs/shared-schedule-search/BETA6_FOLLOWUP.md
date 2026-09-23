# Center-fixed β6 discriminator × global cosine schedule

Four previously frozen global schedule recipes were screened again with one
declared rare-case discriminator, `normstruct_center_fixed96_beta6`. The
discriminator source is byte-for-byte identical to the archived implementation
from commit `7685fbb750a581049b9c8848c9912faaf7b6f31f`. Every other case
retains the existing 18-case discriminator profile. The shared loss,
regularization, G/D/particle rates, Adam betas, data, generator, resources,
budget and behavioral gates are unchanged within each schedule candidate.

| Global hold / floor | Final min eigen ratio | Final covariance error | Final HQ | Passing observations | Final suffix |
| --- | ---: | ---: | ---: | ---: | ---: |
| .30 / .01 | .00843 | .63282 | .96484 | 0 | 0 |
| .30 / .05 | .00580 | .59378 | .96973 | 0 | 0 |
| .40 / .01 | .02307 | .72290 | .96191 | 0 | 0 |
| .50 / .01 | **.17555** | .57207 | .96118 | 4 | **3** |

The .50/.01 candidate passes every final metric but has only three consecutive
final passing observations. The required suffix is five. Its only late gate
failures are the minimum eigen ratio at steps 1000 and 1050:

| Step | Min eigen ratio (≥.15) | Covariance error (≤.85) | HQ (≥.85) | Sliced distance (≤.18) | Mass TV (≤.15) | Min mass ratio (≥.25) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | .11015 | .56207 | .96143 | .06331 | .01606 | .94645 |
| 1050 | .14599 | .58308 | .97241 | .07111 | .01606 | .94645 |
| 1100 | .18699 | .51360 | .95337 | .05588 | .01606 | .94645 |
| 1150 | .18041 | .56070 | .95825 | .07335 | .01606 | .94645 |
| 1200 | .17555 | .57207 | .96118 | .06115 | .01606 | .94645 |

Step 950 also passes, but step 1000 breaks the streak. This is a meaningful
near miss, not a sustained rare-case PASS. Under the predeclared completion
rule, no candidate from this four-card screen was advanced to the remaining 18
hosts. EMA is separate and contributes no selection points.

A default .60/.05 full-step identity replay using the imported β6 critic
matches the archived β6 episode exactly in recipe, candidate, specs, critic,
actual optimizer groups, every live/EMA observation, every action and verdict.
[Identity checks](beta6_control/checks.json) · [identity source](beta6_control/source.tar.gz).
The four crossed trials retain [their plan](beta6_screen_plan.json),
[index](beta6_screen/index.json), [protocol](beta6_screen/protocol.json),
[exact source](beta6_screen/source.tar.gz), [curves](beta6_screen/README.md)
and [log](beta6_screen.log). Every one of the 120 action multipliers per trial
matches the public schedule equation exactly. A dry-run import of both schedule
indexes as separate recipe entries validated 523 total episodes with zero
errors; each remains 1/19 INCOMPLETE with two recorded D attempts.

The original Atlas numerical law with table AMSGrad on and particle L2 .02 passes the longer-horizon diagnostic: first passing read280, sustained confirmation294, and final13 passing reads. All four80-update combinations fail. Three320-update combinations remain unrun after a result-report timeout stopped the batch at a safe boundary; no failed GAN verdict is assigned to them.

This task checks escape from the zero-particle bank while the critic gradient remains small. Success requires mean absolute particle position>=.30 and median absolute critic gradient<=1 on the final five saved checks. It does not require complete coverage of both target poles. Spread is an ungated diagnostic. These are explicit optimizer/objective/horizon variants, with no rewrite of the original80-update Task, canonical winner or shipped-default claim.

| Table AMSGrad | Particle L2 | Updates | Verdict | Escape | Flatness | Spread std | Elapsed/error |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| on | 0.02 | 80 | FAIL (reused) | 0.00244565 | 0.01089700 | 0.00271984 | 19.655s; none |
| on | 0 | 80 | FAIL | 0.00328096 | 0.01282401 | 0.00341627 | 16.446s; none |
| off | 0.02 | 80 | FAIL (reused) | 0.01502315 | 0.00998517 | 0.01667638 | 20.862s; none |
| off | 0.0 | 80 | FAIL (reused) | 0.08783505 | 0.02218658 | 0.10288850 | 29.465s; none |
| on | 0.02 | 320 | PASS | 0.64538652 | 0.09771565 | 0.64605671 | 31.326s; none |
| on | 0.0 | 320 | UNRUN | — | — | — | Unrun; report timeout boundary |
| off | 0.02 | 320 | UNRUN | — | — | — | Unrun; report timeout boundary |
| off | 0.0 | 320 | UNRUN | — | — | — | Unrun; report timeout boundary |

The320-update run has96 complete pure scored observations on the original clock lattice, extended to320; final5 clocks are307,310,314,317,320. Its first24 ordered particles, targets, critic gradients and two metric values exactly match the protected92180-update run. Recipe.total_steps remainsNone; only the external stop cap changed. Therefore the measured difference appears after the original horizon. The bounded optimizer trace remains partial UNKNOWN, so full-state or optimizer-tail equality is not claimed.

Nine focused controls passed once, including an actual four-update prefix fixture matching protected932. Two fresh scientific cases completed on CPU1/gpus0 with separate300-second allowances; no baseline was rerun. All five new300 allocations remain counted once, including the three unrun reserved cells. The private software180-second envelope preserves its communication timeout and halt history; declared reserves are neither measured elapsed nor certified upper bounds. comparison.json records a pre-publication cost cut; subsequent publication, Source authoring/model time and persistence/timeout tails are not zero or an exact all-inclusive cost.

Each measured cell folder contains the exact declaration, Task variant, software proof, raw result, grade, initialization, parent process receipt, saved curve and independently verified result. Numeric FAIL still means a completed valid run. The archive contains the exact two Source leaves bound by the controls and declarations, based on repository commit9e55990ad3fbd58c081c9368950023a97a29f251. ROOT-only controls require the supplied retained-prefix bank; they are not installed into general test discovery.

To reproduce in an isolated copy of that commit, extract bound-source-leaves.tar.gz at its root. Run the control script with --checkout pointing there, --retained-prefix pointing to retained-932-numerical-prefix.json, its recorded SHA256, and a fresh --output file. Scientific reproduction requires a fresh bound declaration and owned attempt/output directory; never substitute a prior proof with different Source or credit this diagnostic as a canonical original-Task pass.

The measured result supports testing longer external horizons before discarding this family. The fastest passing factorial configuration remains unresolved until the other three declared320 cells are completed.

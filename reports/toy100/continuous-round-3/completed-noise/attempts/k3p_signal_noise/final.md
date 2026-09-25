K3P stays the selected base. None of the three signal-noise proposals passed its own ring hold, so none earned the 300-update extension or a shift recovery. The parent was not rerun and was not edited.

| Candidate | Hold | Extension | Deadline recovery | Pre-shift |
|---|---|---|---|---|
| K3P (published) | 1200/1200 | 300/300 | 28/81 | pass |
| sn2 absolute-gap noise | not converged, max 7 modes at HQ 1 | not run | 0/81, min modes 6, min HQ 0.811 | 0/120 |
| sn3 settle cosine | not converged, stuck at 5 modes, HQ 0.999 | not run | 0/81, min modes 1, min HQ 0.166 | 0/120 |
| sn1 relative score gap | not converged, input noise stuck at 0.5 | not run | 0/81 | 0/120 |

Toy gates, the frozen recovery control, the two-horizon prefix, and the repeated-shift stress were not run.

The relative score gap never went quiet, so sn1 never left full rate or the initial noise burst. Switching to an absolute gap did turn input noise off by about step 250 and reached 7 sharp modes, but a gradient-size floor never opened because full rate kept the gradient above its early peak, and noise reopen pulses lined up with collapses. sn3 then ran a cosine after that burn-in and did reach the 1% network floor, but it floored the prior at the same time. Coverage stopped at 5 modes with HQ 0.999. On the shift, a 2.5× local gradient test never held for the 30 steps required to reopen, because the baseline rose with the new gradient. Adam still took all 3600 steps, at the floor.

The next change is to keep that burn-in with no noise reopen, leave the prior at its base rate through acquisition, and floor only the network after coverage can finish. Details, hashes, and replay commands are in the attempt `result.md`.

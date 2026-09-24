# Cross-only drift replay

Attribution only. Gates were not relaxed. Nothing here is a production recipe.

Torch in this run is 2.14.0+cpu. The archived cross-only failure was recorded
on 2.13.0+cu126. The fresh scheduled warm-state hash is
`57c10b571a2538075ea842cd0c9ea6f0ff677062ae9287e8e3c21d1241c48861`.
Identity fork parity held. Constant Adam still fails the local window (4/200).

## Reproduced warm failure

Stock cross-only, official fork, updates 1001–1200:

| Update | HQ | What moved |
| --- | ---: | --- |
| 1001 | .9907 | matched |
| 1002 | .8938 | G output RMS .203, prior RMS .013, total .213 |
| 1003 | .9993 | recovered |

Update 1002 tries α=1, cross residual .757, rejected. Accepted α=.5, cross
residual .390, correction/explicit norm about 1.06. Own/cross finite-difference
ratio about 3.8. Archived cold acquisition ratios are about 35–50, so own-player
curvature is larger while the target is still being acquired than at this
matched-state failure.

The archived 1194–1197 HQ dip (minimum .8894, G output RMS about .02–.04) did
not reappear. Those steps remain in
`continuous-evidence/cross-implicit/warm/cross_only.json.gz`.

## Cap that was tested

Optional `matched_output_rms_limit=.05`: if every clean particle is inside the
HQ ball and mean nearest distance is at most .12, a proposal whose clean output
RMS exceeds .05 is rejected and α is halved. Trajectory has no mode centers, so
the cap does not run there.

| Check | Result |
| --- | --- |
| Warm local, cap on | 200/200, minimum HQ .9331, 39 guard rejections |
| Cold trajectory, cap on or off | MSE .069011, 0/24 passing checks |

Cold mode-hold, extended hold, and the shift test were not run.

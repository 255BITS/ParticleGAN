# Correcting cached contact flags

The archived calibrated `slow.pt` and `fast.pt` are the exact live checkpoints
previously reported as 28/30 and 29/30 on seeds 94000–94029. `report.json`,
`config.json`, and `validation.json` preserve that historical scoring and
selection; they are not the corrected current command result.

`frozen_rescore.json` replays these same weights on the same 30 worlds and
requires both legs to have enabled, touching contacts with the terrain, with
the lander asleep on the pad and no crash. Both policies score **30/30**.
No retraining, action change, EMA, or relaxed physical landing requirement
produced this correction.

`event_trace.json` identifies the bug on fast seed 94020: at step 169,
Gym's `EndContact` clears the left leg's cached flag while another enabled
ground contact remains. At termination step 202 both legs physically touch
terrain, but the cached flags still say only one does. The fast expert has
the same false negative. `terminal_contact_replays.json` records this case
and both slow false negatives, including their final physical contacts.

The current evaluator reads Box2D contact edges. Gym observations, physics,
expert actions, and network inputs are unchanged. The correction also affects
which collected expert flights qualify as policy targets, so a fresh command
run is recorded separately in the parent evidence folder. The 94000 cohort
is now an inspected regression cohort, not a new untouched test set.

Hashes in `manifest.json` identify the archived checkpoints and records.

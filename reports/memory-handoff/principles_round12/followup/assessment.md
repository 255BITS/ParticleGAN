# Round12 exact extensions: retain shuffled mismatch training

Both selected2k checkpoints continued exactly to5k on the unchanged10k schedule.
Two completed, zero failures; no jobs remain. All full cold/warm passes remain
0/128 at256/1024 and prefixes8/32; late stopping remains zero.

| Model | Minimum warm Q | Q prefix32 | LateQ prefix32 | Radial32 ↓ | First32 error ↓ | Correct arc turns mean |
|---|---:|---:|---:|---:|---:|---:|
| match_shuffle25_5k | 0.011008 | 0.011099 | 0.009708 | 0.912 | 0.986 | 0.0741 |
| match_shuffle25 | 0.010901 | 0.011161 | 0.010869 | 0.945 | 1.237 | 0.0640 |
| match_nearest25 | 0.009932 | 0.009932 | 0.007620 | 0.994 | 1.293 | 0.0662 |
| proposal_mixed_pair25 | 0.008180 | 0.008274 | 0.006773 | 1.159 | 1.528 | 0.0588 |
| match_nearest25_5k | 0.007605 | 0.008121 | 0.005370 | 1.579 | 1.238 | 0.0849 |

## Decision

Use match_shuffle25_5k as the nominal new reference by the predeclared minimum
warm-Q ranking; retain match_shuffle25 at2k as the stronger late-window control.
The5k minimumQ advantage over its2k checkpoint is only~1%, not a convincing
new quality regime. Q32 is essentially unchanged (-.6%), radial error improves
~3.5–4.6%, early position error improves15–20%, and mean correct arc length
improves8–16%. LateQ worsens~20% at prefix8 and~11% at prefix32. No further
length extension is justified. The recipe is the finding; keep both checkpoints.

Relative to the old proposal_mixed_pair25 baseline, shuffled5k improves Q by
~34–35% and radial error by~19–21% across both prefixes. These remain small
absolute Q values and zero full-circle success. Phase/history fidelity is unsolved.

Nearest25 degrades with longer training: Q falls~18–24%, radial error worsens
~58–59%, and lateQ declines. Local noisy-next-point evaluation MSE still improves
0.004296→0.002997.
This repeats the local-accuracy/autonomous-dynamics disconnect. Keep its2k
checkpoint only as a comparison, not the primary direction.

## Process retention

Shuffled5k median late radius response is-.0134 and speed response-.0021
(ideal1). Both original/flipped mean directions are correct for14.1%, versus
9.4% at2k, but this remains weak and is a weaker test than per-step correctness.
Nearest5k also has near-zero radius response and poor speed response. No
claim of useful radius/speed retention follows from the quality gains.

D nearest-history ranking at prefix32 is89.1% for shuffled5k (88.3% at2k),
versus67.2% old baseline. D classification improved substantially; the long
closed-loop process still does not retain the requested dynamics.

## Execution

Followup queue wall364.7s (~6.1min),711.9 additional training GPU-seconds.
No training code changed between scouts and extensions; archived source hashes
match. Reference paths and observed prefixes are bitwise identical to baseline.
Both exact resumes retain optimizer/prior/RNG state and10k schedule.

The entire round used12 fresh2k scouts plus two2k→5k continuations:3728 training
GPU-seconds (~62.1GPU-min), about32.6min combined queue wall time, excluding
implementation, smoke checks and CPU diagnostics.

Artifacts: [leaderboard](leaderboard.md), [results](results.json),
[signal probes](signal.json), [process probes](process.json), [execution](execution.json).

# Original moving-task score comparison

HQ is the percentage of 20,000 noisy generated samples within Euclidean distance 0.09 of the nearest current target center.
A mode counts when at least 10 of those HQ samples are assigned to it.
Each turn requires at least 95 modes and HQ ≥90% of that task’s update 500 baseline; the same baseline applies to both turns.

|Task|Turn / update|HQ minimum|RA14 HQ / modes|RA15 HQ / modes|HQ change|RA14 → RA15|
|---|---|---:|---:|---:|---:|---|
|grid100|+30° / 1000|85.176%|94.16% / 100|93.79% / 100|-0.37 pp|PASS → PASS|
|grid100|+60° / 1500|85.176%|96.24% / 100|96.20% / 100|-0.04 pp|PASS → PASS|
|rotated100|+30° / 1000|86.481%|95.48% / 100|96.09% / 100|+0.61 pp|PASS → PASS|
|rotated100|+60° / 1500|86.481%|85.57% / 99|92.59% / 98|+7.02 pp|FAIL → PASS|
|staggered100|+30° / 1000|85.932%|93.25% / 99|93.40% / 100|+0.15 pp|PASS → PASS|
|staggered100|+60° / 1500|85.932%|97.28% / 100|94.76% / 100|-2.52 pp|PASS → PASS|

RA15 repairs the rotated100 second-turn failure. HQ does not improve uniformly: grid and staggered remain passing with some lower scores.

Both versions use the original seed 1234, two 30° turns, 1,500 updates, and unchanged scoring and acceptance rules.
These rows label the actual historical sources RA14 and fresh RA15; they are not relabelled as RA17 executions.

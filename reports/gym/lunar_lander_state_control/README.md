# State-only Lunar Lander control

Selection uses validation landing rate, then mean return. Test worlds are paired and fresh for this round. Intervals are 95% Wilson intervals over evaluation episodes.

| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit |
| --- | --- | ---: | --- | ---: | ---: | --- |
| imitation | selected (2500) | 50/50 | 92.9%–100.0% | 283.16 | 281.84 | 0 / 0 / 0 |
| expert | selected (0) | 50/50 | 92.9%–100.0% | 282.45 | 283.10 | 0 / 0 / 0 |
| probes | selected (2500) | 50/50 | 92.9%–100.0% | 282.24 | 280.67 | 0 / 0 / 0 |
| probes | final (2500) | 50/50 | 92.9%–100.0% | 282.24 | 280.67 | 0 / 0 / 0 |
| auxiliary | selected (2500) | 27/50 | 40.4%–67.0% | 130.92 | 213.13 | 22 / 0 / 1 |
| auxiliary | final (2500) | 27/50 | 40.4%–67.0% | 130.92 | 213.13 | 22 / 0 / 1 |

Full action traces, paired comparisons, inference/training costs, engine usage, expert reconstruction and learner-successor metrics are in `leaderboard.json`. G3 has no alternative-action input; its consistency with G2 is measured on actual learner rollouts.

# Lunar Lander with scarce action labels

Selection uses validation landing rate, then mean return. Test worlds are paired and fresh for this round. Intervals are 95% Wilson intervals over evaluation episodes.

| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit |
| --- | --- | ---: | --- | ---: | ---: | --- |
| expert | selected (0) | 50/50 | 92.9%–100.0% | 280.97 | 281.78 | 0 / 0 / 0 |
| full_label | selected (2500) | 49/50 | 89.5%–99.6% | 271.11 | 275.39 | 1 / 0 / 0 |
| probes | selected (2500) | 44/50 | 76.2%–94.4% | 224.47 | 255.09 | 2 / 0 / 4 |
| probes | final (2500) | 44/50 | 76.2%–94.4% | 224.47 | 255.09 | 2 / 0 / 4 |
| auxiliary | selected (2500) | 34/50 | 54.2%–79.2% | 126.41 | 237.66 | 2 / 11 / 3 |
| auxiliary | final (2500) | 34/50 | 54.2%–79.2% | 126.41 | 237.66 | 2 / 11 / 3 |

Full action traces, paired comparisons, inference/training costs, engine usage, expert reconstruction and learner-successor metrics are in `leaderboard.json`. G3 has no alternative-action input; its consistency with G2 is measured on actual learner rollouts. The full-label reference uses its original scaler; normalized errors are directly comparable between the two sparse arms, while physical errors support comparison to the reference.

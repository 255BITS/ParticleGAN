# Lunar Lander control baseline

Checkpoints and playable default are chosen on validation landing rate, then mean return. Test results below are held out; final checkpoints are reported separately. Intervals are 95% Wilson intervals over the finite paired reset episodes.

| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit | Main / side usage |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| expert | selected (0) | 50/50 | 92.9%–100.0% | 287.29 | 287.53 | 0 / 0 / 0 | 57.2% / 12.7% |
| imitation | selected (2500) | 50/50 | 92.9%–100.0% | 286.76 | 286.80 | 0 / 0 / 0 | 57.1% / 13.6% |
| imitation | final (2500) | 50/50 | 92.9%–100.0% | 286.76 | 286.80 | 0 / 0 / 0 | 57.1% / 13.6% |
| joint | selected (2500) | 12/50 | 14.3%–37.4% | 74.87 | 26.48 | 37 / 1 / 0 | 40.8% / 18.3% |
| joint | final (2500) | 12/50 | 14.3%–37.4% | 74.87 | 26.48 | 37 / 1 / 0 | 40.8% / 18.3% |
| original | selected (1000) | 0/50 | 0.0%–7.1% | -374.53 | -430.87 | 33 / 17 / 0 | 44.4% / 57.2% |

Per-episode results, paired return/landing wins, action error, route counts, simulator cost, and inference latency are in `leaderboard.json`. Full state/action traces are in `traces/`. The expert is a reference and is excluded from the learned-controller default selection.

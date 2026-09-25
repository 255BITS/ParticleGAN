# Lunar Lander GAN-only control leaderboard

All listed controllers trained with adversarial losses. New joint/marginals arms use five labeled episodes from scratch; legacy_joint was pretrained and fine-tuned with 47 labeled episodes and is an unmatched historical reference. Selection uses fresh paired validation landing rate, then mean return. Intervals are 95% Wilson intervals over evaluation episodes.

| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit |
| --- | --- | ---: | --- | ---: | ---: | --- |
| joint | selected (2500) | 34/50 | 54.2%–79.2% | 157.33 | 241.80 | 15 / 1 / 0 |
| joint | final (2500) | 34/50 | 54.2%–79.2% | 157.33 | 241.80 | 15 / 1 / 0 |
| legacy_joint | selected (2500) | 23/50 | 33.0%–59.6% | 137.35 | 43.81 | 27 / 0 / 0 |
| marginals | selected (2500) | 16/50 | 20.8%–45.8% | -0.72 | -48.60 | 20 / 14 / 0 |
| marginals | final (2500) | 16/50 | 20.8%–45.8% | -0.72 | -48.60 | 20 / 14 / 0 |

Full action traces, paired comparisons, inference/training costs, engine usage, expert reconstruction and learner-successor metrics are in `leaderboard.json`. G3 has no alternative-action input; its consistency with G2 is measured on actual learner rollouts. The legacy GAN reference uses its original scaler; normalized errors are directly comparable between the two sparse arms, while physical errors support comparison to the reference.

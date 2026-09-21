# Native RpGAN gate versus a state pad

The writeup is [docs/native16-autopsy.md](../../../docs/native16-autopsy.md).
This file is the leaderboard from one CPU run of
`experiments/toy_native16_autopsy.py`. It is not a Lunar score.

```bash
python -u experiments/toy_native16_autopsy.py
tail -F results/gym/native16_autopsy/live.log
```

`summary.json` is that run. Old gate: teacher-forced action MSE ≤ 0.18.
Honest gate: landing rate ≥ 0.80 after 24 steps with the learner's previous
command. Adversarial weight on the GAN arms is 1. The L2 row is a plant
check, not a Lunar setting.

| Arm | Teacher-forced MSE | On-policy MSE | Landings | Old | Honest |
| --- | ---: | ---: | ---: | --- | --- |
| Fixed live `(record, z)` | 0.0151 | 0.3099 | 0.326 | PASS | FAIL |
| Action-latent pair only | 0.0185 | 0.2452 | 0.574 | PASS | FAIL |
| L2 probe on this plant | 0.0082 | 0.0740 | 0.868 | PASS | PASS |

Analytic false passes on the same pad: bias +0.4 has MSE 0.1600 and landing
rate 0; copying the previous command has shuffled MSE 0.0273 and landing rate
0.082. A zero action on the original previous-only target has teacher-forced
MSE 0.5513 and closed-loop MSE 0.0689.

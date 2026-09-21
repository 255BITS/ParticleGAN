# Lunar Lander baseline leaderboard

Single transitions; 11 terrain heights are privileged context. Lower errors are better.

| Model/checkpoint | Next-state MSE | vs persistence | Contact Brier | Action-effect MSE | 20-step MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_final | 0.004820 | +58.3% | 0.00484 | 0.002304 | 0.231991 |
| direct_best | 0.007048 | +39.1% | 0.00450 | 0.003010 | 0.330456 |
| persistence | 0.011567 | +0.0% | 0.00525 | 0.012995 | 0.552613 |
| adversarial_best | 0.063822 | -451.8% | 0.00732 | 0.039382 | 0.520222 |
| reconstruction_best | 0.082694 | -614.9% | 0.00489 | 0.039649 | 0.640620 |
| adversarial_final | 0.204280 | -1666.1% | 0.00603 | 0.023113 | 1.634448 |
| reconstruction_final | 0.487572 | -4115.3% | 0.02862 | 0.246211 | 1.815117 |

Checkpoints selected by validation continuous MSE; final checkpoints also listed. Recursive contacts use p >= .5, and evaluation ends at reference termination. The deterministic encoder predicts a point; joint generation does not establish conditional uncertainty.

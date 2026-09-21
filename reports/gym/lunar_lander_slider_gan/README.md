# Slider-error versus L2 supervision

GAN-only ranking on paired fresh worlds; checkpoints/default selected by validation.

| GAN | Validation landings | Test landings | Mean test return |
| --- | ---: | ---: | ---: |
| joint | 15/20 | 37/50 | 186.88 |
| previous_l2 | 2/20 | 11/50 | 13.95 |
| sliders_all | 1/20 | 6/50 | -26.40 |

Non-GAN imitation reference, excluded from ranking: 50/50, mean return 278.56.

Validation-selected GAN: `joint`. New model selected update: 2500.

The L2 arm shares initialization, data, minibatch streams, and budget; the slider arm replaces paired MSE/BCE with an extra adversarial critic. Other references have different histories.

See [the readout](READOUT.md) for interpretation and training costs.

# Previous-action GAN from scratch

GAN-only ranking on paired fresh worlds; checkpoints/default selected by validation.

| GAN | Validation landings | Test landings | Mean test return |
| --- | ---: | ---: | ---: |
| joint | 13/20 | 27/50 | 130.67 |
| marginals | 7/20 | 14/50 | -4.12 |
| legacy_joint | 6/20 | 28/50 | 159.83 |
| previous_marginals | 4/20 | 7/50 | -4.12 |

Non-GAN imitation reference, excluded from ranking: 50/50, mean return 285.92.

Validation-selected GAN: `joint`. New model selected update: 2500.

The references differ in labels, initialization, encoder, and losses; these are benchmark comparisons, not a one-factor ablation.

See [the readout](READOUT.md) for interpretation and training costs.

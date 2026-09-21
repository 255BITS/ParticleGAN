# Slider-error action fine-tune

No landing evaluation yet. The leaderboard cites existing readouts only.
This arm is the imitation fine-tune with action MSE replaced by the paired-error
critic. It is not ranked.

| Controller | Test landings | Mean test return | Status |
| --- | ---: | ---: | --- |
| Imitation L2 | 50/50 | 286.76 | control readout; different protocol from the slider rows |
| Joint L2 fine-tune | 12/50 | 74.87 | control readout |
| Previous-action L2 | 11/50 | 13.95 | slider readout, reselected |
| Scratch sliders, all heads | 6/50 | -26.40 | slider readout |
| Slider-error fine-tune | — | — | recipe and CPU smoke only |

See [READOUT.md](READOUT.md) for the loss graph and the single recommended run.

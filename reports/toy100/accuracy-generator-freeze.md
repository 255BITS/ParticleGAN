# Generator-only freeze diagnostic

The archived affine/square-prior rotated100 run reaches 100 modes at update
750, then collapses after update 1750. This [scratch probe](accuracy_freeze_probe.py)
repeats its exact initialization, recipe, seed, resources, and 7,000-update
budget, but sets only the generator optimizer rate to zero after update
1,000. The discriminator and learned prior retain their original rates and
schedules. This is an additional proposed training rule, explicitly **not**
evidence for the production common-22 recipe. No target labels, centers, or
samples initialize the model.

All recorded live metrics through update 1,000 match the archived control
exactly. Source, declared config, the freeze event, every checkpoint, five
terminal 20,000-draw clouds, and the independent 100,000-draw holdout are
retained locally at `artifacts/toy100-accuracy/affine-freeze/rotated100`.

| Update | Qualifying modes | HQ | Nearest-mode mass TV | Center RMS / σ |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 100 | .9180 | .04845 | .283 |
| 1,500 | 100 | .87895 | .04790 | .218 |
| 2,000 | 100 | .90170 | .04715 | .200 |
| 2,250 | 0 | .03980 | .07425 | unavailable |
| 2,750 | 1 | .03775 | .11255 | unavailable |
| 7,000 | 2 | .06880 | .15400 | unavailable |

Freezing G delays the loss of qualifying modes but does not prevent it. The
large fall in HQ occurs while nearest-center allocation is still relatively
broad; the cloud has moved away from the small mode disks. The subsequent
mode count alone should not be interpreted as every sample occupying just
two locations. Both the original coverage and strict accuracy gates fail,
with zero passing terminal checks. This rules out generator updates after
1,000 as the sole required cause of this run's later failure. It does not
identify whether the remaining discriminator dynamics, particle updates, or
their interaction caused it.

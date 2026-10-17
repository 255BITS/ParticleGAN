The zero-update reproducer checks **local stationarity against a frozen critic**.
It does not establish a joint GAN equilibrium or explain Supra's late convergence
gap. An arbitrary stale critic may push an exact generator because that critic
is not itself at its equilibrium.

The standalone observation used four actual qualified rotated-task critics,
twelve fixed source-balanced fit contexts, and one public-initialized particle
host with zero Up and preserved trainable C/bank/router. Each B4 batch used one
clean FAST routed forward. Its output minus the same detached output constructed
the exact zero residual; this target is not the rotated teacher. Private CPU43
noise panels were shared across all controls; antithetic signs used no new draws.

All 192 context games equaled log(2), within 1.91e-9. The generator's Up-gradient
norms across the four critics and twelve contexts were:

| Critic/noise law | Up-gradient norm range | Stationary contexts |
| --- | ---: | ---: |
| Current / single | 0.148892–0.652891 | 0/48 |
| Current / antithetic | 0.132347–0.487753 | 0/48 |
| Even / single | 0.007342–0.463978 | 0/48 |
| Even / antithetic | exactly zero | 48/48 |

The complete observation took 2.263 seconds on CPU with one thread, three actual
generator forwards and **zero model/optimizer updates**. Native state, tensors,
flags, modes, gradients, RNG, data and sources remained unchanged. Read-only
state snapshots and owner lookup methods are used; the zero native call count
refers to training/update/diagnostic hooks. Hook checks compare registration
keys, rather than certifying arbitrary hook callable identities. At zero Up,
code/H/b/C/down/bank/router
gradients necessarily vanish and do not diagnose particle inactivity later.
The exact continuous-residual negative-gradient slope is negative in the three
nonstationary controls. The reported Up slope is native autograd linearization
through BF16 casts, not a literal infinitesimal derivative of a quantized model.

**Cancellation alone does not imply safe attraction.** The minimal original
fixture D(x)=.7x+.2x² has zero force after even+antithetic correction but its
origin remains a local maximum of the generator game. A separate deterministic
concave fixture D(x)=.7x−.2x² has a wrong stationary attractor at residual
1.7103369216103252 under current antithetic pairing; even+antithetic gives a local
minimum at its correct zero. These scalar native-RpGAN derivatives are algebra
checks, not training results. Evenization also removes first-order signed mean
discrimination and may slow acquisition. An optimizer improvement remains
untested.

The frozen implementation, card, software receipt and compact actual results
are [the core](../examples/e22_routed_game_stationarity.py),
[the card](e22_routed_game_stationarity_v1.json),
[software checks](e22_routed_game_stationarity_software.json), and
[actual results](e22_routed_game_stationarity_results.json). Bulk artifacts remain
local under `runs/routed-game-stationarity-v1`.

The tests require no saved critic artifacts or quality training:

```sh
PYTHONPATH=. /ml2/ntc-image-studio/.venv-anima/bin/python -m pytest -q \
  tests/test_e22_routed_game_stationarity.py \
  tests/test_rpgan_local_stationarity_attraction.py
```

The additional attraction tests and their separate software receipt do not
change or rerun the completed saved-critic observation.

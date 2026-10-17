# R1 in the E22 integration loops

[R1](../../docs/r1-rotation.md) is part of the E22 preset. This report checks it in each caller-owned
integration: the loop settles, the target turns, and the loop runs with R1 (`reopen_signal="optimizer"`,
`reopen_anchor="release"`) and without it (`"none"`, `"hold"`, E22 before R1). Everything else is the same:
one seed, the same batches and the same initialization. CPU; `probe.py` prints one JSON line every 50
updates.

Routed loops report held-out clean RMSE. A turn rotates the paired edit (target minus the neutral host) on the
fitting, guard and held-out contexts. The external loop reports the share of 2,048 served samples within .087
of a corner of its 4-corner toy, and a turn rotates the corners.

| loop | turn | without R1 | with R1 | R1 fires |
|---|---|---|---|---|
| `e22_routed_paired` (`examples/e22_routed_moving.py`) | 30° every 500, two turns | .0021 / .0023 / .0034; .0815 50 updates after turn 2 | .0021 / .0014 / .0012; .0013 | one per turn |
| `e22_routed_sites` | 90° at 800 | 900: .014, 950: .027, 1000: .021, 1200: .018, 1400: .0077 | .065, .0093, .0087, .0077, .0071 | 849 |
| `e22_routed_sites` | 30° at 500 | 550: .0081, 900: .0070 | .0094, .0074 | 523 |
| `e22_routed_replay` (sites with activation-checkpointed generation) | 30° at 500 | identical to sites | identical to sites | 523 |
| `e22_routed_support` (tokens 16, particles 32) | none, 2,000 updates | 400: .0131, 800: .0055, 1200: .0044, 1600: .0039, 2000: .0042 | .0246, .0062, .0047, .0039, .0045 | 360 |
| `e22_external_loop` (particles 256, batch 64) | 30° at 800 | the toy never settles (its score swings between .95 and .10 either way) | identical | none |

- With a real turn (paired, and sites at 90°), R1 recovers several times faster. Sites at 30° is not a test:
  the turn barely moves the error without R1.
- Support shows the cost. Around update 350 the game destabilizes by itself: the fit error rises from .006 to
  .34 in 15 updates, while every structural proposal is rejected. R1 reads this as a moved target and fires.
  Recovery is then slower until about update 1,400; after that the two runs agree.
- `e22_routed_game` replays copied controllers and never calls the policy's lifecycle hooks. Its live-owner
  test (`tests/test_e22_routed_game.py`) compares the whole checkpoint before and after a replay, and the
  checkpoint includes the R1 detector's state.
- `e22_routed_sites.make_loop` and `e22_routed_support.make_loop` take `recipe_overrides=` (as the paired
  example does), so an integration can turn R1 off.

Reproduce (each run takes a few minutes on one CPU thread):

```bash
python reports/r1-integrations/probe.py sites 1 --turn 800 --steps 1400 --deg 90   # 0 = without R1
python reports/r1-integrations/probe.py support 1 --turn 100000 --steps 2000
python reports/r1-integrations/probe.py external 1 --turn 800 --steps 1400
python examples/e22_routed_moving.py --turn-every 500 --r1
```

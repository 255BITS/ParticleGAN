# R1: following a moving target

R1 is part of the [E22 preset](e22.md) (`get_recipe("e22")`, `get_recipe("e22_routed")` and
`configs/100gaussians/e22-noout.json`) through two fields:

```json
"reopen_signal": "optimizer",
"reopen_anchor": "release"
```

`"none"` / `"hold"` turns it off (E22 before R1). The Recipe defaults outside the E22 preset are unchanged.
On a static target it does not fire, and the run is then bit-identical to E22 without it. Harness evidence
(frozen harness, new-API package): native gates 3/3 at 7k and 14k, S4 3/3 at lr x.75 and x1.33, 0 fires in all
15 native runs; the 13-gate suite matches E22 on 11 gates with 0 fires, including vector_overlap after the
abrupt-rise rule below. ring_shift and stationary do not run on this API in the frozen harness (E22 as well).

## The test

The rotation test turns the real target of the native 100-Gaussian problems by 30° every 500 updates, twice
(1,500 updates, seed 1234, frozen native host). After each turn, a 20,000-point draw is scored against the rotated
centres: PASS needs at least 95 modes and a share within 3σ of at least 0.9× the pre-turn value.

| config | grid100 (bar .828) | rotated100 (bar .793) | staggered100 (bar .841) | passes |
|---|---|---|---|---|
| E22 | .920 / .707 / .702 | .881 / .855 / .876 | .934 / .857 / .700 | 1/3 |
| **R1** | .920 / .865 / .882 | .881 / .840 / .947 | .934 / .875 / .950 | **3/3** |

(share within 3σ at 0° / 30° / 60°.) On a static target R1 never fires, so its native 7,000-update results are
identical to E22's (grid .98485, rotated .98575, staggered .98265).

## Why E22 does not follow

1. **The turn is a gradient shock.** The generator's gradient grows ~1,000×, its Adam second moment grows ~80× and
   stays there (β2 = .999 plus the AMSGrad maximum), so the generator's steps shrink ~50× and it rotates about 1°
   of 30°; the particle table absorbs the turn row by row.
2. **KA2's anchor holds the critic in place.** After KA2's warm-up, the critic penalty pulls the critic toward its
   averaged (EMA) copy. In a recipe without a data-drift signal, nothing can release that pull, and the turn freezes
   the EMA copy, so after a turn the critic stays pinned to the old target. The first turn in the test lands in the
   warm-up and escapes this; the second does not, and the generator rotates backwards.

## What R1 adds

- **`reopen_signal: "optimizer"`** (`particlegan.continuous.OptimizerSurprise`). After each optimizer step, every
  param group reports q = mean |g| / sqrt(v̂): the gradient size in units of Adam's own memory. A fast average of
  log q is compared with a slow level that is frozen while the game is surprised. When the geometric mean of fast /
  slow over groups stays above 2 for 12 consecutive updates, it fires once: every stationarity ladder re-opens
  (`restart(reopen=True)`) and each group's second moment (and AMSGrad maximum) is divided by that group's ratio
  squared, so its steps grow by the ratio. Only an abrupt rise counts: the ratio must cross 2 within 24 updates
  (2K) of its last calm update (below 1.25). A slow ramp is the game's own evolution: on the harness's
  vector_overlap gate, KA2 leaving its warm-up lifts the ratio to 2.5 over ~80 updates, and a fire there failed the
  gate (18/24; 20/24 without it). Real target shifts cross within 2-13 updates. After a fire the detector waits one
  slow horizon (96 updates, 8K) and for calm before it re-arms, so its own re-open cannot trigger the next one.
- **`reopen_anchor: "release"`** (requires `reopen_signal: "optimizer"`). A fire counts as drift evidence for KA2
  until KA2's own surprise ratio has gone above 3 and back below 1.75 (KA2's existing release levels). While it
  holds, KA2's native rules release the anchor, let the EMA critic track and reseed it. On the native host the
  release lasts ~180 updates.

Both read only network and optimizer state: no statistic of the data or of generator outputs, no step schedule.
The state is part of the trainer checkpoint and resumes bit-exactly (`tests/test_r1_rotation.py`).

## Routed (`e22_routed`)

The same two fields work unchanged with the conditional dense-bank recipe:

```python
recipe = get_recipe("e22_routed", reopen_signal="optimizer", reopen_anchor="release", ...)
```

[`examples/e22_routed_moving.py`](../examples/e22_routed_moving.py) runs the
[routed paired-edit example](e22_routed.md) with a moving target: every `--turn-every` updates the paired edit
(target minus the frozen host) turns 30° on the fitting, guard and held-out contexts (`--r1` adds the two fields).
Held-out clean RMSE at the end of each period, seed fixed by the example:

| turn every | recipe | before turn 1 | after turn 1 | after turn 2 | 50 updates after turn 2 |
|---|---|---|---|---|---|
| 500 | `e22_routed` | .0019 | .0023 | .0033 | .0821 |
| 500 | + R1 fields | .0019 | .0019 | .0012 | .0022 |
| 250 | `e22_routed` | .0021 | .0023 | .0100 | .0574 |
| 250 | + R1 fields | .0021 | .0015 | .0015 | .0056 |

`e22_routed` never re-opens: each turn costs it about 300–500 updates at 10–40× its settled error. R1 fires within
50 updates of every turn, including turns 250 updates apart: this small task calms down between turns, so the
detector re-arms. Without turns (3,000 updates) the two recipes are bit-identical, with no fires.

# R1: following a moving target

`configs/100gaussians/r1-rotation.json` is the [E22 configuration](e22.md) with two added fields:

```json
"reopen_signal": "optimizer",
"reopen_anchor": "release"
```

Both default to off (`"none"` / `"hold"`), so E22 and every other recipe are unchanged.

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
  squared, so its steps grow by the ratio. The detector re-arms when the ratio falls below 1.25.
- **`reopen_anchor: "release"`** (requires `reopen_signal: "optimizer"`). A fire counts as drift evidence for KA2
  until KA2's own surprise ratio has gone above 3 and back below 1.75 (KA2's existing release levels). While it
  holds, KA2's native rules release the anchor, let the EMA critic track and reseed it. On the native host the
  release lasts ~180 updates.

Both read only network and optimizer state: no statistic of the data or of generator outputs, no step schedule.
The state is part of the trainer checkpoint and resumes bit-exactly (`tests/test_r1_rotation.py`).

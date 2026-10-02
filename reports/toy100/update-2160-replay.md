# Read-only replay of fence-restore update 2160

No rule change. Same host as the hold probe: scheduled prefix through update 1000, then constant rates `.00425 / .00425 / .0085` and the fence-restore clip. torch 2.14.0+cpu, one thread, AVX2. The unpinned AVX512 dispatch does not reach this state (6 modes at update 1000).

Graded checkpoints around the miss:

| Update | Modes | HQ | Notes |
| ---: | ---: | ---: | --- |
| 1000 | 8 | .997 | warm prefix |
| 1200 | 8 | .999 | warm window still up |
| 2140 | 8 | 1.000 | |
| 2150 | 8 | .926 | |
| 2160 | 8 | .891 | the graded miss |
| 2170 | 8 | .996 | recovered after 2161 restores |

Eval counts at 2160 are `[66, 314, 653, 663, 719, 698, 369, 167]` out of 4096. Every mode still has HQ mass. Mode 0 falls 341 → 66 and mode 7 falls 339 → 167. That is the whole HQ drop (about 447 draws).

## Update 2160, proposal versus applied

Rho `.672`, curvature factor `.372`. The exit clip does not fire (`clipped=0`, every particle scale 1). Applied output motion is the curvature-scaled proposal. Median output step `.016`, max `.047`.

The minibatch fence on this step is `.111`, not the usual `.08`. HQ radius is `.21` (`3 × 0.07`).

Particle 9 is the one that leaves the HQ ball:

| | Nearest real | Margin to fence | Distance to mode center |
| --- | ---: | ---: | ---: |
| Before | .075 | .036 | .186 |
| Proposed = applied | .094 | .017 | .233 |

Nearest-real distance stays under the fence, so the clip correctly does nothing. The same step is radially outward and finishes `.023` past the HQ radius. Particle 10 moves `.173 → .210`, still just inside the ball, with nearest-real `.031 → .064`, also under the fence. Its eval mass is what cuts mode 7 roughly in half. The other ten support points stay well inside.

2159 does clip (particle 10, scale `.385`, fence `.099`). 2161 clips both outer particles back onto a `.096` fence (scales 0) and the graded HQ is back to `.996` by 2170. The miss is one step whose fence is wide enough that an outward move finishes inside it and outside the HQ ball.

## Why the fence cannot see this

A particle within `.094` of a real sample is inside a `.111` fence. That real sits near the edge of its mode, so the particle, further out on the same ray, is `.233` from the mode center. Nearest-real margin and HQ radius are different lengths. A positive margin does not keep the particle in the HQ ball when the fence is inflated by this minibatch's neighbor spacing.

Script: `reports/toy100/replay_update_2160.py`.

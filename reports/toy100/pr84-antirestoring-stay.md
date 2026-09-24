# Mild anti-restoring G step — kill

Host: neural. Torch 2.14.0+cu130, one CPU thread. Draft only. No merge.

## Mechanism
When PR84 would accept a full generator step (`rho ≤ 0.25`) and the signed own-curvature alignment along that step is negative, shrink it to `max(0.5, 1+alignment)`. Restoring steps and already-bounded steps are unchanged.

## Purity
GAN dynamics only — no coverage/likelihood term. No target centers, HQ clip, rest-damping retune, or barrier probe.

## Same-process PR84 pin
Warm fork at update 1000, constant rates `.00425/.00425/.0085`, stay checks every 10 updates through 1600. Cold trajectory 400 and cold ring 1200 use constant LR (`lr_floor=1`).

| Gate | PR84 pin | Mild stay | Call |
| --- | --- | --- | --- |
| Warm 1001–1200, 8 modes and HQ ≥ .9 | 196/200, min HQ .866, modes 8 | 200/200, min HQ .932 | warm not worse |
| Stay 1210–1600 (40 checks) | **40/40**, min HQ .920, modes 8 | 37/40, min HQ .827, dips at 1300, 1500, 1590 | **worse stay** |
| Cold trajectory | PASS (2.82 s) | PASS (2.84 s), mild updates 0 | no regression |
| Cold ring live | **8 / HQ .9988** (28.8 s) | **8 / HQ .9988** (29.0 s), mild updates 0 | ring not killed |

The shrink fired 101 times on the warm continuation and **zero** times on cold trajectory and cold ring, so acquisition matches the pin because the rule was idle there. It does not buy the warm window by freezing an incomplete cloud: both cold rings are 8 modes. It also does not improve stay. The pin itself holds 8 modes through 1600; the extra shrink introduces HQ dips.

## Rank
On the GAN-native track this does not beat the PR84 pin. Stay is worse. Acquire is the same only because the mechanism never turned on.

## Keep / kill / next
**Kill.** Do not retune the floor. Next bet, not run here: a game signal that stays idle on this continuation unless a mode is actually leaving. Signed step-curvature is the wrong idle condition — it fires through a hold the pin already keeps.

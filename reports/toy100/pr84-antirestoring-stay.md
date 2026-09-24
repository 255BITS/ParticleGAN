# Mild anti-restoring G step — idle-condition follow-up, kill

Host: neural. Torch 2.14.0+cu130, one CPU thread. Draft only. No merge.

## Mechanism
Same shrink as PR #104: a full generator step (`rho ≤ 0.25`) with negative signed alignment is scaled to `max(0.5, 1+alignment)`. The floor is unchanged. The shrink now stays idle unless the base-point critic advantage is also negative (discriminator loss above log 2).

## What was checked and discarded
On the PR84 pin, the only warm HQ break (updates 1129–1132, still 8 modes) has **positive** alignment (0.59–0.82). `max(0.5, 1+alignment)` cannot shrink those steps. A particle-clump leaving fraction, using other generated particles rather than target centers, is 0.42 at the break versus a median of 0.5, so it does not mark the leave. Negative alignment is common on the healthy continuation (86 of 400 full steps) and rare before the break (3 of 200). Gating on the generator-network group does not separate those sets: every joint-negative full step is also network-negative.

## Purity
GAN dynamics only — no coverage/likelihood term. No target centers, HQ clip, rest-damping retune, or barrier probe.

## Gate table vs this harness's PR84 pin

| Gate | PR84 pin | Critic-losing idle | Ungated #104 |
| --- | --- | --- | --- |
| Warm 1001–1200, 8 modes and HQ ≥ .9 | 196/200, min HQ .866 | 196/200, min HQ .866 | 200/200 |
| Stay 1210–1600 (40 checks) | **40/40**, min HQ .920 | 38/40, min HQ .710, dips at 1570 and 1580, back to 8/HQ 1 at 1600 | 37/40, min HQ .827 |
| Cold trajectory | PASS | PASS, shrink updates 0 | PASS, 0 |
| Cold ring live | **8 / HQ .9988** | **8 / HQ .9988**, shrink updates 0 | **8 / HQ .9988**, 0 |

The new gate fired 4 times, all after update 1200 (1294, 1449, 1533, 1557 on the pin path). Warm matches the pin because it never fired there. The ring matches because it never fired there either. The four fires still open an HQ dip the pin does not have. Modes stay 8 and the last check recovers, which is a softer miss than #104, not a better hold.

## Rank
On the GAN-native track this does not beat the PR84 pin. Stay is still worse. Acquire is unchanged only because the rule was idle on the cold run.

## Keep / kill / next
**Kill.** Do not retune the floor and do not add another predicate on `alignment < 0`. Next bet, not run: the verified leave is a fully accepted step with **positive** alignment and a collapsing critic advantage. This shrink shape is a no-op on that sign, so a further idle tweak cannot engage it.

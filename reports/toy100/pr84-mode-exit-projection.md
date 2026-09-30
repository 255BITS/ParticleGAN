# Mode-exit projection on PR84 G step — kill

Host: neural. Torch 2.14.0+cu130, one CPU thread. Draft only. No merge.

## Mechanism
After the PR84 curvature-bounded G step, evaluate the directional derivative
of D along the G step for particles in high-D regions (above the per-batch
median D value at their base positions). If the mean directional derivative
for these high-D particles is negative (the step moves them to lower D,
i.e. exiting the mode), shrink the G parameter step by
`max(floor, 1 + mean_dd / max(|mean_dd|, dx_norm * 0.01))`.

The projection only activates when: (1) the PR84 curvature bound accepted
the full step (rho <= 0.25, factor = 1.0), AND (2) the mean DD for high-D
particles is negative. Steps already bounded by PR84 curvature, or steps
where high-D particles are not exiting, stay unchanged.

## Purity
GAN dynamics only — uses D and its input gradient (directional derivative
along the G step). No target centers, coverage loss, HQ ball clip, bank
screens, or forward-KL term.

## Same-process PR84 pin
Warm fork at update 1000, constant rates, stay checks every 10 updates
through 1600. Cold trajectory 400 and cold ring 1200 use constant LR
(`lr_floor=1`).

Note: this process's warm state at step 1000 has only 6 modes (not 8 as
in PR84's original report). The pin itself never reaches 8 modes in this
process. Cold ring reaches 7 modes for both pin and projected.

## Gate table vs same-process PR84 pin

| Gate | PR84 pin | Projected | Call |
| --- | --- | --- | --- |
| Warm 1001–1200, 8 modes / HQ >= .9 | 0/200, min modes 6, min HQ .935 | 0/200, min modes 6, min HQ .935 | same (both miss 8 modes) |
| Stay 1210–1600 (40 checks) | 0/40, min modes 5, min HQ .847 | 0/40, min modes 5, min HQ .843 | same |
| Cold trajectory | PASS (3.07 s) | PASS (3.04 s), projected 0 | no regression |
| Cold ring live | 7 / HQ .993 (24.9 s) | 7 / HQ .993 (25.1 s), projected 0 | same |

The projection fired **6 times** in the warm continuation and **zero times**
during cold trajectory and cold ring. Acquisition matches because the rule
was idle during cold, not because it helped.

## Why the mechanism doesn't help

1. **Firing rate is too low.** Only 6 out of ~600 warm G steps triggered the
   projection (those where PR84 accepted the full step AND the high-D
   particle DD was negative). The curvature bound `rho <= 0.25` is rarely
   satisfied simultaneously with a mode-exit signal.

2. **The warm state already has only 6 modes.** There are no acquired modes
   to protect that the projection could save. The mechanism can only shrink
   steps that exit high-D regions, but with only 6 modes the generator is
   still in acquisition mode, not hold mode.

3. **Cold acquisition is unaffected.** During cold ring training, the
   curvature bound is almost always active (rho > 0.25), so the projection
   condition `pr84_factor >= 1.0` is never met. The mechanism is inherently
   idle during acquisition.

## Rank
On the GAN-native track this does not beat the PR84 pin. Warm and stay
are identical. Cold ring is identical. The mechanism fires too rarely
to produce meaningful signal.

## Keep / kill / next
**Kill.** The mode-exit projection fires too rarely because the curvature
bound and the DD condition are nearly mutually exclusive during active
training. When rho is low enough for the full step to be accepted, the
generator step is small enough that it doesn't exit modes; when the step
is large enough to exit modes, the curvature bound already clips it.

Next bet (not run here): instead of conditioning on the curvature bound
accepting the full step, apply the DD projection as a post-hoc correction
to the *already-bounded* step. The bounded step still has a direction, and
that direction might still exit modes even at reduced magnitude.

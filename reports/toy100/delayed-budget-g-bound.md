# Delayed-budget G own-curvature clamp .125

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Ranking pin:
`ATEN_CPU_CAPABILITY=avx2` with `MKL_ENABLE_INSTRUCTIONS=AVX2`,
`ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. CPU capability is on every
receipt. AVX512 is a build check, not a second seed. Warm rank is refused if
the identity fork is not 200/200. Receipt: [continuous-evidence/delayed-g125/summary.json](continuous-evidence/delayed-g125/summary.json).

## Mechanism

#107 stall reach is unchanged below the budget: width rule, D, losses, and G's
own-curvature bound **.25**. On the mode-hold host the update index is the
checkpoint number of the update in progress. When that index is **≥ 1200** the
clamp latches and every later G step uses own-curvature bound **.125**. It does
not read modes or HQ. It does not scale Adam. Each apply logs `armed`,
`g_bound_before`, `g_bound_after`, `update_index`, and the running fire count.

Stay receipt: 1199 pre-budget applies at .25/.25, then 1201 fires at .25 → .125
(updates 1200 through 2400). Trajectory never reaches the budget (fires 0).

## Why

#138 armed .125 on the first 8/.9 graze (~650). The next ~550 G steps killed
the ring. Always-on .125 did the same. This clamp waits until update 1200, after
that acquire window, and only then limits the trust ball that opens during the
1720–2150 dropout (G's own-curvature ratio falls, so the .25 cap lets the step
grow).

## Gates

| Gate | PR84 pin | #107 stall reach | **Delayed G .125** |
| --- | --- | --- | --- |
| Warm AVX2, identity | — | 200/200 | **200/200**, min HQ .990 |
| Warm AVX2, method | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921 |
| Cold trajectory AVX2 | PASS | PASS | **PASS** (fires 0) |
| Cold ring AVX2 | 8, suffix 5 | 8, terminal HQ 1.0 ×5, suffix 8 | **PASS**, 10/24, suffix 8, terminal 8 at HQ .999–1.0 |
| Cold ring AVX512 (build check) | 7 | 8, suffix 8 | **PASS**, 8/24, suffix 8 (not ranked) |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, min 0 modes | **111/120**, final **8 / 1.0**, min **5** modes, suffix 5 |

Failing stay checks: 1560, 1580, 1610, 1820 (all still 8 modes, HQ .84–.90),
then 2280–2350 (8 → 7 → 5 → 6 modes, min HQ .505). No check falls to ≤4 modes.
The 1720–2150 collapse is one HQ graze (1820, 8 modes, HQ .873). The cloud is
back on 8 modes at HQ 1.0 from 2360 through 2400.

Post-budget G trust factor max is .190 (mean .040). Pre-budget max is .578,
and 159 pre-budget steps exceed .2. The old dropout's trust opening (factor
~ .27) does not recur.

## Verdict

**Keep. Not solved.**

Acquire holds: warm matches #107, and both cold rings pass. Stay is better than
#107 (111 vs 97, floor 5 modes vs 0, final HQ 1.0 vs .971) and better than PR84.
It does not clear. Nine checks still fail, and a new 5-mode dip sits at
2310–2320, after #107's clean tail. This is not a 22/22 result and not a
production claim.

## Next bet

One mechanism, not a retune of .125 and not the in-flight Adam lr×0.5.

After the clamp, G's trust factor sits near .04, so #107's stall test (mean
factor ≤ .1) trips whenever D's slope is high. Critic width .5 rises from 11%
in the solved 800–1199 hold to 53% in 2100–2279, just before the residual dip.
**Next:** once update ≥ 1200, score that stall test on the factor the .25 cap
would have produced, so a clamp-shortened step is not itself a stall. Leave the
.125 ball, D, and the losses alone. No mode-count arm.

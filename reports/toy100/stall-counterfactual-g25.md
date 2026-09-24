# Stall scored on the .25-cap factor after the .125 latch

Host: #141 delayed-budget G own-curvature .125. Seed 0, one thread, PyTorch
2.14.0+cu130 with the CPU capability pinned to AVX2
(`ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`,
`ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`). CPU capability is on every
receipt. Warm rank is refused if the identity fork is not 200/200 on AVX2.
Receipt: [continuous-evidence/stall-cf25/summary.json](continuous-evidence/stall-cf25/summary.json).

## Mechanism

#141 still applies. Pre-budget G own-curvature bound **.25**. At mode-hold
update index **≥ 1200** the bound latches at **.125** and stays there. The
accepted step uses that .125 factor.

Separately, once the latch is on, the same rho is turned into the factor a
**.25** cap would have produced, `min(1, .25 / rho)`. #107's stall predicate
(D slope ≥ .6 and mean trust ≤ .1 over 50 updates) reads that counterfactual
factor. Width .5, D, Adam, and the losses are unchanged. Pre-budget rows do
not carry a stall score, so the predicate keeps the actual factor. Trajectory
never reaches the budget.

Each armed apply logs `stall_factor`, `stall_cap` .25, and `applied_factor`
next to `g_bound_after` .125, so `tail -f` shows the two factors on one line.

## Why this was the bet

#141's post-budget trust factor sits near .04, so the stall test trips
whenever D's slope is high. Critic width .5 rose from 11% in the solved
800–1199 hold to 53% in 2100–2279, just before the residual dip. The bet was
that those stalls were the clamp talking, and that scoring the .25-cap factor
would leave the .125 ball alone while stopping a shortened step from counting
as a stall.

## Gates

| Gate | PR84 pin | #107 stall reach | #141 delayed G .125 | **Stall on .25-cap factor** |
| --- | --- | --- | --- | --- |
| Warm AVX2, identity | — | 200/200 | 200/200, min HQ .990 | **200/200**, min HQ .990 |
| Warm AVX2, method | 196/200, min HQ .866 | 200/200, min HQ .921 | 200/200, min HQ .921 | **200/200**, min HQ .921 |
| Cold trajectory AVX2 | PASS | PASS | PASS (fires 0) | **PASS** (fires 0) |
| Cold ring AVX2 | 8, suffix 5 | 8, suffix 8 | PASS, 10/24, suffix 8 | **PASS**, 10/24, suffix 8, terminal HQ .999–1.0 |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, min 0 | **111/120**, final 8 / 1.0, min 5 | **108/120**, final 8 / .825, min **4**, suffix 0 |

Acquire matches #141. Stay does not improve on #141 (111/120) or on board #140
(114/120).

## What the run did

Stay fires: 1199 pre-budget applies at .25/.25, then 1201 fires at .25 → .125.
Post-budget applied factor max is .151 (mean .040). The counterfactual factor
max is .302 (mean .080). On 227 post-budget steps the applied factor is ≤ .1
while the scored factor is > .1.

That split does not touch the window that motivated the bet. Width .5 is
10.5% in 800–1199 and **53.3%** in 2100–2279. The 50-update mean of the .25-cap
factor there is .071, still under the .1 stall line, so the predicate stays
on. The same four early misses as #141 remain: 1560, 1580, 1610, 1820
(8 modes, HQ .838–.896; 1820 is HQ .873).

The predicate first disagrees at update **2296** (applied 50-mean .050, scored
50-mean .101). From there the scored factor stays above .1 through 2400, width
.5 falls to 5.8% of 2280–2400, and the tail does not come back. Failing checks:
2280, 2290, 2310, 2330 (4 modes, HQ .402), 2350, 2360, 2390, 2400. Final is
8 modes at HQ .825. Suffix 0. One check falls to ≤4 modes. #141's tail was
8 modes at HQ 1.0 from 2360 through 2400, with a floor of 5.

## Verdict

**Kill.**

Warm and both cold acquires hold, so the pre-budget path is #141. Stay is
worse on passing checks (108 vs 111), on the mode floor (4 vs 5), and on the
final HQ (.825 vs 1.0). The .25-cap score does not disarm the pre-dip width
climb, and it turns the stall off inside the dip, which drops the recovery
#141 had. This is not a 22/22 result and not a production claim.

## Leaderboard (stay 1210–2400)

| Line | Stay | Final | Floor |
| --- | --- | --- | --- |
| #140 delayed-arm G Adam lr×0.5 (board, other fork) | 114/120 | — | — |
| #141 delayed G .125 (this host) | 111/120 | 8 / 1.0 | 5 |
| **This bet** | **108/120** | **8 / .825** | **4** |
| #107 stall reach | 97/120 | 8 / .971 | 0 |
| PR84 pin | 53/120 | 6 / .904 | — |

## Recommendation

Leave the .125 ball, the 1200 latch, and stall width .5 where #141 set them.
Do not retune those three. This scoring change is not the next host.

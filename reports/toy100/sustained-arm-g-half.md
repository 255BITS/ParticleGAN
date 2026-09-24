# Sustained-arm G×0.5 on stall reach — KILL

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Pin `ATEN_CPU_CAPABILITY=avx2` (logged capability AVX2). Identity warm is 200/200, min HQ .990, so the warm result is ranked. AVX512 is a build check. Draft only. Not solved. Not production.

Receipts: [continuous-evidence/sustained-arm-g-half](continuous-evidence/sustained-arm-g-half/).

## Mechanism (one)

Stall reach is unchanged until a fixed 4096-draw through the unwrapped generator reports **8 modes and HQ ≥ 0.9 for 200 consecutive updates**. That draw is the #129 ring metric. It is not a loss. The counter resets on any miss. The first time it hits 200, the run arms, and every later G+prior Adam displacement is ×0.5 (scaled after Adam computes the step; moments stay on the full gradient). D, the stall width, both curvature bounds, and the D step count stay as in #107.

Not a falling-trust detector, not a first-touch arm, not a mode-count freeze.

## Gates

| Gate | PR84 pin | #107 stall reach | Sustained-arm G×0.5 |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921. Armed at 1199. **1 half-step** |
| Cold trajectory | PASS | PASS | **PASS**, identity MSE .00094. Never armed |
| Cold ring (AVX2) | 8, suffix 5 | 8, suffix 8, terminal HQ ~1.0 | **PASS 10/24, suffix 8**, final 8 / .999. Never armed. max hold 185. Observations match #107 |
| Cold ring (AVX512) | 7 | 8, suffix 8 | **PASS 8/24, suffix 8**, final 8 / .998. Never armed. max hold 155. Observations match #107 |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, dips to 0 | **97/120**, suffix 25, final 8 / .971, min 0 modes. **0 half-steps**. Same 23 failing steps as #107 |

## Why

Warm can arm, because the scheduled prefix is already a full ring and the 200 continuation checks stay at 8 / HQ ≥ 0.9. That single last-step half-step does not move the hold.

From scratch the counter never reaches 200. The longest unwrapped streak in 2400 updates is **185**, reset at update 1057 by one frame of 7 modes at HQ .923. Later streaks die the same way (135 at 1718, 125 at 1465, 117 at 2084). The official 50-step checks still read a held ring; the per-update draw flickers to 7 modes inside those gaps. Acquisition is therefore #107, and so is the 1720–2150 dropout. The 1755 fork's G×0.5 effect is not reached.

## Call

**KILL.** The 200 and the 0.5 were not retuned. **#107 stall reach remains #1.**

## Line

| Entry | Acquire | Stay | Call |
| --- | --- | --- | --- |
| #107 stall reach | ring 8 on AVX2 and AVX512, warm 200/200 | 97/120, ends on 8, dips to 0 | board #1 |
| Sustained-arm G×0.5 | identical to #107 (never armed) | identical 97/120, 0 half-steps | **KILL** |
| #125 first-touch G×0.5 | cold ring FAIL (armed at 650) | not run | KILL |
| #129 mode-drop freeze | AVX2 ring FAIL (latched at 569) | 0/120 | KILL |
| #122 falling-trust G×0.5 | cold ring never 8 | — | KILL |

Do not shorten the hold. On this draw a streak of 200 does not occur before, or through, the dropout, so a lower threshold would arm on the same flicker that killed #125.

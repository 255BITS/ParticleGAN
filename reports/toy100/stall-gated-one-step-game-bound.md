# Stall-gated one-step game bound: KILL

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. `ATEN_CPU_CAPABILITY=avx2` (logged CPU capability: AVX2). Draft only. This is not 22/22 and it is not production-ready.

Receipts: [continuous-evidence/stall-gated-game](continuous-evidence/stall-gated-game/).
Runner: [gan_followup_probe.py](gan_followup_probe.py) method `reachstall_sgame`.
Candidate: [pr84_reach_candidate.py](pr84_reach_candidate.py) (`stall_game=True`).

## Mechanism

#107 stall reach widens G's critic stencil when D's slope is at least .6 and the mean of G's trust factor over the last 50 updates is at most .1. This bet uses that same predicate on the G step only:

- Stall true: one virtual D Adam step answers G's proposal, and G's trust factor uses `max(ρ_own, ρ_game)`. D's parameters and Adam state are restored.
- Stall false: G is placed from its own curvature, the #107 update. No extra replay.

Two-step is off. Stall-reach width, both curvature bounds, the losses, and the D step are unchanged. No second gate, no coefficient, no coverage, anchor, likelihood, or clip term.

## Gates

| Gate | PR84 pin | #107 stall reach | This bet |
| --- | --- | --- | --- |
| Warm 1001–1200 | 196/200 | 200/200 on valid AVX2 | **not ranked** |
| Cold trajectory, MSE ≤ .02 | PASS | PASS | **PASS**, MSE .00094 (gate fired 0/400) |
| Cold ring, AVX2 | 8 | **PASS** here: 8 / HQ .998, 8/24, suffix 8 | **FAIL**: modes 8, HQ .701, 1/24, suffix 0 |
| Continued stay 1210–2400 | 53/120, final 6 | 97/120, final 8, 0-mode dips ~1770/1820 | not run |

Warm identity on this process is 0/200, minimum 6 modes, minimum HQ .953, with 6 modes already at update 1000. That is the unsolved 6-mode prefix. Warm ranking is refused. The method fork was 0/200 as well (minimum 5 modes, final 6 / 1.0) and is not a score.

The cold-ring control is the same command with `reachstall` (stall reach, game bound off) on this process. It acquires the ring. The gated bound does not.

## Why it fails

The predicate is true while the ring is still being acquired. Of 1200 constant-rate updates the virtual D step ran on 506, including 182 of the first 200. Updates 400–800 did not fire; updates 800–1200 fired on 182 of 401, which is the window where #107 is already on 8 modes and this run is still at 6. By step 1000 #107 is at 8 / .992 and this run is at 6 / .518. Step 1200 is still 8 modes but HQ .701, so the checked ring fails.

That is the same acquisition cost as the always-on one-step bound, reached through the stall predicate rather than an always-on switch. The predicate does not separate the late dropout from early acquisition.

## Call

**KILL.** No coefficient change, no second gate, no stay run after this counterexample.

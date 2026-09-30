# Lane A kill: anti-gradient critic field on PR84

Host: neural. Seed 0. One CPU thread. PyTorch 2.14.0+cpu (this runner). Codex’s practical board pinned PR84 cold ring at seven modes on PyTorch 2.13. This note does not replace that pin.

## Hypothesis

At PR84’s archived seven-mode state the missing mode has the highest critic, but the local stencil gradient points away and only flips farther along that path. G kept the five-point stencil unless a sharp critic read at two or four stencil widths opposite that gradient was strictly higher. The offset was detached. D, both curvature bounds, and the game objective were unchanged. No coverage, Chamfer, mode-count, assignment, or clip ladder.

## Same-harness PR84 baseline (this build)

| Gate | Result |
| --- | --- |
| Cold trajectory | PASS, identity MSE `.000942668`, suffix 18 |
| Cold ring | Terminal checks 1000–1200 are **8 modes**, HQ `.9951 / .9878 / .9880 / .9956 / .9988`. Verdict PASS (10/24 observations; step 950 is 8 modes at HQ `.862`). Wall 28.2 s |
| Warm 1001–1200 | **196/200**. Failures 1129–1132, min 8 modes, min HQ `.866`, final 8 / HQ `.999`. Identity fork 200/200 |

Terminal ring 8 on 2.14 is the build sensitivity the board already records. It is not a claim that Codex’s 2.13 seven-mode pin is wrong.

## Barrier field

| Gate | Result vs baseline |
| --- | --- |
| Warm 1001–1200 | **47/200**. Min **6 modes**, min HQ `.494`, final **6 / HQ `.654`**. Identity fork still 200/200 |
| Cold trajectory | Not run |
| Cold ring | Not run |

Warm regresses (196/200 and final 8, versus 47/200 and final 6). The family stops here. The warm receipt’s proposal-phase `barrier_uses` total is 0, so the collapse is not a count of accepted anti-gradient replacements on that phase; the curvature replay still evaluates the same field. No scale retune.

## Rank

No rank claim over board #1 or #2. The candidate does not acquire, and it does not stay on the learned solution.

## Keep / kill / next bet

**Kill** this probe family. Do not retune the 2×/4× widths. Next bet is a different critic-response channel that is idle on an already covered warm state; this one is not that channel.

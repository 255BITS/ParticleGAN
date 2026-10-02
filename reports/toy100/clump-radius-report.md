# Clump HQ ball on the fence-restore clip

Starts from the PR #93 fence restore. One added rule, from the read-only replay of update 2160 (`reports/toy100/update-2160-replay.md`). No coefficient sweep, no rest-slope damp, no mode catalog.

## Why this rule

Update 2160 does not cross the minibatch nearest-neighbor fence. Rho `.672`, curvature factor `.372`, exit-clip scale 1 on every particle, so the applied step is the capped proposal. The fence on that minibatch is `.111`. Particle 9 moves nearest-real distance `.075 → .094` (still inside the fence) and mode-center distance `.186 → .233`, past the HQ radius `.21`. Particle 10 lands at `.210`. Eval HQ falls to `.891` with all 8 modes still up: mode 0 loses most of its mass and mode 7 about half. Updates 2159 and 2161 do clip, and HQ is `.996` again at 2170.

The spacing fence is not the HQ ball. A particle can sit just outside an outer sample and still be inside a wide fence. The added rule leaves every existing fence restore unchanged. A particle that rule does not touch, whose step would finish outside the HQ ball of its minibatch clump, is shortened onto that ball. The center is the mean of reals linked by the existing fence. The radius is the graded HQ radius, `3 × 0.07`.

## Same-machine gates

torch 2.14.0+cpu, one thread, AVX2. Unpinned AVX512 does not reproduce the 8-mode prefix. Constant rates `.00425 / .00425 / .0085` after update 1000 on the warm fork.

| Rule | Warm | Min HQ | Hold to 2400 | Worst hold |
| --- | ---: | ---: | ---: | --- |
| Scheduled identity | 200/200 | .990 | 120/120 | 8 / .998 |
| PR84 (#84) | 196/200 | .866 | 114/120 | 7 / .829 |
| Nearest-real exit clip (#91) | 200/200 | .939 | 117/120 | 8 / .808 |
| Open-cap rho tighten (#91) | 200/200 | .959 | 115/120 | 8 / .784 |
| Fence restore (#93) | 200/200 | .965 | 119/120 | 8 / .891 |
| Clump HQ ball | 200/200 | .965 | **120/120** | 8 / .911 |

Hold failures for this rule: none. Minimum hold HQ `.911` at 8 modes.

## Cold

Trajectory passes (identity MSE `.00094`, 18/24 checks, confirmed at 184). Cold ring fails: 6 modes, HQ `.790`, zero passing checks. The clip fires on 1195 of 1200 ring updates, so the same ball that stops the late outward step also stops acquisition. Own-state hold was not run.

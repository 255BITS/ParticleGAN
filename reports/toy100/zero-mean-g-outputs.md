# Zero-mean generator outputs on stall reach — KILL

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Rankable pin:
`ATEN_CPU_CAPABILITY=avx2` (receipt CPU capability AVX2). AVX512 was not run.
Draft only. Not solved. Not a 22/22 claim and not production-ready.

Receipts: [continuous-evidence/zero-mean-g-outputs/](continuous-evidence/zero-mean-g-outputs/).
Runner: [gan_followup_probe.py](gan_followup_probe.py) `--method zeromean`.
Candidate: [pr84_zero_mean_outputs.py](pr84_zero_mean_outputs.py) on #107 stall reach.

## Mechanism

One change. After every train-step `fake = G(...)` that feeds D or G, including the
particle cloud on the trajectory host:

`fake = fake - fake.mean(dim=0, keepdim=True)`

Evaluation forwards are not centered. The disabled identity fork does not center
(`passthrough` / `enabled` is a no-op and returns the same tensor). Stall-reach
width, curvature bounds, D, losses, Adam, and the learning-rate schedule are #107.
No coverage, anchor, quota, Chamfer, forward-KL, or clip term.

On the warm continuation the receipt recorded `centered_batches = 1200`
(200 updates × 3 replays × 2 train forwards). The mechanism ran.

## Why this was the bet

#107's shared dropout (~1720–2150) is G-led whole-cloud translation; D's slope
spikes after the cloud has already moved. Translation is 56–83% of G's output
motion in every phase, so a fraction gate cannot separate healthy acquire from
dropout. The 8-mode ring is origin-symmetric, so a hard zero mean deletes the
translating mode and leaves relative arrangement free.

## Gates

PR84 and #107 numbers are the published AVX2 pin from
[pr84-reach-followup.md](pr84-reach-followup.md). This run's identity fork is the
unchanged control on the same harness.

| Gate | PR84 pin | #107 stall reach | **Zero-mean train forward** |
| --- | --- | --- | --- |
| Warm AVX2 identity control | — | 200/200 on this pin | **200/200**, min HQ .990 (harness valid) |
| Warm AVX2 method, 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **5/200**, min modes 0, min HQ 0, final 1 / .0002 |
| Cold trajectory | PASS | PASS | not run (warm fail-fast) |
| Cold ring AVX2 | 8 | 8 | not run |
| Cold ring AVX512 (build check only) | 7 | 8 | not run |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, hard 0-mode dips | not run |

Warm timeline (raw cloud, uncentered eval): 1001–1005 hold 8 modes (HQ .95–.99),
1006 is 6 / .61, 1013 is 2, 1050 is 1 / .12, 1150 is 0 / 0, 1200 is 1 / .0002.
Every continuation update widened the stall-reach stencil (`widened_updates = 200`,
max width .50). On a covered cloud #107 stays at width .15.

## Keep / kill

**KILL.** Warm regresses past both the PR84 floor (~196/200) and the #107 bar
(200/200). The identity control is 200/200, so the fork is rankable. No cold run,
no stay run, no coefficient, no EMA mean, no soft aux loss.

Hard centering does not hold a solved ring. It removes the batch mean from the
tensor D and G train on, so absolute translation of the raw cloud — the cloud the
gate scores — is no longer a direction D can punish. From the solved warm state
that null direction leaves the mode balls within five updates and does not return.
Opening the critic read to width .5 on all 200 updates is the same signature #107
sees when D is no longer looking at a covered cloud.

## Next single bet

Do not retune train-forward centering. The mean has to stay visible to D on the
cloud the gate scores.

Next bet, one mechanism: after the ordinary raw-forward D/G step, cancel only the
realized common translation by a closed-form output-bias correction that restores
the pre-step cloud mean. D and G still train on uncentered outputs. No EMA, no
aux loss, no clip ladder.

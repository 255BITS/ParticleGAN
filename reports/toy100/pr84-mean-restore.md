# Post-step G output-mean restore — KILL

Host: stall reach from #107 (PR #84 smoothed critic, stall width, reach .5, G bound .25, D bound 3). Seed 0, one thread, PyTorch 2.14.0+cu130. Pin: `ATEN_CPU_CAPABILITY=avx2` (logged capability AVX2). AVX512 warm starts are not ranked. Not a 22/22 result and not a production candidate.

## Mechanism

One change. The G Adam step and the stall-reach curvature bound run on the usual uncentered forward, the one D and the loss already saw. After that accepted step, G's existing final-linear output bias is shifted so the clean mean of that same training batch matches the pre-step mean. The train forward is not centered. Loss inputs are not zero-meaned. No falling-trust G×0.5, no extra D step, no always-on G×0.5, no clip.

This is the opposite of the #124 kill, which zero-meaned G outputs inside the train forward before D saw them.

## Gates (AVX2)

Identity warm fork on this harness: **200/200**, min HQ .990, final HQ .999. That is the valid control.

| Gate | PR84 pin | #107 stall reach | This bet |
| --- | --- | --- | --- |
| Warm continued 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .915, final HQ 1.0 |
| Cold trajectory | PASS | PASS | **FAIL**, identity MSE .082 (threshold .02), 0/24 |
| Cold ring (AVX2) | 8, terminal HQ .988–.999 | 8, terminal HQ 1.0 ×5, suffix 8 | **FAIL**, terminal 0 modes / HQ 0. Best 50-step check: 4 modes at 1100 (HQ .208). Never 8 |
| Cold ring (AVX512) | 7 | 8 | not run; not ranked |
| Stay / continued hold | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25; dips then recovery | **0/43** checks from 1210–1630 (max 4 modes, min 0). Run stopped at update 1638 |
| Mean-restore fires | — | — | every update until the stop. Not a no-op |

## Why it fails

On the warm solved cloud the accepted step moves the batch mean by a little: median |delta| .016, max .062, 200/200 fires. Cancelling that held the ring.

From scratch the same correction is large. Cold-ring |delta| median 4.67, p90 about 28, max 160 (ring radius is 3). The bias shift sits outside the curvature bound, so the next Adam step fights it. The cloud never settles on 8 modes. The same mean lock breaks the trajectory map (MSE .082).

Stay uses the same constant-rate ring. From 1210 to 1630 every 10-step check fails. |delta| stays large (median 4.55, max 84). The dropout window 1690–2300 was not reached: at update 1638 the post-correction mean no longer matched in float32 (absolute tolerance 1e-4) and the run raised. That is overflow of an already diverged correction, not a second mechanism.

## Recommendation

**KILL.** Do not retune the restore, do not clip it, and do not combine it with a slower G. Warm did not regress below 200/200, but cold trajectory and the AVX2 cold ring both regress versus #107, and the continued run is degenerate because it never acquires. #107 stall reach stays the GAN-native reference.

# Post-arm G common-mode output-gradient null

Host: neural. Draft only. Not solved. Not 22/22. Not production. Stall reach (PR #107) stays the reference. This bet does not displace it.

One mechanism, added to the stall-reach adapter. The null is a full batch-mean subtract. It was not scaled, and it will not be scaled.

## Mechanism

Stall reach, both curvature bounds (.25 / 3), and the losses are unchanged until an already-scheduled ring check reports 8 modes and HQ ≥ 0.9. That log only arms the mechanism. It is not a loss, and mode count is not a training target.

After the arm, on every G backward, `∂L/∂y` is replaced with `∂L/∂y − mean(∂L/∂y)` over the batch before that gradient enters G's weights. `y = G(z)` is left as D and the loss see it. There is no post-step mean restore, no train-forward centering, no trust probe, and no G-step reject. Adam betas and the D:G step counts stay put.

This is the distinction from the killed translation family: #124 centered the train forward, so D never saw the walk; #126 restored the mean after Adam and broke acquire; #117 bounded common-mode curvature and stayed open during the walk. Here D still scores raw fakes, and G still receives every non-translation direction.

## Gates

Seed 0, one thread, PyTorch 2.14.0+cpu. Ranked pin: `ATEN_CPU_CAPABILITY=avx2` plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. The warm receipt logs `cpu=AVX2` and `aten_cpu_capability=avx2`. The unchanged warm control is 200/200, min HQ .989990234375, so the warm result is rankable.

Receipts: [continuous-evidence/common-mode-grad-null](continuous-evidence/common-mode-grad-null/).

| Gate | PR84 pin | #107 stall reach | Common-mode grad null |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2, control 200/200) | 196/200, min HQ .866 | 200/200, min HQ .921 | **164/200**, min modes 6, min HQ .771. Final 8 / .9990. Suffix 47 from step 1154. Armed at 1000. **200/200 nulls** |
| Cold trajectory (AVX2) | PASS | PASS | not run (warm fail-fast) |
| Cold ring (AVX2) | 8, suffix 5 | 8, suffix 8 | not run (warm fail-fast) |
| Cold ring (AVX512, build check) | 7 | 8, suffix 8 | not run (warm fail-fast) |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, min 0, final 8 / .971, suffix 25 | not run (warm fail-fast) |

## What the null did on the warm continuation

Armed at the prefix check, host step 1000, phase `warm`. The scheduled prefix is passthrough, so the `CM_NULL` step field on this fork counts the 200 constant-rate updates (1 = host 1001, 200 = host 1200). The arm line uses the host check step.

All 200 G updates subtracted a non-trivial common mode (`||mean(∂L/∂y)||₂ > 1e-6`, a log threshold, not a scale). The subtracted L2 ran from .000368 to .001405, median .000854. The hold then failed on host steps 1118–1153 (36 checks, down to 6 modes and HQ .771) and was back on the ring from 1154, ending at 8 / .9990.

## Call

**KILL.** Warm regresses versus the #107 pin (164/200 against 200/200) and versus PR84 (164/200 against 196/200). The control is a valid 200/200 AVX2 pin, so this is a real regression, not an unsolved prefix. Cold acquire and the stay window were not run. Continuous learning is not solved. Board #1 remains #107.

Do not retune this null. Do not multiply the subtracted mean by a coefficient below 1, do not null only some layers, and do not turn the 1e-6 log threshold into a gate that sometimes skips the subtract. The warm dip is the full subtract working on every post-arm update. A smaller subtract would be the same mechanism turned down, which is the clip ladder this bet is not allowed to start.

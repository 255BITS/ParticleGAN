# Sampled-data coverage projection

The declared one-sided coverage correction fails the dense warm filter:
**197/200 passing updates**, all eight modes retained, minimum HQ .84619.
The failing steps are 1133, 1148 and 1186. It did not advance to cold
acquisition or the longer hold. The original PR84 smoothed-only adapter
remains the selected partial candidate, still unqualified.

## Frozen rule and its limits

After the unchanged bounded PR84 G/prior update, the current discriminator
real minibatch assigns each real sample to its nearest clean generated
particle. Each nonempty cell proposes its centroid; empty cells stay fixed.
With the generator network frozen, each output displacement is pulled back
through its own latent Jacobian in the already-updated Adam diagonal metric:

`delta_z = P J^T (J P J^T)^+ (centroid - G(z))`.

The full correction is tried first, then at most eight halvings. It is
accepted only if actual nonlinear mean squared distance from real samples
to their nearest generated particle strictly decreases. Failed trials
restore the prior exactly. The scalar learning rate cancels from this
projection; it is an extra state-dependent projection, outside PR84's G
curvature bound, rather than an ordinary Adam step or an LR schedule.

This explicitly adds a one-sided coverage objective. It is not a GAN
optimizer-only correction, distribution-matching guarantee, or penalty
pulling a critic toward zero. The real minibatch contains 128 examples on
this host; source recipe metadata reports 256 but is not the sampler's
actual batch. There are no target centers, mode counts, seed changes,
extra training draws, or elapsed-time gains in the update. Conditional
trajectory retains the unchanged original PR84 path because unconditional
marginal matching would discard conditioning.

## Cheap filter and full warm result

The clean-state filter replays the failed original PR84 cold run exactly.
Using its actual final D minibatch, one full correction decreases coverage
error .64466 to .01128 and changes clean support from seven modes/HQ1 to
eight modes/HQ .9167. Network parameters, Adam state and RNG are unchanged.
That establishes a usable nonlocal acquisition direction at one state,
not stable training. The earlier output-space Lloyd proxy uses archived
noisy support observations; its scope is kept separate from this clean
network-Jacobian experiment.

| Warm fork | Passing updates | Minimum HQ | Final modes/HQ |
| --- | ---: | ---: | --- |
| Scheduled identity | 200/200 | .99634 | 8 / 1 |
| Ordinary constant-rate Adam | 6/200 | .00537 | 6 / .50464 |
| Frozen PR84 original | 200/200 | .97021 | 8 / 1 |
| Added coverage projection | 197/200 | .84619 | 8 / .97876 |

The scheduled identity has exact cold-host parity. The correction-disabled
active PR84 control exactly matches the archived original warm state,
final state, all observations and dense diagnostics. All 200 coverage
proposals accept alpha1 and reduce the current batch coverage objective;
that criterion nevertheless does not protect every generated particle's
quality. An [exact read-only replay](coverage-failure-diagnosis.md) separates
two causes. At updates 1133 and 1186, the GAN step strands particles whose empty
sampled-real cells receive no correction. At update 1148, one assigned real outlier
sets a bad centroid: the post-GAN state passes HQ .95288, then the projection
reduces it to .89697. Its nonlinear landing is faithful to that target.
The replay reproduces the full final state and all observations; paired
stage scores use the original 4,096 evaluation draws and noise, not a relaxed
clean-support proxy. Saved post-GAN model states support cheap tests of a
bidirectional sampled-data objective before another training run.

Eleven independent tests validate full-rank and rank-deficient pullbacks,
rejected-trial restoration, unchanged optimizer/RNG state, exact disabled
host parity, correct current D batch capture, three field evaluations with
one moment update, placement before EMA, and the unchanged conditional path.
Together with three original-adapter checks, the focused suite passes 14/14.
The integrated research suite, including all previous controller tests,
passes 230 tests in 29.43 seconds. The failed-warm summary also makes the
driver reject a cold invocation before creating its output directory.

## Research checked after the failure

[Chamfer Guidance (2025)](https://arxiv.org/html/2508.10631v2) uses both
real-to-generated and generated-to-real nearest-neighbor terms to address
image diversity and quality. Its intervention is inference-time guidance,
not continuous GAN training. It motivates checking precision separately
from our one-sided coverage measure; it provides no stability guarantee
for this update. [QAL (2026 conference preprint)](https://arxiv.org/html/2511.17824v1)
reports that even symmetric Chamfer and EMD can misbalance recall and
precision in 3D reconstruction. Its task-specific thresholds are not being
substituted into our unchanged GAN gate. These are reasons to diagnose
the actual off-support particles before choosing a second objective, not
evidence that adding a symmetric term will solve the problem.

Two newer directions merit a cheap test rather than a training sweep.
[Weighted MMD quantization (AISTATS 2026)](https://proceedings.mlr.press/v300/belhadji26a.html)
optimizes particle weights as well as positions; its robust results do not
transfer to this host's fixed uniform weights. Its MSIP descent analysis
assumes positive optimized weights and a suitable varying step sequence;
fixed-step convergence remains open. Changing sampling weights would change
the host model and is not being counted as a fix under the existing gate.
[Gradient Flow Drifting (March 2026)](https://arxiv.org/html/2603.10592v1)
connects a Gaussian-kernel drift to the difference of real and generated KDE
scores. That field can use fixed uniform particle weights and rest when the
smoothed distributions match, but bandwidth and minibatch estimation remain
material, and its population argument does not guarantee stability for
twelve equal particles representing eight Gaussian modes. Any next kernel
rule should first recover useful directions on the saved failed states and
preserve the matched fixed cloud, using one declared bandwidth rule.

The [hash manifest](continuous-evidence/coverage-projection-round4/manifest.json)
archives the declaration, exact sources, warm forks and tailable log.
The driver [coverage_projection_probe.py](coverage_projection_probe.py)
refuses cold acquisition unless all 200 warm checks pass, both control
parity checks pass, and the source hashes match the warm declaration.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
 /tmp/pr38-default-env/bin/python -u reports/toy100/coverage_projection_probe.py \
 --phase warm --output NEW_WARM
```

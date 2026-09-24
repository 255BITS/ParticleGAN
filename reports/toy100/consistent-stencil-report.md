# Same spatial critic operator for both players

One declared ablation of PR84 passes warm stability and trajectory but fails
cold ring acquisition: **7 modes, HQ .78125, zero passing observations**.
No long hold or production test was run. This does not improve the leading
partial candidate, PR84's original generator-only smoothing.

PR84 evaluates D's loss with sharp logits and G's loss with a five-point
spatial average. That asymmetry is a plausible source of inconsistent
responses between players. This experiment gives D and G the same stencil,
including D's existing cap penalty. Width is measured at the outer base once
and frozen through all three same-batch replay phases. G bound .25, D bound3,
width cap .15 and the `.5 / sharpness` rule are inherited without tuning.
There is no slope gate, target geometry input, extra penalty or clock decay.

This is an exploratory consistency test. The archived PR84 warm-only failure
does not reproduce locally; therefore this experiment cannot be described as
repairing an established warm defect. It also changes width timing: PR84
recomputes width after D in each phase, whereas the shared operator freezes it
at the outer base. Width is .15 throughout warm continuation; during cold
ring it ranges from .12445 to .15. Therefore this tests a consistent frozen
operator, not the effect of D smoothing independently of width timing.

| Gate | Measured result |
| --- | --- |
| Scheduled identity fork | Exact full-state parity |
| Dense warm continuation | PASS200/200; minimum8 modes/HQ .922607 |
| Constant-rate warm control | FAIL6/200, matching previous controls |
| Cold trajectory400 | PASS MSE .000942662, suffix18 |
| Cold ring1200 | FAIL7 modes/HQ .78125;0/24 passing |

The stencil is currently limited to 2D `SimpleMLPDiscriminator`, matching
PR84's supported scope. Trajectory follows the unchanged bounded alternating
path; its pass does **not** validate spatial smoothing on conditional data.
The adapter is scratch-only and is ineligible for the production common22 gate.

Five focused tests validate the analytic quadratic stencil value, derivatives
for both players and the mixed derivative needed by the cap; exact zero-width
bounded-host parity; delayed activation parity; RNG replay and one Adam
moment update per outer iteration; and invalid width rejection. A separate
Sol review found no formula or update-order defect in this declared scope.

Warm state SHA256 is
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`.
Raw outputs, declarations, source snapshots and logs are in the
[hash manifest](continuous-evidence/consistent-stencil-round4/manifest.json).
The cold trajectory took4.03 seconds and ring63.62 seconds on the pinned
single-thread CPU environment. This costs more than unilateral smoothing
because D and its input-gradient penalty also evaluate the stencil.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
  /tmp/pr38-default-env/bin/python -u reports/toy100/consistent_stencil_probe.py \
  --phase warm --output NEW_WARM
# Only after the warm summary passes:
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
  /tmp/pr38-default-env/bin/python -u reports/toy100/consistent_stencil_probe.py \
  --phase cold --previous NEW_WARM/summary.json --output NEW_COLD
```

The mechanism was isolated before looking for related research.
[Smoothness and Stability in GANs](https://arxiv.org/abs/2002.04185) studies
conditions for generator stationarity; its results do not establish a
convergence guarantee for this capped, alternating Adam experiment.
The recent [adaptive-noise paper accepted August27,2026](https://journals.aps.org/pre/accepted/10.1103/ch8n-wrv6)
instead changes the discriminator's final CDF activation and learns its
noise scale. That is a different intervention and was not substituted into
this isolated experiment.

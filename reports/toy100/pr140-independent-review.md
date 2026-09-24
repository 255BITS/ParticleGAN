# Independent review of PR140

**The global first-place claim is not verified.** This audits PR140 head
`ef4084a46d999af33c8c484ef36dfdafbe3fc517`; the submitted head was unchanged at
completion. Its report claims the best continuation on its own PR107 comparison
line, explicitly says not solved, and does not demonstrate a full toy-suite pass.

Recounting its recorded continuation confirms114/120 passing observations,
evaluated every10 updates from1210 to2400. Six fail; at2190 it has5 modes and
HQ0.362548828. The endpoint returns to8 modes/HQ1.0. These are120 sparse checks,
not1200 dense passes or a complete post-convergence hold. Warm200 passes occur
before the new LR rule takes any steps (zero fires), so they do not validate the
new intervention. The declared10 source hashes and12 warm source hashes match.

| Independent environment | Cold ring at1200 | Continuation1210–2400 |
| --- | --- | --- |
| PyTorch2.13+cu126, full AVX2 pin | 7 modes/HQ1, FAIL | 0/120, final5/HQ.3826 |
| PyTorch2.14+cpu, full AVX2 pin | 7 modes/HQ1, FAIL | 0/120, final5/HQ.3826 |
| PyTorch2.14+cpu, documented ATen AVX2 pin only | 7 modes/HQ1 at1200 | 0/120, final5/HQ.3826 |

Both independently run cold trajectories pass. The delayed LR rule never arms
in these replays because they do not acquire8 modes. Thirteen focused unit tests
pass in each environment (13 distinct cases). The2.14 CPU wheel has the same
PyTorch core revision as the reported2.14+cu130 build; CUDA build/hardware
numerical differences remain unresolved. These observations do not disprove
that the author's environment produced its saved results.

The method remains a GAN. It changes G/prior learning rates from.00425/.0085 to
.002125/.00425 after a benchmark8-mode/HQ>=.9 observation at or after1200.
That uses evaluation information in training control and is a different policy
from our fixed-rate, evaluation-independent candidates. Even accepting that
policy, the submitted evidence does not establish stable-and-passing release
qualification. Retain it as an unverified contender; do not promote it to #1.

[Machine-readable review and raw replay outputs](pr140-independent-review/review.json).
[Submitted report](https://github.com/255BITS/ParticleGAN/blob/ef4084a46d999af33c8c484ef36dfdafbe3fc517/reports/toy100/delayed-arm-g-lr.md).

Replay from the pinned PR checkout, with the selected interpreter:

```bash
env -u PYTHONPATH -u MKL_ENABLE_INSTRUCTIONS -u ONEDNN_MAX_CPU_ISA -u DNNL_MAX_CPU_ISA OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 CUDA_VISIBLE_DEVICES= /tmp/pr140-torch214/bin/python reports/toy100/gan_followup_probe.py --phase stay --method delayg05 --output NEW_DIRECTORY
```

Full AVX2 runs additionally set MKL_ENABLE_INSTRUCTIONS, ONEDNN_MAX_CPU_ISA and
DNNL_MAX_CPU_ISA to AVX2. The ordinary benchmark environment was left unchanged.

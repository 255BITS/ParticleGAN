# 100-mode convergence: baseline failures

The current public GAN recipe fails the frozen live-weight gate on all three
100-Gaussian geometries at 7,000 updates. This report preserves those failures
before configuration search. The benchmark was branched from `develop` at
`3d08a3a`; each run records hashes of the exact training and evaluation source.

![Baseline convergence failures](baseline/toy100-progress.gif)

| Problem | Live modes | Within 3σ | Gate |
| --- | ---: | ---: | --- |
| Square grid | 63/100 | 57.18% | FAIL |
| Rotated grid | 94/100 | 83.63% | FAIL |
| Staggered grid | 88/100 | 72.38% | FAIL |

[Full numerical leaderboard](baseline/leaderboard.md) ·
[Machine-readable gate](baseline/gate.json) ·
[Protocol and commands](../../docs/toy100.md).

A mode requires at least 100 in-radius samples out of 20,000 generated draws.
Quality also requires ≥97% HQ, balanced mass, and noncollapsed per-mode spread
at five consecutive terminal checkpoints. This is stricter than the old
README visualization's 10-hit mode count. All three runs improve late during
the cosine learning-rate anneal but do not meet the gate. Their exact curves
are not a replay of the older README run: the new sampler uses one mode-index
draw per sample instead of two coordinate-index draws.

The GIF includes the untouched step-zero model and dense early observations.
It plots 4,096 saved generated samples per task; metrics use 20,000 draws.
Saved frames are a visualization subset, not the complete evaluation cloud.
All recorded snapshots and complete training/evaluation logs are retained in
the baseline folders. Timing is from an NVIDIA RTX A6000, one fixed seed
(1234), with no seed search. Initial trained evidence is preserved separately
from any later tuned configuration.

```bash
python -u -m benchmarks.toy100 run \
  --config configs/toy100/baseline.json --device cuda:0 \
  --output artifacts/toy100/baseline
# Expected exit status: 1 (0/3 problems pass).
python experiments/leaderboard.py --toy100-output reports/toy100/baseline
```

Validation before search: existing suite **686 passed**, 9 skipped, one
pre-existing expected failure; focused toy100 tests also pass. Skips include
optional Gym/image dependencies and CUDA-only tests in the CPU suite.

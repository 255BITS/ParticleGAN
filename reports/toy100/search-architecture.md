# Architecture search: fixed historical optimizer

This bounded search starts from the recorded failing baseline and changes the
critic's Fourier resolution or the generator/critic MLP capacity. It keeps the
same three 100-Gaussian tasks, seed 1234, 7,000-update budget, 20,000-draw live
evaluation, and frozen five-check gate. The historical optimizer settings used
for every candidate are LR 0.0006, discriminator multiplier 1.5, prior
multiplier 10, `b_cap` coefficient 1, prior regularization 1, and Adam β₂
0.999. Batch 256 and the baseline cosine schedule remain fixed here; the
separate schedule/batch lane handles those variables.

The five declared grid candidates are `hist_f3_w128_d3`,
`hist_f4_w128_d3`, `hist_f6_w128_d3`, `hist_f4_w192_d3`, and
`hist_f4_w128_d4`. `f` is the critic Fourier level, `w` the hidden width of
both MLPs, and `d` their hidden-layer count. Their exact overrides are in
[`search_architecture.json`](../../configs/toy100/search_architecture.json).
The initial plan advanced the best grid candidate to the other two problems.

Selection is fixed before seeing outcomes. A sustained grid PASS outranks any
failure. Within the same gate status, rank by lower mean normalized threshold
shortfall over the final five live evaluations. The nine bounds are 100 modes,
97% precision, minimum in-radius mode mass 0.5%, maximum mode mass 2%, mass TV
at most 0.10, covariance eigenvalue ratios 0.40–1.70, and median radial ratios
0.65–1.40. Each positive violation is divided by its bound and capped at 2.
Ties go to higher worst terminal mode coverage, then higher worst terminal HQ.
An advancing failure remains a failure; the all-problem gate decides whether
the result is a viable recommendation.

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m benchmarks.toy100.search \
  --base configs/toy100/baseline.json \
  --candidates configs/toy100/search_architecture.json \
  --output artifacts/toy100/search-architecture \
  --problem grid100 --device cuda:0 \
  > artifacts/toy100/search-architecture.log 2>&1
tail -f artifacts/toy100/search-architecture.log
```

## Measured grid results

All five candidates finished their 7,000 updates on one NVIDIA RTX A6000.
Every row below is a **FAIL** under the frozen live-weight gate; none ever
reached all 100 modes or a single full-quality passing checkpoint. The
"terminal shortfall" is the prespecified mean over the final five checks,
and is a diagnostic ranking among failures, not a substitute for PASS.

| Candidate | Gate | Terminal shortfall ↓ | Final modes | Final HQ | Final mass TV | Final covariance eig range | Total seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fourier 3, width 128, depth 3 | FAIL | **0.1201** | **98/100** | 96.91% | 0.1221 | 0.094–1.844 | 70.1 |
| Fourier 4, width 128, depth 3 | FAIL | 0.2346 | 92/100 | 98.40% | 0.1567 | 0.146–1.943 | 69.9 |
| Fourier 4, width 128, depth 4 | FAIL | 0.2858 | 93/100 | 97.43% | 0.1623 | 0.057–1.937 | 83.1 |
| Fourier 4, width 192, depth 3 | FAIL | 0.4526 | 79/100 | 86.67% | 0.1743 | 0.006–2.454 | 72.3 |
| Fourier 6, width 128, depth 3 | FAIL | 0.4600 | 73/100 | 88.07% | 0.2060 | 0.009–1.818 | 71.8 |

The final 20,000 live samples saved for each candidate were independently
rescored on CPU with `metrics.evaluate_samples`; every scalar matched its
recorded final event to numerical precision. The 4,096-sample GIF snapshots
are diagnostic subsets. Exact trial configs, complete curves, final clouds,
and snapshots are retained under `artifacts/toy100/search-architecture`.

Fourier 3 is the best **failing** grid arm. Its two least-populated modes miss
the 100-hit coverage floor, and its 96.91% HQ, 0.1221 mass TV, and per-mode
covariance range all miss separate quality bounds. The larger models do not
fix these failures. After seeing all five fail, we stopped before the planned
cross-geometry promotion and moved to a separate, explicitly declared
allocation/sharpness mechanism test. No architecture from this screen is a
recommended all-problem recipe.

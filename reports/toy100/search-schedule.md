# Toy100 schedule and batch search

Seven declared grid100 candidates kept the same seed (1234), 7,000-update
budget, public MLP generator/discriminator, 20,000-particle prior, and frozen
20,000-sample live-weight gate. All started from the historical lower-LR
recipe: generator LR 0.0006, D LR ×1.5, prior LR ×10, Adam β=(0, 0.999),
one-sided cap coefficient 1 with κ=1, prior regularization 1, Fourier-2,
and width 128. Only anneal start or batch size changed. The exact candidate
overrides are in [`search_schedule.json`](../../configs/toy100/search_schedule.json)
and full run receipts are in [`grid7k/results.json`](search/trials.json).

| Grid candidate | Live modes | HQ | Mass TV | Worst covariance eig. ratios | First 100-mode step | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| anneal at 40% | 97 | 0.9716 | 0.1156 | 0.095–2.478 | — | FAIL |
| batch 128 | 98 | 0.9731 | 0.1338 | 0.145–1.338 | — | FAIL |
| anneal at 25% | 99 | 0.9690 | 0.1182 | 0.127–2.386 | — | FAIL |
| constant LR | 2 | 0.2288 | 0.1092 | 0.375–4.179 | — | FAIL |
| anneal from step 0 | 95 | 0.9632 | 0.1225 | 0.077–2.523 | — | FAIL |
| batch 512 | 99 | 0.9817 | 0.1146 | 0.095–2.359 | — | FAIL |
| batch 1024 | **100** | 0.9820 | 0.1164 | 0.060–3.081 | **6,000** | FAIL |

The batch-1024 MLP kept all 100 modes from step 6,000 through step 7,000,
meeting the five-check *coverage* suffix. It did not meet the quality gate:
mode TV exceeded 0.10, and the worst mode's covariance and radial width were
outside their frozen ranges. Constant LR left live weights oscillating far
from a sharp solution. Earlier annealing sharpened samples but did not repair
the allocation that feeds each mode.

The batch-1024 recipe was promoted unchanged to the other two geometries. The
[all-problem leaderboard](search/README.md)
reports FAIL 0/3: rotated100 ended at 99 modes, HQ 0.9727, TV 0.0902, with
one underpopulated and collapsed mode; staggered100 ended at 99 modes,
HQ 0.9817, TV 0.1081, also with poor worst-mode spread. This is evidence for
full grid coverage under the classic architecture, not an all-task pass.

Related fixed-seed follow-ups used saved scratch model sources and kept the
same evaluator. The affine direct-particle F3 recipe that passed grid100
failed unchanged on [rotated100](search/README.md)
(88 modes, HQ 0.8162, TV 0.1490); on
[staggered100](search/README.md)
it covered 100 modes with per-mode spread inside the gate but had HQ 0.9313
and TV 0.1061. A generic disk radius 6.5 initial prior failed even on grid
(97 modes, HQ 0.8188); raising the affine generator LR while keeping absolute
prior/D LRs fixed left rotated coverage at 88 modes. Both tests are retained
under `artifacts/toy100/search-schedule/`.

One schedule follow-up to the separate noisy MLP near-pass changed its anneal
start from 40% to 25%. It ended grid100 at 96 modes, HQ 0.9912, TV 0.1089:
early sharpening starved modes. The [run receipt](search/trials.json)
preserves its full curve. Extending that noisy MLP recipe unchanged from 7,000
to 8,000 updates also failed: the [8,000-step run](search/trials.json)
ended at 96 modes, HQ 0.9953, and TV 0.1081. Its schedule fractions are tied
to the total budget, so this is an 8,000-step recipe test, not continuation
of the 7,000-step weights.

The separate input-noise probe added Gaussian noise to discriminator inputs,
with peak standard deviation 0.5 declining linearly to zero halfway through
training. It also used fresh output noise of standard deviation 0.026. At batch
512, its saved CPU runs passed the strict five-check live gate on
[grid100](search/trials.json)
and [rotated100](search/trials.json),
but staggered100 ended at 99 modes. A one-variable follow-up raised only the
staggered100 batch size to 1024. The
[staggered100 gate](search/trials.json)
passed all five terminal live checks from step 6,000 through 7,000. Its final
20,000 draws covered all 100 modes with HQ 0.9826, minimum in-radius count
113, TV 0.0750, worst covariance eigenvalue ratios 0.499–1.038, and radial
median ratios 0.777–1.118. The [full run](search/trials.json)
and [saved probe source](search/sources/1a9f0daff39025e231fbf66d82222437e76f5391f76769aa9639008dd535f92a.py)
record this result. The declared per-problem batch override was then run by
the production all-problem command. Its [aggregate gate](recommended/gate.json)
passes 3/3 with five terminal live checks on every problem. Every evaluation
metric row and all final scored sample arrays match the saved CPU probes
exactly for all three problems.

These are fixed-seed configuration comparisons, not a robustness estimate.
The gate checks minimum per-mode 3σ mass, 97% precision, TV ≤0.10, maximum
mode mass ≤0.02, and worst per-mode covariance and radial spread. It cannot
prove exact Gaussian shape outside the audited 3σ core.

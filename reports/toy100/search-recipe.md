# Traditional recipe screen on grid100

Seven configurations trained the public `GANTrainer` for 7,000 updates on
`grid100`, with seed 1234, 20,000 evaluation draws, the same 250-update
checkpoints, and the frozen live-weight gate. The original baseline is in
[baseline/leaderboard.md](baseline/leaderboard.md). This screen varied optimizer and
regularizer fields only; every arm kept batch size 256, a 20,000-particle prior,
Fourier-2 discriminator, three hidden layers of width 128, and the delayed
cosine schedule (start 0.6, floor 0.05). The declared candidate list is
[`configs/toy100/search_recipe.json`](../../configs/toy100/search_recipe.json).

The historical reference used G LR 0.0006, D multiplier 1.5, prior multiplier
10, Adam β=(0, 0.999), one-sided cap coefficient 1 and κ=1, and prior
regularization weight 1. The second arm doubled the base LR. The other arms
changed one field relative to that doubled-LR arm.

| Configuration | Changed field | Live modes | HQ | Mode TV | Covariance eigenratio range | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| historical_lr0006 | historical reference | 98 | 0.98665 | 0.11990 | 0.236–2.958 | FAIL |
| historical_lr0012 | LR 0.0012 | 93 | 0.98415 | 0.15040 | 0.253–1.937 | FAIL |
| historical_cap6 | cap coefficient 6 | 80 | 0.93260 | 0.19000 | 0.018–1.356 | FAIL |
| historical_prior2 | prior LR multiplier 2 | 97 | 0.98650 | 0.12970 | 0.270–2.490 | FAIL |
| historical_reg005 | prior regularization 0.05 | 95 | 0.97610 | 0.15385 | 0.097–2.936 | FAIL |
| historical_beta099 | Adam β2 0.99 | 97 | 0.97920 | 0.13125 | 0.256–2.093 | FAIL |
| historical_d1 | D LR multiplier 1 | 92 | 0.98760 | 0.15945 | 0.123–1.941 | FAIL |

No arm reached 100 covered modes at any checkpoint, and no arm passed the
terminal five-check live gate. The historical 0.0006 recipe improved sharpness
and coverage over the public-default baseline, but its two least-populated
modes had only 96 and 98 in-radius draws, below the required 100. Its TV of
0.1199 exceeds the 0.10 limit. The covariance failures are substantive:
one mode with 185 in-radius draws had an eigenratio near 0.237, while another
with 188 draws had an eigenratio near 2.974. Near-perfect aggregate HQ therefore
does not establish balanced, correctly shaped components.

The stronger cap, weaker prior motion, and weaker prior regularization all
reduced coverage or balance in this controlled screen. Further work should
target mode allocation and within-mode spread without changing the gate:
test a larger particle table or a target-free stochastic output/prior mechanism,
then verify the full three-problem suite after a grid arm passes all live checks.
The exact final 20,000 live and EMA draws for each arm are retained in its
`final_samples.npz` next to the complete event curve and source-hash receipt.
This is one fixed-seed configuration screen, not a multi-seed robustness claim.

The run source was commit `8e0a7bd5c78800aac39fe9d17193bec50085447e`,
with `benchmarks/toy100/train.py` SHA256
`4b4d20556946a5f0984bbd64a546b449a3490da267100064e4ad1216697289c7`.
The search plan, full gate results, per-step JSONL, and sample arrays are under
`artifacts/toy100/search-recipe/` in the local workspace.

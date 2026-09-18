# Capacity scout interpretation

Both scouts completed and passed source/configuration certification. Neither
improved on the historical width32 baseline at the same 20k update budget.

| Configuration | FID50k at 10k | FID50k at 20k | Test MSE at 20k | Training minutes |
|---|---:|---:|---:|---:|
| Historical width32 | 19.611 | 20.105 | 0.05816 | 14.96 |
| Width64, depth2 | 37.000 | 25.778 | 0.04244 | 24.04 |
| Width64, depth1 | 21.020 | 35.440 | 0.04964 | 19.03 |

Depth2 wins among the new candidates, but remains 5.674 FID points behind the
historical control. Its reconstruction MSE is about 27% lower. Width64 depth1
regressed by 14.421 FID points from 10k to 20k despite slightly improving MSE.
Widening therefore helped reconstruction without reliably improving generation
under these training settings. This does not establish that larger generators
cannot work with different optimization settings.

The still-running width32 duration track has FID50k 19.424, 18.901, 19.977,
20.355 and 20.340 at 40k through 80k respectively. MSE fell to 0.03642 at 80k,
about 24% below its 30k starting point, while generation FID remained around
19–20. These intermediate results weaken the hypothesis that reconstruction
improvements will translate into a large delayed FID improvement. The final
90k/100k evaluations remain pending at the time of this interpretation.

Recommendation: finish the duration track, then reassess both results before
launching more training. Do not automatically extend width64 depth1. Depth2 is
still improving, so a bounded extension could test delayed convergence, but it
is not yet a demonstrated replacement for width32. If the duration run stays
near 20, prioritize a controlled reconstruction-weight ablation on width32
over another capacity increase. This would test objective balance directly;
the current results do not identify the cause of the generation/reconstruction
gap. Keep lazy bcap N=8 and other settings fixed for that comparison.

See [certified leaderboard](LEADERBOARD.md) for the full protocol and measured
throughput. Historical timing came from a separate execution; these scouts ran
while GPU0 trained the duration track. All runs use the same configuration-control
seed, with no seed sweeps.

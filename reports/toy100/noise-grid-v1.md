# Bounded shared-noise compatibility screen

The [frozen manifest](noise-grid-v1-manifest.json) declared nine public-v3-core
noise combinations and three extra controls before training (manifest SHA-256
`bea4ad1d67453690ce807c8c0dedcd92103c836138c4ce9d1cd556edc57b32aa`).
It reused the exact existing σ=0.029/end=0.15 episode and ran the other 11
once, with seed 0, one CPU thread, the same eight frozen hosts, budgets,
thresholds, and source bytes. All 12 rows have complete G-output/D-input
noise receipts and pass independent episode integrity checks. Each result is
a **subset diagnostic**; 8/8 was required before any full 19-host replay.

The nine-cell grid uses β₂=0.99, prior LR multiplier 2, a 60% LR anneal start,
5% LR floor, output warmup over 20% of each host budget, and input-noise peak
0.5. Cells show live passes out of eight, then the sum of missing terminal
checks across failed hosts. Five final passing checks are required per host.

| Output σ | Input ends 12.5% | Input ends 15% | Input ends 17.5% |
| --- | --- | --- | --- |
| 0.0275 | **7/8; gap 1** (overlap) | 7/8; gap 5 (unequal-mass) | 6/8; gap 6 (anisotropic, stripes) |
| 0.028 | 3/8; gap 19 | 4/8; gap 15 | 6/8; gap 10 |
| 0.029 | 5/8; gap 12 | 6/8; gap 4 (exact reused row) | 7/8; gap 5 (overlap) |

The predeclared controls also fail: changing the β₂=0.999/prior-3 core to
output σ=0.028 at input end 20% gives 3/8; reducing input peak to 0.25 gives
6/8 for the public-v3 core at σ=0.029/end 15%, and 4/8 for the β₂=0.999/prior-3
core at σ=0.029/end 20%.

The best row is `beta99_o0275_e0125`. Its overlap case has four terminal
passing observations, one short of the frozen requirement; every final metric
passes. The other seven cases pass their sustained gates. The next two 7/8
rows have larger terminal-check deficits. This rank uses live pass count first,
then total terminal-check gap, then the frozen normalized final-metric
shortfall. It does not imply transfer to the 11 untested canonical hosts or
to the three 100-mode problems. **No row reached 8/8; none was promoted to a
full 19-host replay.**

The complete numeric ranking and per-case misses are in the local raw result
`artifacts/toy100-accuracy/compatibility/noise-grid-v1/results.json`; the
tail-friendly execution log is alongside it as `master.log`. Each run has its
own source archive, copied config, protocol, compressed episodes, summary,
and log. Raw `artifacts/` paths are local workspace evidence, not GitHub links.
The [shared search ledger](shared-recipe-search.md) also includes all 11 new
episodes and identifies the reused row.

# More particle identity is visible in features; coverage gains flatten

All seven read-only probes passed certification after the GPU smoke and five CPU tests. Parent checkpoints/frozen state are unchanged, and every full probe used the same exact cached real-feature tensor. The diagnostic queue completed at16:46:09MDT on2026-09-18. Both GPUs are idle; no further training is queued.

## Matched20k results

| Particles | FID50k | Additional decodable sibling bits | Available sibling bits | Density | Coverage |
|---|---:|---:|---:|---:|---:|
|1024|19.8932|0|0|0.6281|58.13%|
|4096|18.5207|1.1166|2|0.6344|60.28%|
|8192|18.1602|1.7195|3|0.6352|60.29%|
|16384|18.0136|2.2756|4|0.6294|59.43%|

Bits are conditional on the original1024 parent identities. The baseline zero is zero additional sibling information, not zero image information. The restricted held-out decoder recovers about56–57% of the available additional label-bit budget at each expanded count. Adding particles therefore creates distinctions visible in frozen Inception features, rather than merely assigning redundant labels. This is consistent with part of the proposed compression explanation; it does not prove that extra information is semantic or useful by itself.

FID improves at both15k/20k as count increases, but the marginal20k gains shrink:1024→4096 gains1.3724;4096→8192 gains0.3605;8192→16384 gains0.1466. The latter differences are small and one trajectory per count does not estimate training-run uncertainty. Training cost remains close:7.25/7.37/7.28/7.46minutes respectively. Reused1024/4096 benchmarks share the original10k parent and protocol; no seed repeats were run.

The quality/diversity proxies do not show monotonic improvement with count. Coverage rises about2.15percentage points at4096, is effectively unchanged at8192, then falls0.85points versus4096 at16384. Density stays approximately0.63 across the20k arms. Between-sibling feature-variance fractions rise6.25%→8.45%→10.14%, while within-child fractions stay about31–33%. More of the variability is associated with sibling choice, but that does not automatically broaden coverage of real features.

## What happened to the4096 run after its best point

| Step | FID50k | Bits | Decoder accuracy | Density | Coverage |
|---|---:|---:|---:|---:|---:|
|35k|16.5033|1.4938|90.63%|0.6303|62.58%|
|40k|17.2350|1.4854|90.44%|0.6148|61.39%|

The FID rebound coincides with lower density and coverage while sibling identification remains strong. This points toward deteriorating real-distribution alignment despite maintained particle distinctions. It does not establish causation or a statistically significant entropy change.

In particular, the0.0084-bit aggregate change is much smaller than the descriptive parent standard errors (~0.062/0.098). One40k parent has98% test accuracy but−1.253bits because validation selected an overconfident decoder and a few test mistakes receive very large cross-entropy penalties. We retain that result, rather than tuning on test data or clipping negative bits. Do not interpret the near-equal aggregate bits as a precise entropy estimate. The20k scaling arms have no negative per-parent observed estimates; their information increase is not driven by this outlier.

The1024 control deteriorates by40k: FID26.0216, density0.4662, coverage45.14%. Its overall feature-variance ratio is still1.096 versus real, another concrete example of why total variance alone is insufficient.

## Validation and recommendation

Identical-clone controls yield zero additional bits (floating error below1e-15). Shuffled-label controls lie between−0.0017 and0bits. Training, validation and test use separate noise streams; decoder selection never reads test data. These controls support that positive20k scores reflect recoverable particle identity. Bounds remain decoder-dependent estimates; feature distinguishability may include artifacts. Density/coverage use10000real/fake samples,k5 and fixed Inception features; they are not direct human-quality or semantic-recall measurements.

Recommend continuing both8192 and16384 from20k to40k, comparing against the already measured4096 duration curve. Their20k difference is too small to confidently choose a winner, and additional centers receive fewer direct samples per center. This tests whether their modest early gain strengthens with training. Do not increase count again solely to maximize information bits, and do not jump to200k based on these short scouts. Nothing new was launched for this results review.

Sources for definitions: [InfoGAN](https://arxiv.org/abs/1606.03657), [density and coverage](https://proceedings.mlr.press/v119/naeem20a.html). Exact numbers and per-parent estimates are in results.json; compact tables in LEADERBOARD.md.

# Fixed bounded-neighbor supplement

Evidence VALID, original quality remains FAIL. Helper/protocol/input bytes frozen before measurement. Modes69,80,8 (largest holdout covariance) and67 (median) are descriptive annotations. Fixed radial extremes plus evenly spaced rows give64 FAST/62 EMA queries and2.52M query-to-table distance pairs, with no all-table quadratic graph. Actual immutable production geometry agrees bitwise with reconstructed candidate radius/width. CPU RNG unchanged; no draws, emissions, updates or protected writes.

| Saved population | Exact positive nearest captured | Median/max radius inflation | Median/mean output jitter RMS² bound in targetσ² |
| --- | --- | --- | --- |
| FAST |54/64|1.0/5.4169|.001929/.007947|
| EMA |58/62|1.0/2.1700|.000686/.007936|

Missing positive neighbors can inflate radius and widths. Example FAST row5554(mode80) radius .002832 versus exact .000523 (5.42×); bounded widths(.00818,.01966) versus exact nearest8(.00171,.00111). Its clipped output-displacement RMS² bound is still .00861 targetσ². The original approximation is not an exact nearest-neighbor kernel. All selected rows have zero full-vector duplicates, so zero-distance exclusion does not explain these misses.

Some axis/lineage candidates repeat the same row;29/64 FAST and26/62 EMA nearest8 selections contain fewer than8 unique rows. Some selected neighborhoods cross annotated modes (2 FAST and8 EMA queries). These can inflate local coordinate widths, but global-bandwidth and especially radius clipping restrict displacement. Cross-mode candidates alone do not establish realized tail inflation.

For persistent worst mode69, all14 selected EMA rows capture the exact positive nearest radius (inflation1.0). Their output-displacement RMS² bound has median .000900, mean .009248 and max .057288 targetσ². FAST mode69 captures15/16; its mean bound .003460, max .009277 and greatest radius inflation1.443. These subset bounds are small compared with raw EMA anchor max-covariance1.47276 and served clean holdout1.52010. Radius misses do not explain the main persistent mode69 covariance excess on this fixed subset; broad anchor geometry is the stronger current hypothesis.

This result does not certify all20000 neighborhoods or rule out another row's jitter. RMS bounds are theoretical bounds for existing clipped Gaussian displacement, not measured sample variance. Original sampled row IDs were not saved. No production correction is justified solely by these126 queries; a prospective geometry change would still require the original toy and Grid100 gates.

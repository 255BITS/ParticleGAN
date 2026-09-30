# Reserved population-flux diagnostic

Status: VALID diagnostic, RESERVED design. No production change or quality claim. Reads use sealed RA7 CPU tensors; no model forward, gradient, optimizer call, sample, seed or CUDA context.

## What fails

| Saved step | Own-row participants at b | At 2b | Required | G scale / stamp | Table scale / stamp |
| --- | ---: | ---: | ---: | --- | --- |
| 500 | 488 | 277 | 973 | .25 / stationary | 1 / drift |
| 1000 | 299 | 152 | 973 | .125 / stationary | 1 / drift |
| 2000 | 281 | 75 | 973 | .0625 / stationary | 1 / drift |

Moved rows lose every unfinished and completed pair entry. Their absence is selected by support/count actions, so survivor averages cannot represent all rows without further assumptions. All saved pair completion intervals span zero. For example, the final oldest 2b pair has 75 observed rows, survivor mean −.18149 and full-population identification interval [−.94005,+.91347]. These are missing-data bounds, not confidence intervals. The negative decision at 1464 also rejected coverage (323/973); its cleared pair tensors cannot be reconstructed from the final state.

The G tester tracks displacements of the full network parameter vector. Its stationary stamp certifies its declared direction statistic under that tester's assumptions. It does not certify prior rows, emitted support, FAST/EMA correspondence, or equality to real data. Serving currently uses the table stamp alone.

## Tiny fixed reachability proof

The immutable base tester is exercised on 64 deterministic two-dimensional rows with reversible coordinate changes and three row permutations per block (below 5%). Replacements receive no ancestor evidence. After 20 blocks it has 49/61 qualifying current rows. In contrast, first-plus-second bounded population moment flux gives ten consecutive-pair cosines exactly −1. Permutations preserve population observables: maximum measured reaction flux 2.8e−17, flux-accounting error 1.8e−18. This demonstrates a population observable can remain measurable through churn; it does not show RA7 has this property.

First moments alone miss symmetric spreading. The fixed [-1,+1]→[-2,+2] example leaves its first moment zero while bounded second moment changes. Even both moments do not identify an arbitrary distribution.

## Smallest reserved law

Fix bounded observables before testing, such as tanh of each latent coordinate and its square with an immutable initial normalization. For every update, record the full-population observable mean before the coordinate update, after that update, and after all reactions. Coordinate flux plus reaction flux exactly equals total change. Historical statistics describe past populations; no newborn receives a parent's own evidence. The approach costs O(Nd) work and O(d) extra state for a direct implementation. Fixed projection could reduce the observable dimension only under a separately declared law.

A bounded conditional-mean law can avoid Gaussian variable-cohort assumptions: for cosine X in [-1,1] and the declared non-reversion null E[X|past]≥0, each predictable factor 1−lambda*X (0≤lambda≤1) has conditional expectation≤1. Nonnegative products and fixed mixtures therefore permit an anytime threshold. A fixed four-lambda illustration crosses 80 after eight all-negative pairs. Exact enumeration of 4096 fair-null sequences gives crossing probability 1/256 and terminal expected wealth 1. This algebra assumes its explicit conditional null; it does not prove full-density equilibrium. The supermartingale basis is described in [Howard et al.](https://arxiv.org/abs/1810.08240).

A production law would still need prespecified scale/epoch error allocation, finite-state log wealth, zero-motion handling, chart/normalization identity, drift and reaction expiry, strict checkpoint versioning and atomic old-law rejection. Existing 24-block resets cannot inherit an infinite-horizon claim. Saved checkpoints do not contain separate coordinate/reaction flux histories, so no retrospective stationarity verdict is available. Current support counts establish discoveries under their conditional assumptions, not equivalence.

## Next candidate

Root selected a smaller change: empirical current paired-average eligibility from EMA support and same real topology group. It leaves training and the population optimizer law unchanged. That guard is an anti-blur control, not a stationarity, distribution-equivalence or quality certificate. Fresh chart provenance, postreaction counts, turnover/TTL expiry, immutable semantic state and atomic checkpoint rejection require independent review. Population flux remains reserved. Original RA7 toy FAIL and unmet toy-plus-Grid100 target are preserved.

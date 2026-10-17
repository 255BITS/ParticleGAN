# RA6 table participation after the completed toy failure

RA6 toy remains FAIL: emitted precision .516968,23/25 modes,TV .483032. The saved artifact audit is VALID. This report uses saved parameter-displacement vectors and gradient sufficient statistics only.

## What the coverage count measures

The525-row rejection at904 tested b32 and changed b to64. It was a negative **average displacement-direction** test over a surviving subset, followed by a population participation veto. Participation requires two finite cosine observations for the same current row incarnation at the negative scale. It is not a count of sampled indices, gradient updates, independently stationary rows, or G/D convergence. The rejected decision clears its observation arrays; the525 and earlier599 row identities are no longer saved.

Each b cosine uses two complete blocks; two such observations need at least four clean b blocks. A2b cosine uses four blocks; two need eight clean b blocks. Copy and novel birth rebase keeps the clock, erases the moved row in **every previously completed pair**, and masks its unfinished block. Future gradients do not repair an erased old pair. More sampling therefore cannot preserve evidence for repeatedly replaced incarnations.

## The clock ran; participation and evidence were insufficient

| Step | b | Completed blocks since904 | Finite-pair participants b /2b | Moves cumulative | Median touches since reset |
|---|---:|---:|---:|---:|---:|
| 1000 | 64 | 1 | 0 / 0 | 6039 | 12 |
| 1250 | 64 | 5 | 143 / 0 | 7404 | 14 |
| 1500 | 64 | 9 | 133 / 38 | 8931 | 12 |
| 1750 | 64 | 13 | 105 / 25 | 10449 | 11 |
| 2000 | 64 | 17 | 204 / 53 | 11802 | 12 |

All these checkpoints have table s1, LR .0085, zero excluded rows, no hold, and no whole-table restart. At2000,17×64+8=1096 intrinsic updates exactly equals2000−904. The full24-block window would first finish at2440 if s stays1. At the latest early look1928, log Bayes factors −.70454/.62499 were below log80=4.38203, so no new negative direction decision occurred. This is not a stalled clock or another coverage rejection at b64.

Saved evidence directly demonstrates lineage turnover: the143 eligible b-row incarnations at1250 retain only3 voters in those same original pairs at2000;140 were invalidated by rebase. At final, only204 rows have two b observations and53 have two2b observations, versus973 required. Their finite sets are nested; participation equals the final two clean-pair intersection at each scale.82 rows have zero gradient touches since the latest reset. Median touches are12 overall,52 among b participants,86 among2b participants. Current-mask identities and observation histograms are retained in observations.json.

## Sampling reference and continued transport

Uniform sampling of128 indices from1024 gives each row touch probability .117557 per update. Without replacement, a64-step block has touch probability .999666; ordinary sampling by itself provides ample opportunities. W=50(1−.98^touches) recovers the **actual** nonzero-gradient touch count to numerical precision. The all-row median12 is far below the approximately235 expected sampled touches over2000 updates without resets. Sparse sampling also causes some missing pairs at shorter b, but cannot explain this final cohort loss.

The actual second half makes5763 moves over125 evaluations,46.104 moves per8-step evaluation on average. Under an illustrative uniform replacement assumption, that corresponds to a mean incarnation lifetime about178 updates and21 gradient touches. Under the maximum51-per8 rate, survival across the four64-step blocks needed for two b pairs is about.195; across eight blocks it is about.038. A95% survival rate across four blocks would allow only about1.64 uniformly spread replacements per8, not51. These formulas are descriptive assumptions; the actual count-guided donors are not uniform, and no simulation or statistical level was fitted.

It is mathematically possible to certify during continuing moves if the moves stay within at most51 rows and at least973 other incarnations remain unchanged and gather negative-direction evidence. It is also possible after sufficiently few rows are replaced. The observed broadly changing table does not meet that condition. Doubling b after a coverage rejection increases the lineage lifetime needed by the next test; it does not cure churn. This explains the scheduler's lack of a whole-table certificate, without showing that lowering the coverage gate would be valid.

## Separate optimizer and row-evidence limitations

G/D/prior parameters remain trainable at every endpoint. controller.closed is already true at500 and is a mobility diagnostic; under stationarity control it does not freeze optimizers. Final G LR is .00006640625, prior LR .0085, actual D LR about.0031875 due the existing .75 prior-rate floor. D's own small s is not its applied LR. The speed change after1072 therefore is not explained by a newly closed optimizer in the saved state.

RowEvidence is a separate touch-gradient test. Its effective sample cap99 is below3×128=384; every checkpoint has zero mature rows and no flags, so it cannot provide the intended hot-row/exclusion/hold signal. Even an infinite untouched run cannot mature this configured full128-dimensional test. A projection would need its own independently specified subspace, sufficient statistics and null interpretation; existing scalar norms cannot justify reusing p-values in a smaller rank.

No production change is qualified by this report. Preserve the population continuity veto and unchanged serving/gates. Any next transport scheduling or gradient-subspace mechanism requires a fresh declared law and prospective toy plus original Grid100 validation. An EMA serving override would not establish current-population stationarity or repair fast training drift.

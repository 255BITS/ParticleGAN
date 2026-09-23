# Shared-default search: first results

**Best complete score: 15/19. No all-pass shared default yet.** The public
presets remain 8/19 proposed and 5/19 current master under this same protocol.
The search has not changed the package defaults.

| Shared recipe | Required | Data | Images | Overall |
| --- | ---: | ---: | ---: | --- |
| `lr00425_prior2`: cap coefficient 3 | 8/9 | 3/6 | 4/4 | 15/19, FAIL |
| `shared_c6`: cap coefficient 6 | 9/9 | 2/6 | 4/4 | 15/19, FAIL |

Both use Rp logistic, κ1.25, spread .05, no particle L2, Adam(0,.99),
G/D LR .00425 and particle LR .0085 on **every** test. Both retain the common
delayed cosine schedule. The cap coefficient is the only difference between
these two complete recipes. [Exact candidate cards](leading_candidates.json).

The coefficient-3 candidate fails mode-hold, rare mass, unequal width and
overlap. Rare/unequal-width failures are severe within-component flattening:
minimum normalized variance is about .000051 and .00486, respectively, against
the .15 floor. Overlap meets final metrics but has only three final passing
observations instead of five. Mode-hold ends with seven of eight modes.

Coefficient 6 fixes mode-hold and reaches 9/9 required tests, but loses the
anisotropic data pass. Rare mass, unequal width and overlap still fail. These
two rows cannot be pooled to claim a better shared score.

## Search evidence

- Six global-rate recipes ran all 19 cases: 114 new episodes.
- Twelve global refinements screened six difficult cases: 72 episodes.
- Three screened recipes ran their remaining thirteen cases unchanged: 39
  episodes. Their complete scores are 15/19 (`shared_c6`), 14/19 (`shared_k075`)
  and 13/19 (`shared_spread01`). Screening improvements regressed other cases.
- Total: **225 new training episodes**, plus 38 existing baseline episodes.
  Nine new candidates are complete; nine remain explicitly incomplete.
  Every failure, curve, applied optimizer group, action trace and source archive
  is retained. The importer recomputes all verdicts and refuses mixed recipes.

The variants covered common learning rates, particle-rate ratios, cap strengths
and targets, Adam betas and reduced spread weight. This bounded search does not
establish that one shared default is impossible. It demonstrates that stronger
full-suite baselines exist and identifies the remaining tradeoffs.

## Useful next targets

Focus on preserving rare/unequal-width component variance while retaining the
required movement/identity tests. A mechanism must use one shared rule, not a
lookup keyed by the example or its evaluation score. Discriminator changes
remain architecture trials under the same formulation; record those explicitly.
Keep the current reference architecture scores available for comparison.

Promote a screened improvement only after the remaining cases are measured.
The coefficient-6 result is a concrete example: solving mode-hold alone would
have hidden its anisotropic regression.

```bash
python -u -m benchmarks.transfer_suite.shared_default_search \
  --plan reports/transfer_suite/unadjusted/leading_candidates.json \
  --output /tmp/shared-leaders-replay > /tmp/shared-leaders-replay.log 2>&1
tail -f /tmp/shared-leaders-replay.log
```

[Primary leaderboard](README.md) · [Contributor instructions](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md).

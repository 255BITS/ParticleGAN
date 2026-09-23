# Shared-default search results

**Best supported score: 18/19 with one unchanged recipe and explicit discriminator
choices, up from 17/19. Rare-component variance is the only remaining failure.**
The same recipe scores 15/19 with the reference discriminator profile. Public
presets remain 8/19 proposed and 5/19 old; this search does not change package defaults.

| Shared recipe | Required | Data | Images | Live total | Reference D profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_c6`, including declared D trials | 9/9 | 5/6 | 4/4 | **18/19** | 15/19 |
| `lr00425_prior2`, cap coefficient 3 | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| `ratio_g34_d68_p85` | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| `relative_cap_05`, adaptive step limiter | 7/9 | 2/6 | 3/4 | 12/19 | 12/19 |

The leader uses Rp logistic, b_cap coefficient **6**, κ **1.25**, particle spread
**.05**, no particle L2, Adam **(0,.99)**, and absolute **G/D LR .00425, particle
LR .0085 on every test**. Hold rates for the first 60% of the budget, then cosine
toward 5%. No optimizer setting or objective is adjusted by example.

## The new width pass

A raw-input discriminator with **128 units in each of three hidden layers and
Softplus β=8** passes all five behavioral metrics at the last five observations,
steps 1000, 1050, 1100, 1150 and 1200. It has 33,537 parameters, no Fourier features,
no extra skip head and no output scaling. Here β controls activation sharpness;
Adam's betas remain `(0,.99)`.

| Metric | Worst of the final five observations | Required |
| --- | ---: | ---: |
| Minimum normalized component variance | .25617 | ≥ .15 |
| Component covariance error | .62818 | ≤ .85 |
| High-quality sample fraction | .93042 | ≥ .85 |
| Mixture mass error | .03540 | ≤ .15 |
| Normalized sliced distance | .09772 | ≤ .18 |

The otherwise matching β=5 discriminator passes only its last observation. This
paired comparison supports the sharper activation for this case; it does not
establish a general mechanism or unseen-task transfer. EMA also passes with β=8,
but contributes no selection points. An independent replay matches every
live/EMA checkpoint, action, optimizer receipt and verdict exactly.

The width winner **fails rare-component variance**, ending at .00302 versus .15.
That failure is retained alongside its successful case. [All 35 width-study
runs](runs/shared-width-search/README.md) · [Independent replays](runs/remaining-replays/README.md).

## Selected architecture support

| Case | Supported discriminator | Final passing observations |
| --- | --- | ---: |
| Broad mixture and spiral | Original reference discriminators | 20 and 23 |
| Anisotropic | Raw MLP plus small Fourier branch, 64×2, Softplus | 8 |
| Overlap | Raw-input MLP, 96×3, Softplus β5 | 10 |
| Unequal widths | Raw-input MLP, 128×3, Softplus β8 | 5 |
| Rare component | None found | 0 |

Architecture is separate from recipe identity. **This is not one universal
network architecture.** The leaderboard shows both architecture-supported and
reference-profile scores and retains every failed architecture trial. Required
and image cases still use their original discriminators.

## Remaining failure: rare-component spread

The strongest final result in this round is a raw-input Softplus96×3 critic with
LayerNorm after each hidden linear layer and activation β4. Its final minimum
normalized variance is **.12525**, below **.15**, with zero final passing
observations. Its other final bounds pass; the full late curve still determines
the verdict. All five late variance readings are below .15 (.0310–.1253);
step 1100 also fails sample quality and covariance. This failure reproduces
exactly in an independent replay. It improves the final variance statistic over the previous raw-SiLU
near miss (.01891), but supplies no additional behavioral PASS.

This case is demonstrably passable with the lower-rate public preset. These
negative architecture trials do not show that ParticleGAN cannot solve it.
They show that this shared high-rate recipe still lacks a passing architecture
in the recorded search. A future global schedule or update-rule change must
compete as its own unchanged full-suite recipe, preserving all other successes.

## This round and retained evidence

Three agents, including GPT-6 Sol at max reasoning, ran **127 new candidate
episodes** with the exact shared-cap6 recipe:

| Study | Episodes | New sustained live passes | Finding |
| --- | ---: | ---: | --- |
| Width/depth, activation, skip and score parameterizations | 35 | 1 | β8 fixes unequal widths; rare cross fails |
| Local radial features, quadratic experts and multiplicative models | 44 | 0 | Local features do not preserve rare variance at these rates |
| Ensembles and pointwise normalization | 48 | 0 | LayerNorm improves a final rare statistic, but no sustained pass |

[Width evidence](runs/shared-width-search/README.md) ·
[Local-feature evidence](runs/shared-local-density-search/README.md) ·
[Ensemble evidence](runs/shared-ensemble-search/README.md) ·
[Normalization evidence](runs/shared-pointnorm-search/README.md).

All complete curves, failures, exact source archives and actual optimizer groups
are retained. Data, generators, initialization rules, particle counts, batch
sizes, original budgets, thresholds and auxiliary objectives stay fixed. No seed
sweeps or longer-budget replacements. EMA remains separate. These are inspected
development tests, not unseen holdouts.

The primary importer now covers **515 episodes across 29 recipe entries**.
Fourteen entries cover all 19 tests; fifteen partial entries cannot win. The
previous 125-run rate/adapter/architecture round remains intact: rate finalists
scored 15/19 and 13/19, while the generic Adam proposal limiter scored 12/19.
[Rate study](runs/shared-ratio-search/README.md) ·
[Adapter negative result](runs/shared-adapter-search/FINDINGS.md).

**88 focused tests pass**, alongside three exact independent replays and nine
verifier mutation checks. The complete 515-episode import reports no errors.

Independent review hardened replay validation: it requires every numerical
comparison field and a complete, nonempty source manifest. The primary importer
also checks the exact source-manifest coverage. Nine mutation checks reject
missing or changed evidence while accepting timing-only changes.
[Validation audit](runs/remaining-verifier-audit/README.md).

## Reproduce

The [leader recipe cards](leading_candidates.json) reproduce the original
19-case reference profiles. The [earlier D plan](leading_discriminator_trials.json)
reproduces anisotropic and overlap support. Reproduce both the new width pass
and its rare cross-failure with:

```sh
python -u -m benchmarks.transfer_suite.shared_width_last_refinement \
  --plan reports/transfer_suite/unadjusted/leading_width_trials.json \
  --output /tmp/shared-width-witness > /tmp/shared-width-witness.log 2>&1
tail -f /tmp/shared-width-witness.log
```

[Primary leaderboard](README.md) · [Contributor instructions](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md).

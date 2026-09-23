# One shared relative-step adapter: results

**Selected candidate `relative_cap_05` sustains 12/19, versus15/19 for the unchanged Adam baseline.** This is an altered optimizer mechanism applied with one equation and one scalar fraction everywhere. There is no overall PASS unless all19live tests pass. No production default changes.

The baseline remains G/D LR.00425, particleLR.0085, Adam(0,.99), Rp logistic, b_cap3/κ1.25, spread.05, no particleL2, delayed cosinehold60%/floor5%. All data, auxiliary objectives, architecture profile, particle counts, batches and budgets are unchanged. Seed0only.

Each ordinary Adam proposal delta is multiplied by min(1, fraction×max(RMS(parameter_before),.1)/(RMS(delta)+1e−12)). The fraction is the only candidate knob; .01,.025,.05 were declared before results. Adam moments receive unchanged raw gradients. The equation never receives role names, task names, target labels, metrics or time; roles only label logs. Its interaction with the unchanged schedule occurs through the Adam proposal.

| Candidate | Screen live /6 | Attempted /19 | Full live /19 |
| --- | ---: | ---: | --- |
| relative_cap_01 | 1/6 | 6/19 | INCOMPLETE |
| relative_cap_025 | 1/6 | 6/19 | INCOMPLETE |
| relative_cap_05 | 2/6 | 19/19 | 12/19 |

Selection was frozen: screen pass count, then mean final normalized bound shortfall, then candidate name. Exactly one candidate completes the remaining13tasks without further tuning. Incomplete candidates cannot claim19-task coverage.

| Task | Baseline | Selected live | Final suffix | Selected EMA |
| --- | --- | --- | ---: | --- |
| ae_gan_hold | PASS | PASS | 22 | N/A |
| img_bars4 | PASS | PASS | 9 | FAIL |
| mode_hold | FAIL | FAIL | 0 | N/A |
| vector_unequal_mass | FAIL | FAIL | 0 | FAIL |
| vector_unequal_width | FAIL | FAIL | 0 | FAIL |
| vector_overlap | FAIL | FAIL | 2 | PASS |
| two_pole | PASS | PASS | 11 | N/A |
| trajectory | PASS | FAIL | 0 | N/A |
| residual_student | PASS | PASS | 22 | N/A |
| unipolar | PASS | PASS | 18 | N/A |
| cover_leftover | PASS | PASS | 14 | N/A |
| unused_token_hold | PASS | PASS | 13 | N/A |
| mid_scale_identity | PASS | PASS | 17 | N/A |
| vector_two_broad | PASS | PASS | 17 | PASS |
| vector_anisotropic | PASS | FAIL | 1 | FAIL |
| vector_spiral | PASS | PASS | 24 | PASS |
| img_stripes2 | PASS | FAIL | 4 | PASS |
| img_blobs4 | PASS | PASS | 7 | FAIL |
| img_intensity2 | PASS | PASS | 8 | PASS |

## Measured attenuation and overhead

| Reported role | Tensor-updates attenuated | Mean factor | Minimum factor |
| --- | ---: | ---: | ---: |
| d | 2.23% | 0.9955 | 0.16388 |
| g | 5.56% | 0.9889 | 0.18959 |
| prior | 3.87% | 0.9896 | 0.30166 |

Selected full-suite wall time sums to 108.18s; measured adaptation/instrumentation overhead is 8.19s (7.57%). This includes cloning, norm calculation and tracing, and excludes ordinary Adam.step time. It is a concurrent CPU measurement, not a production performance comparison. Role aggregates weight each tensor-update equally; they are descriptive, never inputs to the rule.

## Validation and retained failures

All3final identity controls match archived24live/EMA observations, actions, recipes, actualgroupLRs/betas and specs exactly apart from runtime timestamps. Two initial control attempts stopped on reporting comparisons (tuple/list serialization, then created_at); their numerics also match exactly after canonicalization. They remain intact with original source archives and failure logs. No numerical training error occurred.

All38completed episodes are retained:31new candidate episodes and7identity control episodes across attempts. Four analytical contract tests verify identity parity, exact capped direction/relativebound, unchangedAdam moments, and role independence; invalid cards failclosed. Source snapshots, all raw payload hashes, recomputed live verdicts and all traced relative-stepbounds validate. EMA is separate.

The source includes a reusable mechanism and frozen study CLI; importing the rows must preserve the mechanism card alongside the base recipe. Do not label these rows plainAdam or combine their passes with another recipe. The bounded negative or mixed result does not rule out other generic adapters.

```bash
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_adapter_search \
  --output /tmp/shared-adapter-replay > /tmp/shared-adapter-replay.log 2>&1
```

[Incremental leaderboard](study/README.md) · [All episodes and settings](study/index.json.gz) · [Frozen selection](study/selection.json.gz) · [Exact sources](study/source.tar.gz) · [Raw tensor traces](study/episodes/) · [Artifact manifest](archive_manifest.json).

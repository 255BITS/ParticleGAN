# Cross-family smoke representation admission

All three fixed smoke questions have **SUPPORTED numerical capacity** for each
of the five prepared legacy families: R1/R2, BCap, K3P, KA2 and released v0.7 MoG.
The [receipt](legacy-smoke-admission.json) covers 15 family/task cells and all
28 configuration IDs. There are **zero optimizer and fitting updates** and no
ordinary qualification or acquisition-time credit.

The input is the retained
`toy-forge-family-round1-bcap-representation-20261002-cloud-bound` capacity archive.
Its exact receipt, task/source bindings, saved parameters and numerical artifacts
are verified before reuse. Only parameter states are restored: each family's
actual public BehaviorComponents constructs new public optimizers and named
streams. Every family's configuration cards resolve to the same relevant
host/serving fields within that family; differences in optimizer rates/penalties
remain bound to each complete card. No trained optimizer state crosses families.

| Smoke host | Constructed capacity and exact gate | Result |
| --- | --- | --- |
| two_pole | The actual 12-row direct cloud matches the balanced fixed pole spread; a learned constant critic has zero median slope. `mean_abs=1 >= .3`, `grad_med=0 <= 1`. | SUPPORTED; does not prove travel from the required zero initialization |
| unused_token_hold | Per-slot correction realizes the concept displacement while preserving the unused token and scale zero. `concept_move=unused_hold=1 >= .85`. | SUPPORTED; does not prove learning the correction |
| ae_gan_hold | Actual 32-wide MLP query encoder is identity with zero offset, decoder is identity, and six learned MoG means sit on each anchor. Full public AE routing and prior draws are evaluated. | SUPPORTED at reconstruction/hold tolerances, with each family's actual serving noise |

Each immutable state is evaluated at all 24 original schedule labels. The
two-pole measurement and token measurement explicitly omit training noise, as
their task declares. AE evaluates the actual scheduled output-noise wrapper:

| Actual family serving cohort | Max reconstruction MSE / .05 | Max hold distance / .35 |
| --- | ---: | ---: |
| R1/R2 and released v0.7, output sigma 0 | .002552493243 | .001155285863 |
| BCap, K3P and KA2, output sigma up to .029 | .003351714462 | .003005400766 |

These capacity observations are source-bound and explicit separate serving
cohorts, never pooled training evidence. Origin-cloud, absent-concept and
collapsed-AE controls fail their original numerical bounds. Evaluation preserves
model/prior/optimizer state and has zero unintended RNG deviations. The 24
schedule labels describe evaluations of the same analytic state, rather than
training progress, sustained learned convergence or independent attempts.

The certificate supports the declared **gate tolerance**. The AE decoder's
unconditional kernel width remains .025 before added output noise, whereas the
target data width is .05; the original smoke gate does not test Gaussian shape.
Likewise, two-pole's movement/slope gate does not certify both-pole mass by itself.
No broader distribution-quality claim follows from these passes.

```sh
python reports/forge/family-winner-round1/legacy_smoke_admission.py \
  --witnesses /path/to/restored/zero-update-capacity-archive \
  --output /tmp/legacy-smoke-admission.json
```

The full required-suite representation judgment remains **UNRESOLVED** until
the separately bound remaining host witnesses are joined. Scientific training
outcomes, failed prerequisites, calibration and default adoption remain separate.

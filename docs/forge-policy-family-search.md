# Bounded policy-family common-knob search

This is a separate `policy-family-defaults` cohort for the actual public Atlas
and E22 selected/served particle-cloud API. It supplies no current Forge MoG
qualification, all-toy winner, calibrated default adoption or speed winner.
Historical Atlas19 results remain their original evidence; they are not rerun or
pooled into this study.

The [completed campaign](../reports/forge/family-winner-round1/README.md) records
all four grids. The [primary policy board](../reports/forge/policy-family-inventory.md)
and [machine-readable JSON](../reports/forge/policy-family-inventory.json) are the
common team record for whole configurations, gates, unknown requirements,
costs and goal GIFs. No configuration has qualified every required case.

New automated studies use the [shared policy execution coordinator](forge-policy-execution.md).
It freezes source, deduplicates compatible studies and overlapping physical
attempts, and shares resource admission with ordinary Forge workers. All
cooperating submitters must use the same queue root. Its independently
supervised deadlines and inherited leases bound children after a caller crash;
retained originals can resume certification without training again.

To test one separately declared future config through the public API, use its
shared Recipe overrides and a fresh archive directory. This example shows the
observed original broad-mixture PASS setting. A reproduction records a new
source/runtime cohort; no reproduction was executed for this guide. The command
runs the full original case and returns exit0 for PASS or exit1 for FAIL; its
receipt also distinguishes incomplete or invalid execution. This standalone
`api_run` command does not register a policy study in the shared admission
ledger; automated research should use the coordinator below:

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.toy_audit.api_run \
  --case api-vector-two-broad --recipe e22 \
  --recipe-overrides '{"lr": 0.0053125, "prior_lr_mult": 1.5}' \
  --device cuda:0 --wall-cap-seconds 180 --frames 9 \
  --output /path/to/fresh-candidate-archive
```

Set the assigned GPU and declared numeric values for the intended study. Use the family
coordinator below to enforce shared knobs across all eight required cases and
the additional acquisition/hold gate. A single case PASS is not that whole-config
gate. Commands in this guide describe the interface; completed failed settings
are not automatically reexecuted.

Publish from certified combined receipts with the observation-only exporter:

```sh
python -m experiments.forge.policy_publication \
  --combined /path/to/archive/current-policy-family-results.json \
  --all-media --review-projection
```

Repeat `--combined` for each retained compatible source cohort. Publication
updates the one primary board and copies every reached original GIF; it keeps
separate reviewed media provenance and runs no training or model-metric rescoring.
Commit the compact board, receipts, checked archive manifests and media to
`develop`. Bulk arrays, checkpoints and full logs remain at the manifest paths;
team members need that archive or shared-machine access for independent regrading.
Resolve those exact originals with the [artifact resolver](forge-artifact-resolver.md).
An inaccessible manifest location remains a missing-evidence result. For a
local display readout, use:

```sh
python -m experiments.forge.policy_family_readout /path/to/archive/study.json \
  --output runs/forge/local-policy-readout
```

The dated scripts under `reports/forge/family-winner-round1/` remain frozen
reproduction sources; maintained publication and readout code lives in the
experiment package. Neither command grants qualification or starts training.

Each source cohort freezes one finite grid: the first used `lr` .006375 or
.0085; the linked second uses the public preset rate .00425 or half that rate
.002125. The third freezes intermediate rates .0031875 and .0053125. Those
three profiles cross their rates with `prior_lr_mult` 1 or 2. The final profile
binds LR .0053125 or .006375 specifically to prior multiplier .5 or 1.5.
These are the only accepted complete grid pairings; combining axes borrowed
from different profiles is rejected. A different hypothesis needs an explicit
reviewed revision.
Each of the four configurations per family uses the same two knobs on every
required case. The original host architecture, prior/init, data law, batch,
evaluation draw count, scoring cadence and full horizon remain unchanged.
Existing fixture adaptations, including vector optimizer betas, are disclosed
under `fixed_host_recipe_options`; the exact complete resolved Recipe is bound
and checked. This is a common-knob search within those frozen adaptations, not
proof that one unadapted public Recipe should ship for every host.

| Order | API case | Tier | Updates | Batch | Evaluation samples |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `image-develop-img_intensity2-source-transpose12` | smoke | 600 | 32 | 1,024 |
| 2 | `api-vector-two-broad` | smoke | 1,200 | 128 | 4,096 |
| 3 | `api-grid100` | quality | 7,000 | 2,048 | 20,000 |
| 4 | `api-rotated100` | quality | 7,000 | 2,048 | 20,000 |
| 5 | `api-staggered100` | quality | 7,000 | 2,048 | 20,000 |
| 6 | `api-vector-unequal-mass` | quality | 1,200 | 128 | 4,096 |
| 7 | `api-vector-anisotropic` | quality | 1,200 | 128 | 4,096 |
| 8 | `image-develop-img_bars4-source-transpose12` | quality | 600 | 32 | 1,024 |

All cases retain their original numeric predicates and 24 post-update primary
scoring observations. The study additionally requires the first five consecutive
passing primary observations to confirm acquisition, followed by at least five
additional passing hold observations. Every later primary observation must pass.
There is no later restart window after collapse. Initial or GIF-only observations
do not count. A full run acquired too late for five hold checks is `INCOMPLETE`.
The original per-case verdict remains separate; an original PASS can fail this
study's retention rule. The original GIF footer reports that original verdict;
the published study board must also show the separate study verdict.

Before training, every family/case needs a source-, preset-, host-, sampling- and
artifact-bound capacity card. The planner restores its public state on CPU,
checks the original full horizon and zero optimizer-update clock, reproduces the
same-seed actual served draws, rederives their original numeric gate and verifies
state/RNG parity. This is a necessary finite snapshot-capacity witness. It does
not demonstrate learning, stable adaptive geometry or a GPU trained PASS. Prior
failed constructions remain separate. Conditional/caller-owned and named stress
tuning contracts are unsupported and explicitly BLOCKED.

The preregistration JSON contains `schema: particlegan_policy_family_search_v1`,
an immutable study `id`, families `[atlas,e22]`, seed24002, the exact grid and case
order above, per-case `timeout_seconds`, `export_grace_seconds`,
`candidate_budget_seconds`, `budget_seconds`, `representation_card` path/SHA,
backend, frame count and the frozen stability rule. `speed_ranking` and
`default_adoption` must both be false. Optional `family_budget_seconds` declares
disjoint Atlas/E22 paid quotas whose sum cannot exceed the round cap; otherwise
each gets half. Worst-case reservations are reported separately from the paid cap.

Planning starts no optimizer updates, queues, calibration or default promotion:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.toy_audit.api_family_search \
  plan reports/forge/family-winner-round1/policy-search.json --output /tmp/policy-plan.json
```

The completed campaign's second distinct specification is
[`policy-search-round2.json`](../reports/forge/family-winner-round1/policy-search-round2.json).
Its baseline/half-rate hypothesis follows measured higher-rate smoke failures;
it repeated no failed configuration. Its 10,525.043357 seconds are the
original 10,800-second allowance less the original setup and completed first
grid, rather than a fresh allowance. The linked parent receipt and exact
per-family debits stay in the specification. Source cohorts remain separate.

The [third specification](../reports/forge/family-winner-round1/policy-search-round3.json)
then tests one intermediate-rate acquisition/retention hypothesis. The second
grid's seven original image passes supplied no study pass: four collapsed after
confirmation, three acquired too late; its other image run failed the original
gate. The next grid receives only 10,336.011929 unspent seconds from the same
cap. Its four knob pairs are new, with all original cases, gates and horizons
unchanged. Nonmonotonic prior results do not guarantee an intermediate rate wins.

The [fourth and final specification](../reports/forge/family-winner-round1/policy-search-round4.json)
tests a shared prior-rate balance at rates that previously retained intensity.
This is a lower or intermediate table-rate ceiling; actual transport also
depends on the policy, row evidence and birth/death. It does not establish a
prior optimizer cause or promise slower actual drift. All four knob pairs are
new. The first three grids and setup leave exactly 9,997.188535 seconds from
the original cap. This concludes the finite screening search; failures do not
automatically start more local tuning, a seed study or default promotion.

After one common code/spec/capacity freeze, the coordinator can admit one serial
family lane per visible GPU. Each lane must use a separate archive and fixed
runtime/device model. The exact child command uses the existing `api_run` public
API path with the full default protocol; it never passes shortened `--steps` or
`--eval-samples` values or creates extra workers:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.toy_audit.api_family_search run \
  reports/forge/family-winner-round1/policy-search.json \
  --family atlas --device cuda:0 --output /path/to/archive/atlas
```

Use the other assigned visible GPU and `--family e22` for its separate lane.
The coordinator refuses admission while another cooperating runner owns that
physical slot. Set `--queue-root /path/to/shared/runs/forge` when overriding the
default main-repository ledger; use the same location for core and policy work.
Tail `<archive>/<configuration-id>/<case-id>.log`. Actual arrays, checkpoints,
GIFs and full logs stay in the archive. The compact `study.json` retains recipe,
source/runtime, costs, original/study gates and exact artifact identities.

A config stops at its first non-PASS. Later required cases remain UNKNOWN, not
zeroes or removed denominators. The coordinator reserves a complete next task
allowance before starting; a hard subprocess cap includes a separate media-export
grace, and an acquisition cap cannot become PASS even if a child exits0.
Registration precedes the child, and each result is saved durably. A busy
coordinator returns its canonical attachment or `coordinator.waiting_reason`;
only a later explicit `run` resumes unlaunched work. An interrupted
or orphan paid attempt is retained as INCOMPLETE and never automatically retried.
The child command is checked through the real public CLI parser in software
controls. Its seed comes from the case's fixed protocol, which preflight must
match to the study; this CLI has no caller-supplied seed option. An execution
repair requires a linked new study/source/archive, retaining the original error
and subtracting its setup cost from the total remaining paid allowance.
Unknown interruption cost is conservatively charged at its reservation ceiling;
`measured_paid_seconds` reports only measured child execution separately.
Metric/sample/checkpoint/source/recipe/runtime/exit consistency is rechecked
before granting qualification or combining archives. Every completed original
receipt is rechecked, including an original PASS whose additional study hold
is INCOMPLETE. Saved paid cost cannot be below its bound acquisition time.

```sh
python -m benchmarks.toy_audit.api_family_search combine \
  /path/to/archive/atlas/study.json --archive /path/to/archive/e22/study.json \
  --output /path/to/archive/current-policy-family-results.json
```

The combination rejects different source/spec/case/runtime-version cohorts.
It retains all eight configurations and all eight case denominators, every
fully qualified config and any ties. Incomplete comparisons cannot name a final
winner; a deterministic pass-count display is only best observed evidence.
Monotonic acquisition timestamps synchronize CUDA at observations and include
initialization, updates and evaluation; they exclude queue wait and post-run
export. They are diagnostic under external contention and different GPU models.
No fastest config is selected from them. The screen remains provisional until
separate calibration, confirmation and reserved robustness evidence justify any
public-default decision.

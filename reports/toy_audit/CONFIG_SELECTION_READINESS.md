# Selecting a config with Forge's PR228 requirements

**The selection machinery works, but no current config clears the declared
requirements.** The read-only selector exposes qualified config options and
their recipe/source/runtime identities. Its frozen local snapshot reports
`NO_QUALIFIED_OPTION` for both useful starting views. It selects no winner and
changes no default.

PR228, merged as `3f02957a`, added the
[experiments-by-tier inventory](../forge/EXPERIMENTS_BY_TIER.md). It lists
requirements; it did not add an aggregate quality metric or new scientific
acceptance criteria. The actual predicates remain in Forge's validated views,
independent evaluators, compatible-receipt board, calibration verifier and
separate promotion protocol.

From the repository root:

```sh
# Defaults to quality_coverage. Omit backend to inspect all exact runtime cohorts.
python -m benchmarks.toy_audit.forge_selection_readiness
python -m benchmarks.toy_audit.forge_selection_readiness \
  --view discriminator_stability --backend cpu
python -m benchmarks.toy_audit.forge_selection_readiness \
  --view quality_coverage --backend cpu --candidate atlas

# Explicit optional publication to a new file; ordinary stdout mode writes nothing.
python -m benchmarks.toy_audit.forge_selection_readiness \
  --view quality_coverage --output /tmp/forge-quality-new.json
```

Exit 0 means at least one current screen-qualified option and no unresolved
board conflicts. Exit 1 means no option or conflicts requiring review. Exit 2
means an invalid request or changed identity. A screen option can still have
`calibration_verified: false`: exit 0 does not establish calibrated selection
or default adoption. Multiple options remain separate; the tool never chooses
one by name, wall time, or an undeclared weighted score.

## What must pass

| Decision | Existing Forge requirement | What the selector reports |
| --- | --- | --- |
| Provisional screen qualification | Every required task in the selected view passes under compatible source, recipe, prior, initialization, full budget, sampling, RNG and runtime | `qualified_options`; actual board qualification, with full required denominators |
| Calibrated config evidence | Independently verified accepted calibration covers that exact scientific cohort and required smoke set | `calibrated_options`; a changed status flag alone is insufficient |
| Default adoption | Concluded exact qualification readout, accepted calibration, and separately registered and passing frozen robustness stage with controls | `NOT_ASSESSED_REQUIRES_SEPARATE_REGISTERED_ROBUSTNESS`; no registration or promotion happens here |

The [frozen calibration criteria](../../configs/forge/calibration/criteria-v1.json)
require at least 3 paired lineages, at least 1 independent positive and 2
independent negatives, paired fraction at least 0.9, false rejection fraction 0,
false acceptance fraction at most 0.1, complete smoke/reference costs, maximum
smoke cost 900 seconds, and maximum smoke/reference cost ratio 0.1 within the
declared prior/runtime cohort. Missing or blocked reference evidence remains
unknown and blocks adoption; it does not become a scientific failure. Current
profiles remain provisional or blocked. This command does not fill a matrix,
run a calibration, change thresholds or authorize training.

The required denominators depend on the scientific question:

| View | Smoke | Quality | Endurance | Separate diagnostics |
| --- | ---: | ---: | ---: | ---: |
| quality_coverage | 3 | 19 | 0 | 0 |
| discriminator_stability | 3 | 19 | 2 | 0 |
| adaptation | 3 | 19 | 1 | 0 |
| clockfree_continuous | 4 | 19 | 6 | 0 |
| formulation_comparison | 3 | 19 | 2 | 15 |
| host_profile_transfer | 3 | 19 | 2 | 13 |

The selector includes PR228's source-bound task roles, evaluator kinds,
declared budgets, dependencies and shared execution groups. Backend/candidate
filters select whole rows and preserve every required task. Diagnostics and
ranking tasks retain their original roles. Archived, pinned and calibration
results cannot supply current qualification cells; CPU/CUDA and hardware
cohorts do not pool.

## Frozen local readout

[config-selection-readiness.json](config-selection-readiness.json) is a compact
snapshot generated after committing the selector at
`febe46c6e0a7b21f70ecf480107e1e7ccd09a627`, based on develop
`fc19f89e8d37b9b7e823ac718f3025161c41bbf0`. It covers this source and runtime;
incoming PR239–243 and subsequent report updates require a separate read-only
snapshot of their combined checkout. The counts below are not timeless.

The inventory contains 47 tasks, 44 assigned to 6 views, and 3 unassigned named
affine 14k continuation variants. Every view's declared calibration is
provisional. There are 13 candidate declarations and 35 current source/runtime
rows; all attain tier 0, with 25 INCOMPLETE and 10 BLOCKED rows in both selected
views. They represent configuration/cohort combinations, not 35 independent
scientific trials.

| Local view | Required cells | NOT_RUN | BLOCKED | Current qualified options |
| --- | ---: | ---: | ---: | ---: |
| quality_coverage | 770 | 568 | 202 | 0 |
| discriminator_stability | 840 | 618 | 222 | 0 |

No current PASS or FAIL is inferred from these missing/blocked cells. Four
pinned cohorts, 22 calibration diagnostic cohorts and 143 historical rows remain
separate and unranked; these counts are not added as independent experiments.
Atlas/E22 currently lack the policy-aware served-sampling/evidence task
contract needed by these Forge hosts. Released-GAN variants also encounter
frozen-host recipe/resource ownership blockers. Other current rows lack
compatible evidence; their historical outcomes retain their original meaning.

Forge hashes the broad source tree. Adding the new audit module changes the
source digest from
`2f8e6686446071456967c834d77d79922a75ddcaef2cd96256e203ac8ad2896c`
to `38d8007b628b8e4756885cb1402d0cd2dcd5de17e59d68b0e72fb8bffbb75b8e`.
The receipt keeps both identities and imports no old qualification. It binds
the full external machine reports, source-board digests, view fingerprints,
config/recipe identities, task roles/budgets and tool bytes. Native code,
configs and policies were unchanged; no unchanged science was rerun to fill
the new source identity.

## New standalone tests and validation

PR239–243 are separately bounded, fixed-recipe public API caption diagnostics.
Their scientific results and goal GIFs are useful evidence for their declared
questions. They are not registered Forge tasks or generic recipe/config
comparisons. For example, PR239/240 freeze E22 ordinary and E22_routed particle
recipes with declared conditional controls; their `--protocol` validates
task/gate metadata, and their caller accepts no arbitrary `--recipe` override.
An Atlas/KA2 config selected elsewhere was not executed by those examples.
Their original FAIL/PASS outcomes do not automatically fill Forge requirements,
verify accepted calibration, or justify a default promotion. A policy-aware
config comparison would need an explicitly frozen task and fair protocol; none
is silently registered or launched here.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -p no:cacheprovider -q \
  tests/test_toy_forge_selection_readiness.py \
  tests/test_forge_tier_report.py tests/test_forge_board_filters.py
```

45 software checks passed in 2.18 seconds, including 16 new self-contained
controls. They cover unknown/blocked required cells, diagnostic separation,
historical/pinned exclusion, provisional and forged accepted-calibration
claims, conflicts, whole-row filters, source/runtime drift, missing required
tasks and multiple options without a fabricated winner. The actual empty-board
CLI runs with queue construction, source snapshotting and runtime execution
disabled, and verifies no files or queue directories are created. Independent
read-only review found no material blocker. These tests perform no scientific
training or default changes and require no historical Git objects.

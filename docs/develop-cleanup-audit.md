# Develop repository cleanup audit

The first cleanup priorities are the public Forge package's optional dependency
imports and the outdated experiment policy guidance. Further work should repair
evidence navigation, reduce recall output, and control artifact growth. Existing
storage, declaration, leaderboard, and research freshness checks pass; these
findings do not call for a new experiment or a change to recorded qualifications.
The implementation record below closes the ten findings while preserving this
audit's original measurements and evidence references.

Audited **2026-11-03**, branch `develop`, commit
`a1e8073ae8352d3aff25f04bb51446454e873d90` (PR #505). The checkout was clean at
the start. This report is the only repository change from the audit.

## Scope and repository measurements

The audit inventoried all 22,762 tracked files, scanned local links across all
1,460 Markdown files, and parsed all 12,131 JSON, 3,583 Python, and seven TOML
files. Current documentation, public packages, CI, examples, Forge declarations,
and retention rules received focused review. This combines broad automated
coverage with targeted reading; it is not a line-by-line semantic review of every
historical source snapshot. No training, image inspection, package installation,
or full test suite was needed.

| Measurement at the audited commit | Result |
| --- | ---: |
| Tracked file contents, uncompressed | 1,772,556,041 bytes / 1,690.44 MiB |
| Files under `reports/` | 19,403 / 85.24% |
| Content bytes under `reports/` | 1,725,408,766 / 97.34% |
| GIF files | 2,092 / 585.83 MiB |
| NPZ files | 1,459 / 264.40 MiB |
| PT files | 82 / 30.25 MiB |
| Groups sharing an identical Git blob | 844 |
| Additional bytes from identical copies, beyond one copy per group | 163,093,268 / 155.54 MiB |
| Additional bytes from identical GIF and HTML copies | 113,194,077 / 107.95 MiB |

These are checkout measurements from `git ls-tree -rl HEAD`, not estimates of
Git pack savings. Git already shares identical blobs. Retained media, fixtures,
and source snapshots are not automatically disposable.

## Prioritized findings

P1 affects advertised use or mandatory operating guidance. P2 addresses
reproducibility, navigation, integration, or substantial maintenance cost.
P3 covers smaller consistency and code hygiene improvements.

| ID | Priority | Cleanup | Completion criterion |
| --- | --- | --- | --- |
| C01 | P1 | Decouple Forge core imports from plotting dependencies | Base wheel imports both public packages without experiment extras |
| C02 | P1 | Replace obsolete current tier and budget guidance | Current guidance matches live declarations and planned execution groups |
| C03 | P2 | Correct reproduction environment instructions | Advertised minimum Python matches package metadata |
| C04 | P2 | Separate historical recipe guidance from current defaults | Current claims match `DEFAULT_RECIPE`; historical results retain their protocols |
| C05 | P2 | Repair generated evidence links | Missing originals lead to durable records or explicit archive guidance |
| C06 | P2 | Make recall concise by default | Ordinary recall omits full manifests while preserving source identity |
| C07 | P2 | Resolve the public prior return annotation | Runtime type inspection succeeds without an import cycle |
| C08 | P2 | Reduce future artifact duplication and budget generated data growth | Publications reuse retained media; retention and size checks remain valid |
| C09 | P3 | Remove duplicate imports and review a lint baseline | Obvious merge residue is removed without changing fixtures or exports |
| C10 | P3 | Align project terminology with the research glossary | Project overview uses declared terms and includes held requirements |

### C01 Forge core import requires undeclared optional dependencies

[particlegan_forge/__init__.py](../particlegan_forge/__init__.py), line 25,
eagerly imports the GIF renderer. [gif.py](../particlegan_forge/gif.py), lines
8–12, imports Matplotlib and Pillow. The base dependencies in
[pyproject.toml](../pyproject.toml), lines 11–14, contain Torch and NumPy;
plotting dependencies belong to the `experiments` extra. Consequently even
`from particlegan_forge import TaskSpec` fails when Matplotlib is unavailable.
A subprocess import hook simulating absent optional modules reproduced
`ModuleNotFoundError` without changing the environment.

Make renderer dependencies lazy and provide an actionable error when rendering
is requested without its extra. Add a minimal installed-wheel import smoke for
both public packages. The existing [CI workflow](../.github/workflows/tests.yml),
line 64, installs the relevant extras before its wheel smoke and therefore masks
this case. Validate core imports in a base-only environment and rendering in an
environment with the declared extra.

### C02 Mandatory experiment guidance mixes obsolete current policies

[EXPERIMENTATION.md](../EXPERIMENTATION.md), lines 321–345, describes a current
six-task Tier 1 with a 2,220-second reservation and revision-8 campaigns. Lines
445–467 label revision 11 and a `6/10/1` counted denominator current. The live
[discriminator stability view](../configs/forge/views/discriminator_stability.json)
is revision **14**, with **11/18/1 required tasks**. Its Tier 1 task timeout
allowances sum to **4,140 seconds**. That sum is a declaration measurement; an
actual reservation must come from `plan`, accounting for execution groups and
reuse. The generated [tier guide](../reports/forge/EXPERIMENTS_BY_TIER.md), line
50, reflects the newer declarations.

Replace present-tense policy tables with current generated references or a
small generated summary. Label older budget explanations by their pinned view
revision. Preserve historical campaigns, task contracts, results, and calibration
qualifications. Validate the replacement against the tier guide and a bounded
read-only plan, rather than changing historical budgets to match today's view.

### C03 Reproduction instructions advertise an unsupported Python version

[docs/reproducing.md](reproducing.md), line 3, says Python **3.10 or newer**.
[pyproject.toml](../pyproject.toml), line 11, requires **3.11 or newer**, consistent
with README and CHANGELOG. Following the guide with Python 3.10 reaches an
installation refusal.

Correct the guide and check support claims against package metadata. Preserve
the separately recorded environments of historical studies.

### C04 Historical recipes and integration notes appear as current guidance

[docs/toy100.md](toy100.md), line 14, says `get_recipe("gan")` now selects KA2.
[docs/convergence-tips.md](convergence-tips.md), lines 184–205, similarly presents
KA2 settings as current defaults. [particlegan/recipes.py](../particlegan/recipes.py),
line 7, selects `bispf_optimized`. The historical
[100-Gaussian runner](../examples/100gaussians.py), lines 17 and 67, explicitly
pins KA2; its protocol should remain intact.

Two related passages need historical scope: [EXPERIMENTATION.md](../EXPERIMENTATION.md),
lines 130–135, still describes E22/KA2 as incoming to `develop`, although the API
is public; [docs/reproducing.md](reproducing.md), lines 51–65, calls the older
VICReg prior comparison current, while README links the
[2026-10-25 matched ablation](../reports/prior-ablation-2026-10-25/README.md).

Refresh current default statements, name the older recipes explicitly, and date
historical protocol sections. Keep original scores, initializers, sampling laws,
and source pins. Validate recipe claims through `DEFAULT_RECIPE` and
`get_recipe`, without retraining old demonstrations.

### C05 Generated memory links point to absent attempt receipts

[EXPERIMENT_MEMORY.md](../reports/forge/EXPERIMENT_MEMORY.md) contains **21** local
`Evidence` links whose target attempt receipts are absent: 11 records marked
`current` and ten marked `research_diagnostic`. One example is line 4025,
attempt `00db84a9a51c43ad8782fce53359e1b4`.
[knowledge.py](../experiments/forge/knowledge.py), lines 729–732, constructs links
directly from `source.path`; line 952 emits them without an availability check.

This is a navigation defect, not proof that evidence was lost. Nine IDs occur in
the [family word repair archive](../reports/forge/family-wide-word-repairs/archive.json),
two in the [K3P horizon archive](../reports/forge/k3p-two-pole-horizon-v1/archive.json),
and four in prior-ablation compact receipts. The example's original also exists
in the ignored local queue. Freshness checks explicitly separate publication
freshness from original artifact availability.

Generate links to available durable records and known archive manifests, with
[artifact resolver guidance](forge-artifact-resolver.md) where appropriate. Mark
unpublished or unavailable originals explicitly instead of emitting a dead local
link. Check navigation in a checkout without local runs. Preserve identities and
verdicts; do not hydrate or rerun training simply to fix a link.

### C06 Recall limits record count but forwards large source manifests

[knowledge.py](../experiments/forge/knowledge.py), lines 693–697, passes complete
`source` objects through recall. [The CLI](../experiments/forge/__main__.py), line
591, limits the number of matches rather than the size of each one. Query
`cleanup documentation maintenance` returned three matches totaling **167,087
bytes of rendered JSON**; one source manifest contained **1,314 file hashes**.
This makes the mandatory recall step expensive to read even for a narrow query.

Return compact source pins, conclusions, costs, and relevant failed metrics by
default, with an explicit full-provenance option. Keep stored records unchanged.
Verify exact-ID lookup, source identity, negative-result visibility, and bounded
output size on this query and a scientific query.

### C07 A public return annotation fails runtime inspection

[ParticlePrior.manipulate](../particlegan/particle_prior.py), lines 243–246,
annotates its return as `"ManipulatedPrior"` but imports that name only inside
the method. This fails:

```python
from typing import get_type_hints
from particlegan import ParticlePrior

get_type_hints(ParticlePrior.manipulate)
# NameError: name 'ManipulatedPrior' is not defined
```

Bind the annotation target in the module's runtime namespace while checking for
import cycles. A `TYPE_CHECKING`-only import would leave this failure unresolved.
A sweep of 288 exported functions and methods found this one resolution failure;
ordinary method execution was not shown to fail.

### C08 Publication copies and generated data need explicit growth budgets

Identical GIF and HTML copies add **107.95 MiB** to the checkout beyond one copy
of each blob. For example, the same 584,380-byte word-smoke GIF occurs at 14
paths, including candidate and control media under
`reports/forge/bcap-default-baseline/media/` and several prior-search publications.
The [prior-search aggregator](../reports/forge/bcap-prior-search/aggregate.py),
lines 130–135, copies result files and entire media trees into each publication.
Shared controls can legitimately reuse the same evidence; duplicate media is
not evidence of duplicate experiments.

Prefer canonical retained media references for future publications. Review
existing copies against inbound links, retention manifests, and source pins
before moving any file. Keep required actual-training GIFs accessible. Existing
PT/NPZ artifacts also have consumers and pinned retention contracts; their
extensions alone do not justify deletion. Follow the exact commit/blob archive
procedure in [reports/README.md](../reports/README.md), lines 26–71.

Separately, `reports/forge/technique-inventory.json` is **23.63 MiB**, close to
the **25 MiB per-file cap** in
[the hygiene test](../tests/e2e/test_repository_hygiene.py), lines 16 and 40–43.
Plan a compatible split or compact summary before further growth hits the cap.
Any migration must preserve consumers, evidence IDs, and deterministic generated
results. Removing checkout duplicates will not reclaim old Git pack history.

### C09 Obvious import residue warrants a reviewed lint baseline

[word_adapter.py](../experiments/forge/word_adapter.py), lines 20, 22, and 24,
imports `recipe_owned_prior` three times. A bounded `ruff check --select F,E9`
over packages, examples, libraries, experiments, benchmarks, scripts, and tests
reported **318 diagnostics**. Many involve intentional fixture imports or
exports; this count is not a count of confirmed runtime bugs.

Remove the obvious duplicate imports, review a small useful rule set, and then
enforce a baseline in CI. Avoid bulk autofixes or formatting archived snapshots.
`mean_transport.FIT_CHUNK`, for example, is consumed as a re-export; the frozen
`FrozenFeatureReferenceBirthDeath` implementation also explicitly documents why
it duplicates earlier behavior.

### C10 Project overview disagrees with the canonical terminology

[PROJECT.md](../PROJECT.md), line 9, expands BCAP as “Bounded Critic Alignment
Penalty,” while [RESEARCH_GLOSSARY.md](../RESEARCH_GLOSSARY.md) records no
expansion. Lines 34–35 omit held importance and conflate scientific records with
execution receipts.

Use the glossary's BCAP description and distinguish required, held, and
diagnostic assignments. Link the existing ownership and schema contracts rather
than maintaining another abbreviated definition.

## Recommended work order

1. Fix C01 and its minimal dependency import smoke, then C07's annotation.
2. Correct C02–C04 and C10 together as a documentation cleanup, preserving
   historical declarations and evidence.
3. Repair C05 and C06 in the reporting layer, checking fresh-checkout navigation
   and concise recall while leaving qualification reducers unchanged.
4. Address C08 incrementally through future publication behavior and a reviewed
   retention migration; take C09 as a small separate code hygiene change.

No new leaderboard is needed for this work. The current per-goal index and
historical-page dispositions already exist and passed their checks.

## Verification and exclusions

| Check | Result |
| --- | --- |
| `python -m experiments.forge validate` | PASS: 574 candidates, 178 tasks, 118 studies, 27 views |
| `python -m experiments.forge compile --check` | CURRENT: 886 records, 501 publication records; all 11,785 scoped inventory files classified |
| `python ops/refresh_forge_readouts.py --check --json` | CURRENT across all five stages |
| `python -m pytest -q tests/e2e/test_repository_hygiene.py tests/test_leaderboard_index.py` | 17 passed in 10.21 seconds |
| All tracked JSON and TOML parsing | PASS; no duplicate JSON object keys |
| Python AST parsing outside `reports/` | All 1,126 files pass |
| All tracked Python AST parsing | One syntax error in a preserved historical attempt, described below |
| Inline local file links in top-level Markdown and `docs/` | All resolve |
| Tracked files matching ignore rules | None |

The sole Python syntax failure is the archived
`reports/toy100/lrfree-search/feature-cells-cb64-ra/fixes/performance/training-regression/count-review/population-scheduler/check-population-attempt1.py`,
line 176. Preserve that original attempt rather than silently repairing its
source. Broad Markdown scanning found 1,535 missing local-link occurrences,
mostly in historical or explicitly local-artifact reports. Those are not a
blanket broken-link backlog; C05 isolates a defect in current generated navigation.

Audit outputs remain outside Git. On this workstation, inspect
`/tmp/particlegan-cleanup-hygiene-tests.log`,
`/tmp/particlegan-cleanup-forge-{validate,freshness,readouts}.log`,
`/tmp/particlegan-cleanup-{inventory,parse}.json`, and
`/tmp/particlegan-code-audit-ruff.json`. For example:

```sh
tail -F /tmp/particlegan-cleanup-hygiene-tests.log
```

## Implementation record

The cleanup implements all ten findings. Core Forge imports no longer load
optional plotting packages; rendering reports its required extra and preserves
the caller's Matplotlib backend. The prior annotation resolves at runtime without
an import cycle. A separate CI job tests an installed wheel with base dependencies.

Current documentation now follows revision 14, Python 3.11+, and the BISPF
default, with earlier KA2/VICReg studies and campaign budgets explicitly scoped
to their historical contracts. BCAP and held requirements follow the glossary.

Generated memory links for raw attempts point to durable records and retrieval
instructions. Recall retains source identity pins by default and supports
`--full-provenance`; the audit's three matching records shrink from 167,087 to
6,030 rendered match bytes without changing the stored records.

The current inventory prunes only unreachable content-addressed catalog entries.
It shrinks from **24,773,683 to 19,235,723 bytes**, keeping 559 referenced recipe
contracts and 380 task contracts, with their contents unchanged. The writer checks
the 25 MiB budget before changing publication files. Every scientific row remains
unchanged, and all **2,809 original receipt and record files** retain their byte
hashes. The publication's administrative digest is updated to describe the compact
representation.

Future media aggregation reuses tracked GIFs by SHA-256 or writes one canonical
asset for unseen content. New indexes and Markdown link to those assets;
per-attempt receipts remain byte-identical. Existing archived media, source
snapshots, and their pinned paths are retained. The 107.95 MiB duplicate-media
measurement above therefore remains an inventory of historical copies.

Duplicate prior imports are removed. CI now checks a pinned Ruff version against
a reviewed baseline across live packages, experiments, benchmarks, scripts,
operations, and tests. New diagnostics and resolved exemptions require review;
moving unrelated lines does not invalidate the baseline.

Publication integrity and current-source measurements have separate checks.
`python ops/refresh_forge_readouts.py --check --reports-only` checks retained
publication integrity, while strict `--check` continues to expose source drift.
The cleanup does not transfer old training measurements to the edited source or
rerun unchanged experiments to obtain a green maintenance check.

Implementation verification includes 165 passing core integration tests, focused
storage and original-evidence integrity tests, a built-wheel import smoke outside
the checkout, declaration validation, current compiled memory, and a passing
retained-publication check. Implementation logs remain local under
`/tmp/particlegan-cleanup-implementation/`; the main test log is
`final-core-tests.log`.

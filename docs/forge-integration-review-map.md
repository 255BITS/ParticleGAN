# Forge integration review map — 2026-10-02

Use the [original audit](../reports/forge-audit-2026-10-02.md) as the inspected
baseline at `b8feb1fe8e672c3e7a6b579329af0bd89a069ff0`. This map separates the
review obligations of the release integration from the focused preparation PRs.
It is an engineering completion map, not another leaderboard or new research
qualification. Preserve the [clean-MoG inventory](../reports/forge/technique-inventory.md)
and [served-policy inventory](../reports/forge/policy-family-inventory.md) as the
single authoritative board for each respective goal.

## Review boundaries

| Boundary | Review entry points | Reviewer must establish | Relevant controls |
| --- | --- | --- | --- |
| Public trainer, recipes and state | `particlegan/training.py`, `particlegan/recipes.py`, priors, E22/Atlas policy/state modules; [develop integration](../reports/forge/DEVELOP_INTEGRATION.md) | Exercised recipe, prior, initialization, state, named RNG streams and sampling law match their own source cohort; saved-state continuation preserves the full policy | `test_forge_policy_integration.py`, `test_forge_boundaries.py`, `test_forge_sampling_boundary.py`, E22/Atlas tests |
| Executor and operational ownership | `experiments/forge/queue.py`, `worker.py`, `sources.py`, `runtime.py`; policy coordinator under `benchmarks/toy_audit/api_family_search.py` | Compatible submitters cause one physical launch; source is frozen; live leases fence recovery; complete reservations, interruption/retry costs and resource admission remain accounted | `test_forge_queue.py`, `test_forge_resources.py`, `test_toy_api_family_search.py` and policy ownership PR controls |
| Scientific scorers and protocol | `experiments/forge/views.py`, native/vector/image adapters, `benchmarks/toy_audit/api_contract.py`, task/view/calibration declarations | Full required denominators and unknowns remain; thresholds, terminal/hold windows, recipe/prior/runtime identities and clean/noisy serving cohorts are not exchanged | `test_toy_api_contract.py`, `test_forge_sampling.py`, calibration/lane/promotion controls |
| Evidence storage, reducers and publication | [artifact resolver PR #256](https://github.com/255BITS/ParticleGAN/pull/256); maintained `experiments/forge/policy_publication.py` and `policy_family_readout.py`; inventory publication wrapper | Exact originals are retrievable or explicitly missing; checksums, archive commit/blob/source identities and raw errors survive; reductions preserve measured costs and verdicts; publication grants no qualification | `test_forge_artifact_resolver.py` from #256; default-discovered `test_forge_policy_publication.py` including frozen-output parity |
| Agent recall and research decisions | `experiments/forge/knowledge.py`, planning/lifecycle/readout contracts; focused recall, calibration and decision PRs below | Every concluded trial is discoverable; publication freshness is visible; a proposed bounded measurement binds prior evidence, substantive delta, predicted numerical discriminator, falsifier and terminal action | Recall freshness/coverage tests, decision-contract controls, calibration feasibility controls |

An evidence-publication change can alter the implementation/source digest without
changing the recorded scientific result. Review that new identity explicitly;
do not transfer an old pass to a newer checkout or rerun unchanged science to
make a reporting merge look current. Frozen result receipts and provenance stay
the authority for their original cohort.

## Focused preparation PRs and acceptance

Each branch starts independently from the inspected `develop` base. A row marked
pending is not a claim that the audit priority is complete. Implementation
acceptance remains subject to the PR's tests and combined integration checks.

| Audit priority | Focused change | Acceptance / remaining empirical work |
| --- | --- | --- |
| P1: repair recall and freshness | `codex/forge-audit-recall` — PR pending | Concluded Forge configurations and policy trials are normalized source-bound recall inputs; exact IDs/failed bounds/mechanisms/goals link authoritative evidence; stale/missing publication coverage is visible without training |
| P1: shared policy execution ownership | `codex/forge-audit-policy-ownership` — PR pending | Immutable execution source, coordinated deduplication/admission, locks/leases and charged interrupted work; no clean-MoG qualification from policy results |
| P1: retrieve original evidence | [#256](https://github.com/255BITS/ParticleGAN/pull/256), `codex/forge-audit-artifact-resolver` | Verified fresh-process fixture hydration of full request/evidence/result/source and selected checkpoint; pinned Git originals; explicit invalid/missing outcomes and retention ownership gaps. Actual `/ml2` archives still require a real mounted/transferred copy |
| P1: prevent infeasible calibration spending | `codex/forge-audit-calibration-feasibility` — PR pending | Read-only logical feasibility preflight and registration guard preserve bound published matrices and original hashes. **A justified new screen, independent positive/negative references, and empirical accepted calibration remain pending** |
| P2: hypothesis-to-decision contract | `codex/forge-audit-decision-contract` — branch/PR pending | Prior-evidence identities, substantive delta, numerical prediction, competing explanation, falsification observation and bounded terminal rule must bind mechanically; causal judgment remains explicit research review |
| P2: maintain publication controls and make integration reviewable | [#257](https://github.com/255BITS/ParticleGAN/pull/257), `codex/forge-audit-publication` | Maintained reducers live in the experiment package; original negative controls enter default CI; frozen parity and deterministic bytes pass; original audit/reproduction identities and this review map are preserved |

The calibration preparation does not produce new data or an accepted screen.
Logical infeasibility can block wasteful matrix filling; engineering checks
cannot establish false-reject behavior, population reliability or adoption.
Existing published matrices retain their qualified source/initializer/recipe
bindings; projection or legacy-binding changes must not silently turn those
recorded outcomes into a newer cohort. Unknown independent-reference coverage
remains unknown. No new training, seed study, qualification regrade or promotion
is authorized by this map.

## Merge order and combined verification

Recommended order into `develop`: artifact access (#256), this publication PR,
recall, policy ownership, calibration feasibility, then the decision contract.
These are independent code-review units, not stacked PRs. The order exposes the
evidence and publication surfaces before agents rely on fresh recall and launch
contracts. Integrate the focused heads into a temporary verification checkout
first; resolve any localized shared-file changes in `__main__.py`, planning,
knowledge or contracts while preserving all scientific receipts.

Run the affected software suites and normal default CI discovery in the combined
checkout. Verify that read-only commands construct no worker/submission and that
publication controls remain included by `testpaths = ["tests"]`. Validation
must not train, hydrate nonexistent archives, regrade original matrices, fill
unknown cells, regenerate duplicate boards, or force-add ignored execution logs.
Raw stdout/JSONL/JUnit/checkpoint/state artifacts remain local; compact reports,
final metrics and provenance receipts remain reviewable in Git.

For [release PR #247](https://github.com/255BITS/ParticleGAN/pull/247), review the
public API/recipe behavior, executor invariants, scorer/protocol identities and
evidence source cohorts as separate obligations using the table above. Its
shipping selection rationale requires its applicable fully qualified winner,
accepted calibration and preregistered promotion evidence. Framework readiness
is a separate engineering decision. Focused preparation PRs can improve bounded
agent research without claiming a shipping default or changing #247's release
gate.

## Maintained publication entry points and frozen reproduction

Future maintenance belongs to:

```sh
python -m experiments.forge.policy_publication \
  --combined /path/to/existing/certified-combined.json \
  --output reports/forge --all-media
python -m experiments.forge.policy_family_readout \
  /path/to/existing/original-family-study.json \
  --output runs/forge/local-policy-readout
```

The publisher updates the existing policy inventory filenames. A local readout
is a display projection, not a second leaderboard. Inputs must resolve their
original receipt/artifact bindings; missing originals block publication. These
commands use saved verdict streams and certified bytes without training or gate
rescoring. No board was regenerated for this refactor.

The dated `reports/forge/family-winner-round1/publish_policy_results.py`,
`policy_family_readout.py`, and `test_publish_policy_results.py` remain
byte-identical frozen reproduction sources. New development uses the package
and `tests/test_forge_policy_publication.py`; retaining dated sources is an
intentional evidence exception, not a second maintained implementation. The
[migration provenance receipt](../reports/forge/publication-migration-2026-10-02.json)
records their exact baseline commit, Git blobs, byte hashes and maintained
destinations. Do not replace those originals with wrappers: archived publisher
contracts bind their bytes. The original audit is copied unchanged.

The package migration changes only the publisher's implementation provenance,
package import and repository-root lookup. Controls compare frozen and maintained
scientific projections and Markdown exactly, preserve receipt/artifact/source
identity, compare repeated JSON/media bytes deterministically, exercise the
readout CLI and retain the original source-bound negative cases. These are
synthetic publication controls, not new toy training evidence.

Tail local verification with `tail -F runs/forge/audit-publication/pytest.log`.
Default collection and focused test logs stay in the same ignored directory.

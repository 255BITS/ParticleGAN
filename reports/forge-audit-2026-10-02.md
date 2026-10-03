# Forge audit — 2026-10-02

**Purpose:** prepare ParticleGAN for agent-driven GAN research. Perspective: a senior research engineer accountable for experimental validity, autonomous execution, and the usefulness of each unit of compute.

**Scope:** [PR #247](https://github.com/255BITS/ParticleGAN/pull/247), the current `develop` → `master` 0.9.0 integration, and Forge at `b8feb1fe8e672c3e7a6b579329af0bd89a069ff0`. The PR number was inferred from the checkout and its research-integration purpose; the user did not supply a number. The original framework arrived through [#221](https://github.com/255BITS/ParticleGAN/pull/221); the family-selection plan through [#249](https://github.com/255BITS/ParticleGAN/pull/249). This is a targeted architecture and evidence audit, not a line-by-line review of the entire integration diff.

**Assessment:** Forge is a strong foundation for bounded, supervised agent research. Its central executor has substantially better scientific accounting than a collection of experiment scripts. Preparation for unattended research is incomplete: recent results are absent from recall, the policy-family execution path bypasses central coordination, and original evidence is not reliably accessible from this checkout. Scientific screening also remains provisional. Preserve the engine and close these gaps before expanding autonomous search.

Finding a winning GAN is a research outcome, not a prerequisite for declaring the research infrastructure useful. The release decision has its own stricter contract: #247 explicitly requires a qualified winner and applicable calibration/promotion evidence before merging.

**What the structure gets right.**

The separation between task conditions, formulation mechanisms, hyperparameters, and comparison policy is sound. [Field ownership](../docs/forge-field-boundaries.md), the shared [public formulation binding](../experiments/forge/api.py), and task-owned priors prevent an agent from silently changing the problem while claiming a better optimizer. Named RNG streams, actual recipe resolution, initialization bindings, and clean versus noisy serving identities are particularly important for adversarial training.

The [queue](../experiments/forge/queue.py) freezes source, reserves complete task allowances, deduplicates compatible requests, protects execution with leases, and preserves costs and linked infrastructure retries. Required failures stop downstream spending. [Qualification](../experiments/forge/views.py) retains unknown and blocked requirements; diagnostics cannot manufacture ordinary qualification. These are meaningful operational safeguards, not just conventions in prose.

Whole-configuration family selection also avoids a common research error: assembling a purportedly universal recipe from different per-task winners. The current boards correctly distinguish best-observed configurations, tuning qualification, full qualification, and adoption. Representation witnesses remain separate from learning evidence, and acquisition is followed by an uninterrupted hold. Actual-training GIFs support interpretation; numerical gates determine outcomes.

**What the measured research currently establishes.**

Use the existing [MoG leaderboard](forge/technique-inventory.md) and [public-policy leaderboard](forge/policy-family-inventory.md) as the authoritative rankings. They serve different goals and sampling cohorts. This audit creates no additional leaderboard.

The [completed campaign](forge/family-winner-round1/campaign-completion.json) records 60 whole configurations across seven families: 28 Forge configurations and 32 Atlas/E22 policy configurations. There are 198 certified ordinary task runs and four setup attempts without optimizer updates. Recorded child/setup cost totals **2,721.4755 seconds**; this is summed execution cost, not elapsed campaign duration or total machine cost. There is **no fully qualified winner, shipping default, or speed winner**.

Within the frozen MoG search, KA2's best observed configuration passes 10/24 requirements: 3/3 smoke, 7/19 quality, and 0/2 endurance. Its unequal-mass failure includes rare-component mass ratio .598 and covariance eigenvalue ratio .198 despite HQ about .976. These failures warrant investigation; a high HQ aggregate does not establish distribution fidelity. The round retains 128 PASS, 28 FAIL, and 516 UNKNOWN cells. Unreached requirements are not empirical algorithm failures. [Source-bound explanation](forge/family-winner-round1/README.md).

The latest Atlas/E22 setting, LR .0053125 and prior multiplier 1.5, passes both original smoke-toy protocols. Broad-mixture acquisition occurs at update 1100/1200, leaving only two of the five required later hold observations. Its separate study therefore remains **1/8**, with persistence INCOMPLETE and all six quality requirements UNKNOWN. This does not demonstrate collapse; it demonstrates insufficient retained hold evidence under the frozen budget. Atlas and E22 match numerically on these small-population hosts, which do not exercise Atlas's distinct large-population backend. External contention precludes a speed comparison. [Policy preparation and limits](forge/family-winner-round1/POLICY_PREPARATION.md).

Pass counts describe progression through an ordered suite, not a calibrated scalar measure of GAN quality. Different grid sizes, earlier stopping, source/runtime changes, and different serving laws prevent interpreting these partials as a causal family ranking. The failed native formulation comparison also retains its separate source and full-budget failures; the later search does not overwrite it. [Native comparison](forge/FORMULATION_COMPARISON_READOUT.md).

**P1 — Repair the research-memory loop before agents propose more work.**

This is the clearest demonstrated preparation defect. The committed [memory](forge/EXPERIMENT_MEMORY.md) and [compilation manifest](forge/compilation.json) contain 257 records and six views; current validation resolves eight views. The manifest has 49 changed existing inputs and omits 45 current configuration files. Its recorded inventory is incomplete, whereas a fresh read-only inventory check is complete at **11,712/11,712** scoped paths. The stale report can mislead an agent in both directions.

More materially, none of the 28 current family-search candidate IDs appears as a normalized memory record. Exact searches of all record text find no `family-defaults-round1`, `policy-family-defaults`, or final prior-balance study. `recall` for KA2 configuration prefix `093c6f2bd417` returns zero results even though the primary board publishes it. [Record discovery and recall](../experiments/forge/knowledge.py#L45) read `reports/forge/records`; they do not ingest the new configuration-search or policy publications. Broad token matching can return many unrelated records and conceal this omission.

Register compact, source-bound study and trial readouts for both paths, then refresh memory. Recompilation alone cannot invent absent records. Include publication/readout inputs in freshness accounting and make stale recall coverage explicit in plans. Completion criterion: every concluded configuration is retrievable by exact ID, failed bound, mechanism, and goal; recall links its authoritative board and original evidence without granting a new qualification. Publication freshness should be checked in CI without training or raw-log hydration.

**P1 — Give policy experiments the same execution ownership as Forge.**

The [policy runner](../benchmarks/toy_audit/api_family_search.py#L559) correctly freezes a study specification, verifies evidence, enforces per-family budgets, and retains interrupted attempts. However, it launches children from mutable `contract.ROOT`, rather than a frozen Forge source snapshot. Its study-state read/update cycle has no coordinator lock or execution lease; [JSON publication](../benchmarks/toy_audit/api_run.py#L52) uses a shared temporary filename. Separate output directories also bypass central scientific deduplication.

For multiple agents, this permits races between resume/launch operations, overlapping GPU admission, and redundant execution in different archives. Source verification can detect a mismatch after spending; it does not isolate execution from concurrent edits. These are architecture-derived risks; this audit did not observe or induce duplicate training.

Prefer admitting policy-aware tasks through the shared queue while retaining their separate cloud/served laws and denominators. If a dedicated coordinator remains, it needs equivalent locking, leases, immutable execution source, central resource admission, and cross-output identity checks. Completion criterion: two submitters create one physical compatible attempt; recovery respects a live worker; concurrent checkout edits cannot change its code; interrupted cost remains charged. Never fill clean-MoG cells with policy results during this integration.

**P1 — Calibrate screening before treating it as a research-selection oracle.**

Framework acceptance is supported; predictive screening validity is not. The [control readout](forge/CONTROL_MODE_READOUT.md) shows why the exact 57-cell profile is infeasible under its own adoption criteria: every lineage is smoke-negative, so either a later reference-positive is falsely rejected or the required positive-reference minimum is absent. False-reject probability remains unknown. The subsequent noise-removal profile is also infeasible. Collecting more cells solely to approve either cannot resolve that logical conflict.

The two-pole screen combines 80-update movement and gradient bounds on a direct particle-cloud host. It is useful behavioral evidence, but cannot be presumed to predict learned-MoG, conditional, or selected-serving convergence. The later family search remains valid as a provisional bounded study; it does not calibrate the filter merely by producing smoke passes.

Freeze a separately justified screen and compatible independent positive/negative references under the existing [criteria](../configs/forge/calibration/criteria-v1.json). Assess feasibility before spending, preserve unknowns, and retain the cost ratio requirement. Do not relax thresholds after failure or launch seed-only repeats. A small accepted calibration establishes scope-limited observed screening behavior, not a population-level reliability guarantee. Autonomous exploratory work can proceed under explicit provisional labels; adoption still requires accepted calibration and the applicable registered promotion evidence.

**P1 — Make original evidence retrievable, not just identifiable.**

The first-round archive has a SHA-256, executed commit, and exact receipt identities, but its recorded path is `/ml2/hypergan/forge-family-winner-round1-20261002/phase1-receipts-and-source.tar.gz`, unavailable here. The completion receipt explicitly acknowledges that independent regrading needs transfer or shared-machine access. Hashes support integrity after retrieval; they do not establish availability.

Add a documented artifact resolver with stable storage locations, checksums, retention ownership, and clear missing-artifact outcomes. A fresh agent should hydrate one selected original receipt/checkpoint and verify it without contacting the author or rerunning training. Keep projections display-only. The resolver should distinguish committed summaries, Git-pinned originals, and externally archived state. Preserve archive commit/blob identities and repair links when relocating historical tracked data.

**P2 — Make the research decision itself a compact contract.**

Forge already rejects unfinished scaffolds, seed changes, unsupported mechanisms, and many invalid hyperparameter axes. Keep those checks. The weaker part is the scientific transition from “these settings failed” to “this next measurement will distinguish explanations.” Hypotheses, changed-factor strings, and narrative recommendations alone do not guarantee that transition.

Extend the existing idea/readout schema with required prior-evidence identities, the exact substantive delta, a predicted numerical signature, the competing explanation, the observation that would falsify the hypothesis, and a terminal next-action rule. Validate mechanical bindings; leave causal judgment explicit for review. Admit one bounded round at a time. Search selection, diagnostic authorization, and public-default promotion should remain separate decisions.

The immediate useful research work is saved-state diagnosis of the observed rare-component/shape deficit and late acquisition, after evidence access is repaired. Compare per-component occupancy and moments with actual exercised mechanisms and critic gradients. A training failure does not establish inadequate representation, and an analytic witness does not establish optimizer reachability. This audit proposes no new training formulation or additional sweep.

**P2 — Make the integration reviewable and reusable.**

At the inspected head, #247 contains **7,013 changed files, 4,254,824 additions, 737,297 deletions, and 451 commits** beyond its actual master base `0ff9a7afe5dcb828239369446cfe71971bce687b`. It is a release integration ledger, not a tractable single code-review unit. Roughly 93% of added lines are in `reports/toy100`, `reports/forge`, and `reports/toy_audit`. Preserve that evidence; provide a compact review map separating public training changes, executor contracts, scorer/protocol changes, and evidence publication.

Future work should arrive as focused PRs along those boundaries, each with its affected evidence identities and validation. Keep #247's release selection rationale separate from the infrastructure-readiness decision. Do not rewrite history or rerun unchanged science merely to reduce this diff.

Reusable policy publication logic currently lives under a dated report directory, including [publication tests](forge/family-winner-round1/test_publish_policy_results.py). Pytest's configured `testpaths = ["tests"]` means that file is not discovered by the default CI invocation. This audit explicitly ran it. Move reusable reducers into the maintained experiment package and their controls into `tests/`, keeping frozen reproduction sources where they belong.

The inspected PR adds no raw log/JSONL/JUnit/tensor files by the checked bulk suffixes. Existing tracked Forge attempt envelopes nevertheless occupy about 30 MB across 135 files; those grandfathered identities deserve an archive plan as the repository grows. Required training GIFs are useful publication artifacts. Avoid treating their presence as equivalent to committing raw execution streams.

**Recommended preparation sequence and acceptance measures.**

| Order | Deliverable | Observable acceptance |
| --- | --- | --- |
| 1 | Complete and fresh recall | Every concluded trial discoverable; current input/view coverage; no unnoticed publication omission |
| 2 | Shared execution ownership | One physical attempt per compatible request; full reservations; fenced recovery; immutable source |
| 3 | Accessible evidence and maintained publication controls | Fresh-checkout hydration succeeds; hashes verify; missing state blocks analysis; tests run in default CI |
| 4 | Explicit hypothesis-to-decision contract | Each launch states its counterfactual, numerical discriminator, bounded cost, and stop/next-action rule |
| 5 | Feasible calibration and scoped adoption | Frozen independent references satisfy every applicable criterion; no diagnostic or cohort substitution |

Track these measures using existing receipts and telemetry: duplicate physical launches, budget overspend, unretrievable originals, omitted concluded readouts, and unsupported/cohort-crossing claims should each be zero. Report unknown measurement coverage explicitly. Use observed paired calibration errors and complete cost vectors when available. Do not optimize an agent for number of ideas, passing screenshots, or raw experiment throughput.

**Verification and limits.**

Fresh checks: `forge validate` passed with 49 tasks/eight views; read-only history validation passed with 11,712 scoped paths and zero missing, stale, or unclassified entries. **480 focused tests passed in 50.87 seconds**, covering planning, queue, search, memory, calibration, promotion, sampling/ownership boundaries, seed policy, family boards, policy search, selection readiness, and policy publication. Local runtime: CPython 3.12.13, Torch 2.14.0, NumPy 2.5.2, SciPy 1.17.1; CUDA was disabled for these checks.

The current-head [push CI](https://github.com/255BITS/ParticleGAN/actions/runs/37068218409) and [PR CI](https://github.com/255BITS/ParticleGAN/actions/runs/37068228755) both report success. Those are existing remote checks, distinct from this focused local run. No research campaign, seed experiment, queue submission, checkpoint regrade, or leaderboard regeneration was performed. Archived metric claims were inspected through their published evidence; unavailable originals were not independently replayed. Only this audit document was added to the source tree.

The execution log is local and ignored. Tail it with:

```sh
tail -F runs/forge/audit-2026-10-02/pytest.log
```

Reproduce the focused software check from the repository root:

```sh
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
.venv/bin/python -m pytest -q \
  tests/test_forge_planning.py tests/test_forge_queue.py \
  tests/test_forge_configuration_search.py tests/test_forge_knowledge.py \
  tests/test_forge_calibration.py tests/test_forge_calibration_lane.py \
  tests/test_forge_promotion.py tests/test_forge_sampling_boundary.py \
  tests/test_forge_boundaries.py tests/test_forge_seed_policy.py \
  tests/test_forge_technique_board.py tests/test_toy_api_family_search.py \
  tests/test_toy_forge_selection_readiness.py \
  reports/forge/family-winner-round1/test_publish_policy_results.py
```

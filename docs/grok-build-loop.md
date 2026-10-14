# Grok Build candidate loop

The installed `try-gan.sh` and `run-gan-pool.py` accept `loop CONFIG.json`.
Planning is the default. `--execute-test` generates and tests exactly one
candidate. `--continue-included` resumes only after the completed first test
receipt explicitly records delivery to the user and the included weekly bucket.
No automated merge, deployment, API fallback, purchase, or top-up occurs.

The selected full Git source commit and expected view revision are explicit.
Develop revision9 has6 required Tier1,21 Tier2 and2 Tier3 tasks. Revision10
contains the unmerged PR403 tier declarations and requires an explicit
`acknowledge_unmerged_PR403` acknowledgement. Archived receipts keep their
original revision. Every original task condition, evaluator, dependency,
initialization, sampling law, horizon and observation cadence is frozen.

A proposal is one reusable public Recipe JSON delta and a separate reviewed
READY study, committed on a named `codex/grok-candidate-*` branch/worktree.
The model has no tools or training authority. Source-code proposals are not
supported by this first loop implementation. Alias recipes are deduplicated;
the configured negative history and actual failed candidates feed future turns.

Forge submits the candidate once with its full finite horizon reserved. The
normal queue finishes runnable tasks in the current tier and applies the
required-task veto before the next tier. Every required original Tier1 task
must independently PASS for Tier2 to run. The **interesting gate** is every
required Tier1 and Tier2 task PASSing at the same candidate/source; then all
remaining Tier3 tasks run. BLOCKED, missing, INCOMPLETE and certificate failures
cannot promote. No passes are pooled across candidates.

Each candidate branch records compact actual gate/attempt/metric/cost and
source/certificate receipts under `reports/forge/grok-loop/`. Bulk logs and
checkpoints remain outside Git, with paths in those receipts. The loop records
its uncertain phase before a model turn, refuses automatic retry/reconciliation,
and preserves its original overall deadline on continuation. STOP and finite
proposal, candidate and compute limits fence owned processes. Existing shared
lease files are opened unchanged and inherited by worker/training/grader hops.
Coordinated blocked UTC intervals are checked against each full job horizon.

Build access uses the existing grok.com login and an explicitly verified
first-party model. A nonsecret account receipt must establish positive included
weekly allowance and a provider-enforced zero allowance for paid Additional
Usage. Explicit user account confirmation is accepted; model statements are
not. The account receipt expires after an hour and is checked before each new
turn. Time/turn caps do not prove a money cap. API-key environments are refused;
provider overrides and arbitrary credential environment variables are excluded.
Compatibility scanners, memory, subagents, tools and auto-update are disabled
for this process. Sanitized `grok inspect --json` must show no enabled hooks,
MCP servers or plugins before a live test. No credential files are read/copied.

A configured `allow_bounded_local_compute_increase: true` applies an already
recorded user authorization: it appends only the minimum allowance required
for one READY candidate's full original horizon. It preserves all spent cost,
negative history, event hashes and model-session limits, and refuses active
reservations. Funding follows admission/source/semantic checks.

Required config fields include `source_commit`, `expected_view_revision`,
`view_id`, `parent_candidate`, `study_template`, `scientific_source_semantics`,
`negative_semantic_history`, `model`, `grok_bin`, `account_receipt`, `runs_dir`,
`queue_root`, `budget_state`, `lease_root`, `stop_file`, `first_live_test_receipt`,
`max_candidates`, `maximum_loop_seconds`, `proposal_timeout_seconds`, and
`dispatch_window_seconds`. The source commit must include the current queue
claim fence and worker/fleet FD forwarding before it is frozen. The ledger and
queue must be the actual reconciled shared ones. Use `GAN_PYTHON` to select the
project's compatible runtime for the shell entry point.

Offline tests use a mock provider and real tiny CPU numerical gates. They
exercise failure feedback, tier vetoes, named branches, deduplication and
certificates; they grant no ParticleGAN qualification. A real provider/test
receipt is required before unattended operation. Provider usage/cost stays
unknown unless the CLI actually supplies a verified receipt.

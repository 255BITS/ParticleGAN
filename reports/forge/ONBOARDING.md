# Forge onboarding walkthrough

This walkthrough uses only the checked-in [root guide](../../EXPERIMENTATION.md)
and the actual Forge CLI in the existing feature worktree. It adds one public
mechanism ablation: remove K3P's critic EMA-anchor term with
`recipe_overrides.reg_anchor_weight=0`. The fixed protocol seed, initializer and
prior are inherited. This is neither a seed sweep nor a production-default claim.

The real CPU walkthrough rejected the ablation on its first task in **5.975
seconds**. Exactly one worker attempt ran; both remaining smoke tasks and all
higher-tier tasks stayed unlaunched. The readout and queue lifecycle are
**concluded**. Final compilation succeeded with no conflicts or pending readouts,
after the concurrent source-edit regression described below was repaired.

## Preparation

From `/home/martyn/dev/ParticleGAN-experiment-tiers`, with the project's Python
environment active:

```sh
python -m experiments.forge validate
python -m experiments.forge recall --goal discriminator_stability --query "critic anchor"
python -m experiments.forge new --id forge-onboarding-anchor-ablation --parent k3p \
  --goal discriminator_stability \
  --hypothesis "Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests."
```

Read the [compiled memory](EXPERIMENT_MEMORY.md),
[initial calibration](calibration/initial.md), and the preserved
[historical K3P qualification card](records/history-k3p-84363f69cb3a.json).
Historical cloud results do not establish the outcome of a new candidate.
Although the ordinary declaration inherits the learned-MoG default, this
`two_pole` run explicitly used its task's particle-cloud exception: direct
sample-particle coordinates, `sigma=0`. It is not evidence for MoG transfer.

Edit the generated [idea](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json):
set `recipe_overrides.reg_anchor_weight` to `0.0`, describe that single change in
`changed_factors`, and explain the named mechanism ablation in
`mechanism_rationale`. The committed declaration contains the complete values.
The dedicated [campaign](../../configs/forge/campaigns/onboarding-pilot.json)
caps both campaign and candidate at 900 seconds.

```sh
python -m experiments.forge plan forge-onboarding-anchor-ablation --through-tier 1 --device cpu
```

Observed preparation result: no preflight blockers; only `two_pole`,
`unused_token_hold` and `ae_gan_hold` are permitted by the tier cap, each with a
300-second maximum. The worst-case reservation is 900 seconds. Larger tests are
listed for context but are outside this request's execution cap. Source identity
will be resolved again when enqueue freezes the final implementation.

## Execution and readout

After source-freeze authorization, the actual commands were:

```sh
python -m experiments.forge enqueue forge-onboarding-anchor-ablation --through-tier 1 --device cpu --campaign configs/forge/campaigns/onboarding-pilot.json
python -m experiments.forge drain --gpus cpu --workers-per-gpu 1 --campaign onboarding-pilot
python -m experiments.forge logs --candidate forge-onboarding-anchor-ablation --campaign onboarding-pilot --lines 20
python -m experiments.forge queue
python -m experiments.forge board --goal discriminator_stability
python -m experiments.forge board --goal discriminator_stability --candidate forge-onboarding-anchor-ablation --scope pinned
```

The queue resolved to `/home/martyn/dev/ParticleGAN/runs/forge`. Enqueue returned
request `ac458f106e3b9533fe224b77` with status `queued`, without launching
training. Drain launched attempt `c0178716dd3c4d5695b391919bf9ab81` on the AMD
Ryzen 9 5900X CPU, one thread. The fixed protocol seed was `0`; no seed override,
retry or GPU was used.

| Task | Attempted updates | Scientific verdict | Measured terminal values | Charged wall seconds |
|---|---:|---|---|---:|
| `two_pole` | 80 | FAIL | `mean_abs=0.1048249304 < 0.30`; `grad_med=0.07678461075 <= 1.0` | 5.974550719 |
| `unused_token_hold` | 0 | NOT_RUN | No task receipt; stopped by predecessor failure | 0 |
| `ae_gan_hold` | 0 | NOT_RUN | No task receipt; stopped by predecessor failure | 0 |

The independently recomputed curve had 24 observations, zero passing
observations and no passing terminal suffix. The process completed successfully;
the scientific result was FAIL. This distinction matters: the scheduler stopped
downstream spending without needing a process error. The 900-second campaign had
zero reserved seconds after completion and charged only 5.974550719 seconds.
FLOPs were not measured.

The durable [request](attempts/c0178716dd3c4d5695b391919bf9ab81/request.json),
[result](attempts/c0178716dd3c4d5695b391919bf9ab81/result.json), and
[evidence certificate](attempts/c0178716dd3c4d5695b391919bf9ab81/evidence.json)
bind the frozen candidate revision
`39dc1cc8e516f8127e732edea196f82243ff8cdff4ed761020f0f802a3bcaf0f`
and source digest
`e2b2e0dc730b54a501c2d36549258a6afa81409c74d2191642eb57afa44b9d28`.
The raw public-component receipt records `reg_anchor_weight=0.0`, public
`K3PGeneratorAdam`/`K3PCriticAdam`, named RNG auditing, and the task's prior
exception. Source edits after enqueue did not change this frozen execution;
the board correctly retained it as a pinned FAIL with cost.

The exact readout command was:

```sh
python -m experiments.forge readout forge-onboarding-anchor-ablation@39dc1cc8e516f8127e732edea196f82243ff8cdff4ed761020f0f802a3bcaf0f \
  --conclusion 'The public EMA-anchor ablation failed two_pole after 80 fixed-seed CPU updates: mean_abs=0.1048249304 was below 0.30, while grad_med=0.07678461075 passed its <=1.0 bound. The independently graded terminal curve had 0 of 24 passing observations. One attempt consumed 5.974550719 seconds; no other smoke, quality or endurance task launched.' \
  --comparison 'The previous K3P CPU pilot also failed two_pole, but its source revision differs, so these receipts do not isolate the anchor effect or justify an efficacy ranking. Historical cloud qualification and new learned-MoG defaults are separate evidence cohorts. This host explicitly used its particle_cloud exception, sigma=0; the actual public recipe recorded reg_anchor_weight=0.' \
  --next-action 'Stop this ablation at tier 1. Keep its frozen FAIL receipt as onboarding and fail-fast evidence; do not promote it or spend on downstream tasks. A scientific anchor comparison would require one explicitly bounded compatible parent/ablation design after the initial gate calibration gaps are addressed.'
python -m experiments.forge compile
python -m experiments.forge queue
```

The [concluded readout](records/readout-10242573330bec1494c7602b.json) explains
why this is a useful workflow result and not a causal comparison with the older
K3P pilot. No production promotion or later-tier experiment was attempted.

Final `compile` returned 176 records, four views, no conflicts and no pending
readouts. Its input digest was
`9ac057dc0250658de32833fc4f0a8b98fecdda15822421f8971680800d20512e`.
Inventory coverage still reported seven newly tracked Forge engine files absent
from the older catalog; the coordinator must stage the final implementation and
refresh `history`/`compile`. This known coverage gap did not become a scientific
pass. Subsequent source changes can legitimately change compilation identities.

Tail the live operational logs with:

```sh
python -m experiments.forge logs --follow --candidate forge-onboarding-anchor-ablation --campaign onboarding-pilot
tail -F /home/martyn/dev/ParticleGAN/runs/forge/events.jsonl
tail -F /home/martyn/dev/ParticleGAN/runs/forge/onboarding-pilot/c0178716dd3c4d5695b391919bf9ab81/run.log
```

Operational log and source-snapshot paths are local to this checkout; the linked
request/result/certificate and readout contain the portable evidence.

## Guide review

The guide provides the entry point, memory links, idea fields, budgeting,
global-option placement, queue/log locations and required readout fields without
depending on chat history. It correctly distinguishes a workflow demonstration
from calibration or production qualification.

Usability and integration observations:

- During preparation, the documented `recall --query "critic anchor"` emitted roughly 22,000 tokens
  because matching metadata admits many weak matches. A compact default with an
  explicit result limit would better serve a new agent's context budget. The CLI
  gained compact output and `--limit` during this walkthrough; final verification
  with `--limit 3` succeeded.
- A tier-1 plan lists all higher-tier tasks before its 900-second total. The
  `permitted_by_tier_cap` flag is accurate, but placing permitted tasks and the
  requested backend/cost summary first would make the next action easier to see.
- During execution, the CLI gained compact table output and board filters. A
  helper that assumed JSON without `--json` failed to parse the table; the actual
  board command succeeded. `board --help` identified the explicit `--json`
  option, and `--candidate ... --scope pinned` produced the focused table above.
- The compact table initially showed separate CPU/CUDA current rows with the
  same printed candidate/revision but no runtime column. The underlying JSON
  preserves cohorts. The coordinator added a Compute column, and the final
  focused table explicitly displayed `cpu: AMD Ryzen 9 5900X 12-Core Processor`.
- Concurrent knowledge-module edits briefly broke readout's automatic compile
  and an explicit compile with `NameError: group is not defined` in
  `_historical_row`. The readout record and queue `concluded` lifecycle had already
  persisted. After repair, repeating the same readout and compile commands
  succeeded. The queue still reported exactly 5.974550719 seconds, zero reserved
  time and one attempt; no training was repeated.

These are output/readability observations; this walkthrough does not modify the
CLI implementation or silently substitute a test backend.

# Released v0.7 baseline: explicit Forge task adaptation

The released GAN v3 baseline is now executable across all **24 required toy
tasks** through an explicit task-adapted successor. Its first measured smoke
gate **FAILS**. Integration readiness and scientific qualification are separate.

The [14-technique leaderboard](technique-inventory-unblocked.md) appends this
measured row to the original 12 techniques and Modern GAN baseline. Previous
publications, verdicts and costs remain unchanged under their recorded sources.

| Technique / cohort | Smoke | Quality | Endurance | Evidence |
| --- | ---: | ---: | ---: | --- |
| Original released v0.7 matched-MoG card | 0/3 | 0/19 | 0/2 | 21 integration BLOCKED, 3 UNKNOWN; no ordinary execution |
| Released v0.7 task adaptation | 0/3 | 0/19 | 0/2 | 1 measured FAIL, 23 UNKNOWN; zero integration blockers |

## What was unblocked

The original declaration fixed one native host's batch, dimensions, particles,
7k schedule horizon and objective fields. Every behavioral smoke host rejected
those explicit values before compute reservation. Removing arbitrary overrides
would have hidden a change to the baseline.

The successor retains that complete reference recipe and adds a versioned
`host_adaptation` declaration. A shared binder delegates a value only when the
card lists it and the adapter already owns it. The same binding runs during
planning, reservation validation and construction. Unlisted resource conflicts
remain blocked. Training mechanisms, policies, sampling laws and gates cannot
be delegated. Contract changes enter scientific identity and cannot borrow
another revision's receipts or state.

The released BCap coefficient **6**, cap **1.25**, betas **(0, .99)**, LR
**.00425**, prior multiplier **2**, cosine schedule and disabled K3P hooks/noise
remain fixed. Scalar trainer hosts retain prior spread **.05**. Behavioral
component hosts retain their original objectives instead of importing that
scalar spread term. Dimensions, batch, particles and horizon come from each
frozen task. Applied receipts and report JSON record the actual delegation.
This is a host adaptation of the release recipe; its original native diagnostic
failures retain their separate identities and supply no qualification credit.

The [binding guide](../../docs/forge-task-recipe-adaptation.md),
[idea card](../../configs/forge/ideas/release07-gan-v3-task-adapted-v1.json), and
[immutable campaign](../../configs/forge/campaigns/release07-gan-v3-task-adapted-v1.json)
document the configurable contract and reproduction commands.

## Measured result

One ordinary `two_pole` execution completed all **80 updates**, with **24
observations**, on CPU in the CUDA-requested runtime cohort. The fixed
direct-particle/stored-critic initialization and particle-cloud exception remain
the task's original law.

| Metric | Required | Observed | Gate |
| --- | --- | ---: | --- |
| Particle movement, `mean_abs` | >= 0.30 | 0.29372087121009827 | FAIL |
| Median critic gradient, `grad_med` | <= 1.0 | 0.3106328547000885 | PASS |
| Sustained joint pass | At least 5 terminal passing checks | 0 passing observations / 0 suffix | FAIL |
| Finite state and intended optimizer updates | Finite; both roles complete 80 | PASS | PASS |
| Named RNG isolation | No unintended deviations | 0 | PASS |

The movement shortfall is **0.00627913**. The complete joint curve never passes;
the small final margin does not override the frozen sustained gate.

Actual penalty applications: **80**, coefficient **6** throughout. Both active
optimizer roles record beta2 **.99** and LR **.00425 -> .000222220833**. This
host trains sample particles directly, with no latent table: its particles use
the observed network LR. The configured prior multiplier applies to latent-table
hosts. There is no scalar prior-spread objective or scored EMA on `two_pole`.
Anchor, guard, A2, direct gain and input/output noise remain disabled.

Paid cost: **5.508408383 seconds**, one attempt, no infrastructure errors and
zero remaining reservation. The ordinary scheduler stopped at the required
failure, leaving **23 UNKNOWN** cells and no downstream spending. The campaign
has a 44,100-second ceiling; it did not spend that allowance. A concluded queue
status does not imply a scientific pass.

## Provenance and regeneration

- Executed source: `fc66d7c49259a8977cbb4478b20dad1ffcc6c1cb`.
- Source digest: `d091ec1f6b1db7cfad160b8dcd77f4b6dff432e88dab0c7421c59cff3da997dc`.
- Candidate revision: `d5341d8d12528f9e34030480544dd84fa7b933eed8474ff19e568dfb56b26cbc`.
- Request: `f2ef545b54285551f616bbdd`; attempt: `df1794ebea6444c5a10c2a953843257d`.
- [Compact observed metrics and bindings](release07-task-adaptation-run.json).
- [Validated receipt summary](technique-receipts/df1794ebea6444c5a10c2a953843257d.json).
- [Original envelope/source/log archive manifest](release07-task-adaptation-archive.json).
- [Frozen 14-row source publication](release07-task-technique-inventory.md).

Raw envelopes, stdout, per-update streams, source snapshot and original full
readout stay in the ignored local archive. Compact summaries cannot supply gate
evidence. Hydrate byte-exact originals to independently regrade the recorded
source. Composing the committed publications needs no raw hydration or training:

```sh
python reports/forge/regenerate_technique_inventory.py --root . \
  --compose-original reports/forge/technique-inventory-expanded.json \
  --compose-current reports/forge/release07-task-technique-inventory.json \
  --append-candidate release07-gan-v3-task-adapted-v1 \
  --output-prefix reports/forge/technique-inventory-unblocked
```

Live logs remain easy to inspect:
`python -m experiments.forge logs --follow --campaign release07-gan-v3-task-adapted-v1`.
The experiment is concluded; following logs does not restart it.

## Validation and next action

Forge/release-parity tests: **795 passed**. Focused final reservation/binding and
history tests: **65 passed** (overlap with the full suite). These cover all 24
task preflights, dispatch resource binding, scalar prior-spread retention, AE
routing, reference preservation, rejection of mechanism delegation, changed
scientific identity and recursive publication integrity. Forge declarations
validate 47 tasks and six views. The history catalog includes the new binder
and repairs classification gaps from the latest develop integrations, without
changing their scientific outcomes.

Keep this integration and stop training this exact failed revision. Review the
provisional tier-1 movement/convergence protocol before choosing another
training hypothesis. A justified quality comparison after smoke rejection needs
a separately registered, bounded diagnostic lane. The near miss supplies no
reason to relax a threshold, repeat a seed, continue this run, fill downstream
cells automatically or adopt a public default.

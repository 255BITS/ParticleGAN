# Schedule-preserving acquisition budgets

These diagnostics ask whether settling longer fixes BCAP's scalar mean/CDF drift
or ring16 broad tails. They retain each original host, learned MoG, initialization,
clean/live sampling law and numerical thresholds. Passing these tasks supplies no
ordinary Tier 1 qualification or calibrated default claim.

| Task | Updates | Original schedule horizon | Original numerical checks | Additional terminal checks | Timeout |
| --- | ---: | ---: | --- | --- | ---: |
| `gaussian1d_acquisition_3000_schedule1000_diagnostic_v1` | 3,000 | 1,000 | Original 24 | 2,600–3,000 every 100 | 360 s |
| `ring16_acquisition_1600_schedule400_diagnostic_v1` | 1,600 | 400 | Original 24 | 1,200–1,600 every 100 | 1,200 s |

The public `GANTrainer.max_steps` bounds execution separately from
`Recipe.total_steps`; extending duration does not stretch LR, input-noise or
output-noise schedules. The adapter keeps the original host's `steps` metadata and
original target/data stream. It adds no prefix measurements or sampling draws.
The new gate requires all five additional observations to meet every original
bound and reports the original-prefix result separately, including failure.

At the original endpoint, the receipt hashes the complete context except the
larger external execution cap: learned parameters, optimizer and penalty state,
the original recipe horizon, initialization, every named RNG stream (including
evaluation), prefix metrics and scored draws. CPU contracts compare these hashes
and samples against actual unextended public-API runs. Preflight checks the
source-bound original task card and rejects changes to its host, prior,
initialization, scorer, sampling law or bounds.

Queue admission and dispatch repeat those checks against the frozen source
snapshot before accepting the longer execution budget. An unmarked mismatch or
changed schedule, original cadence, architecture, prior or numerical bound is
rejected even when a caller recomputes job hashes. Admission tests call the actual
`Queue.submit` path and verify that it constructs no model, launches no attempt
and reserves or charges no training time.

Use the separately scoped `bcap_budget_diagnostics_v1` view with the campaign's
registered candidate and shared queue. Inspect the cost before enqueueing:

```sh
python -m experiments.forge plan CANDIDATE --view bcap_budget_diagnostics_v1 --through-tier 1
python -m experiments.forge enqueue CANDIDATE --view bcap_budget_diagnostics_v1 --through-tier 1 --campaign CAMPAIGN
```

Raw JSON observation lines remain in each local attempt's `run.log` for tailing;
arrays and checkpoints stay outside Git. After the coordinator certifies an
attempt, render its already-scored samples without additional training or draws:

```sh
python -m experiments.forge.tier1_media ATTEMPT_DIRECTORY --output LOCAL_MEDIA_DIRECTORY
```

Publish the compact receipt and actual-training GIF alongside the aggregate
readout. Recommendations must distinguish changed duration from any independently
searched recipe changes; preserve historical original-budget verdicts.

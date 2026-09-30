# Existing-control mode screen

Prepared **two `mode_hold` cells only** under the unchanged `host-profile-transfer-v1` profile. Each cell and candidate has a 1,800-second ceiling; the campaign reserves at most 3,600 seconds. Planned execution is serial on GPU0 with one worker and no sharing. Nothing was enqueued or trained during preparation.

| Existing control | Pinned revision | Selected change | Cell |
| --- | --- | --- | --- |
| `forge-onboarding-anchor-ablation` | `d297bd4d9cf9` | `reg_anchor_weight=0` | `mode_hold` |
| `forge-no-critic-penalty` | `5bbe11c14ddc` | `reg_coeff=0` | `mode_hold` |

Both controls were already declared in the profile and their current outcomes are unknown. This lane measures their cheapest first predicate without predicting positivity. It selects no K3P rerun, native task, additional candidate or whole-matrix execution. Inspect both results before any further cells; leave both candidates open for that decision. Scientific failures remain failures; BLOCKED/errors remain unknown. A pass here alone cannot establish the full smoke predicate, an independent reference positive, qualification or calibration adoption.

The frozen task retains 1,200 updates, 24 observations, five required stable checks, coverage >=8 modes and HQ >=.9, live scoring and clean public sampling. Seed 0, learned-MoG sigma .025, named streams, host, complete three-smoke/sixteen-reference profile and criteria are unchanged. GPU index is an operational selection; the scientific hardware cohort remains NVIDIA RTX A6000.

Registration `host-profile-control-mode-v1` has SHA-256 `0c3688b09c50b8dbb5fc036180bbf82fc5940b84f0f45036540faa1f4eec7c51`. Source is `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`; cohort is `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`. Both candidate revisions, task/public preflights, existing frozen source snapshot and immutable registration/request bindings verify. Neither revision has an abandonment or supersession disposition.

Review the [contract](../../../configs/forge/campaigns/host-profile-control-mode-v1.json), [registration](../calibration-lanes/host-profile-control-mode-v1/registration.json), and [preflight evidence](control-mode-v1-preflight.json). The evidence records complete identities, keys, budgets and plan summaries. Re-registering the identical contract preserved the registration bytes. Existing profile, criteria, ideas and scientific source bytes were verified unchanged.

Commands for the coordinator after review, from the feature worktree (not executed here):

```sh
python -m experiments.forge calibration-lane plan host-profile-control-mode-v1
python -m experiments.forge calibration-lane enqueue host-profile-control-mode-v1
python -m experiments.forge drain --gpus 0 --workers-per-gpu 1 --campaign calibration-host-profile-control-mode-v1
python -m experiments.forge logs --follow --campaign calibration-host-profile-control-mode-v1
```

Inspect exact receipts, gate reasons, mechanism counters and measured costs after the two cells. Publish a batch readout before deciding whether any additional bounded comparison is warranted; keep the exact revisions open until their intended work is finished. This preparation makes no lifecycle conclusion or automatic follow-on authorization.

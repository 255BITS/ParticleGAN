# Fixed-data failure investigation

These records preserve the investigation that preceded the corrected full run.
They use the same expert training worlds and validation worlds throughout;
changes concern losses, dynamics data/features, or update budgets, not seed
search. The original test cohort became diagnostic after its failures were
inspected. The corrected run declares a fresh final test cohort before training.

`original_report.json` and `original_slow.pt` preserve the first release.
`training_replay.json` reproduces its flyaway on an exact training reset.
`original_budget.json` shows that longer RpGAN training alone worsened it.
`loss_ablation.json` separates adversarial and dynamics losses.
`calibrated_world.json` compares raw versus engine-power action features with
the same real simulator branches. `calibrated_slow_1200.json`,
`calibrated_slow_2400.json`, `calibrated_fast_400.json`, and
`calibrated_pair.json` report the corrected model's bounded training schedule.

[Contact scoring correction](contact_correction/README.md) archives the
calibrated checkpoints and proves that the three later apparent incomplete
landings were false negatives from cached Gym contact flags. All earlier
tables retain their historical scoring; they have not been silently rescored.

Paths inside these historical JSON records identify the original local scratch
artifacts, some of which are intentionally not committed. The historical
`reports/lunar_fast/slow.pt` reference means `original_slow.pt` here. Original
outcomes used the old classifier, which mislabeled incomplete settled landings
as crashes; strict success counts are unchanged. Every recorded flight used
live weights. The old recipe's `ema_decay=0.995` field was unused metadata.

See the [causal analysis](../../../docs/lunar-failure-analysis.md) for the
evidence and its limits. At 2,400 updates the corrected policy still drifts;
this is not a claim of unlimited adversarial-training stability.

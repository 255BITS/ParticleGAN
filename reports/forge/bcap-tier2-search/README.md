# BCAP optimizer, loss and retention search

This preregistered search evaluates **72 BCAP-on configurations** through the
unchanged **6/6 Tier 1** gates, then scores every survivor on the **21 Tier 2**
requirements. It covers all eight public BCAP optimizer families and all five
adversarial losses, with extra DualNorm retention tuning. The finite target is
at least ten Tier 2 passes on one complete configuration.

The [study](STUDY.md) declares hypotheses, exact scope, gates, budgets and
stopping. [The plan](plan.json) verifies all 72 configurations are READY;
[the manifest](compiled.json) freezes the finite population, every draw,
source/runtime bindings and sampler states. Preparation submits no jobs.
All 87 Forge search-space/configuration-search tests passed, and repository
validation passed before launch.

[run.py](run.py) admits the entire population and drains one isolated queue on
both A6000 GPUs, one worker per GPU. It disables per-completion publication
callbacks and evaluates all results only after every submission concludes.
Required failures block higher tiers automatically after runnable current-tier
peers finish. There is no mid-run intervention, adaptive expansion or retry.

Execution and bulk artifacts use the persistent local data drive:

```sh
python -u reports/forge/bcap-tier2-search/run.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-search-v1 --gpus 0,1 \
  > runs/software/bcap-tier2-search/run.log 2>&1

# Optional human inspection.
tail -F runs/software/bcap-tier2-search/run.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-search-v1/events.jsonl
```

The wrapper creates `runs/software/bcap-tier2-search/completed.json` after the
joint evaluation, or `failed.json` on an operational exception. All numerical
results, explanations, artifact provenance, actual-training media and
recommendations will be published after completion. The
[existing technique inventory](../technique-inventory.md) remains the single
leaderboard for the goal. Archived outcomes keep their original source and
initialization identity. No public-default adoption or Tier 3 claim follows.

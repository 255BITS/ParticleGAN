# R1/R2 Modern GAN configuration search

The bounded four-configuration search found a smoke-qualified R1/R2 toy-host
configuration: learning rate **0.0085**, penalty gamma **1 → 0.1**. It passes
all three smoke requirements, then fails the first ordinary quality requirement,
`trajectory`. Its recorded tier is **1**. The
[single current leaderboard](technique-inventory.md) contains its tier counts;
the [search report](configuration-search/r1r2-modern-toy-v1.json) retains every
trial, task status, selection binding and cost. This is a provisional screen,
without calibrated robustness or public-default adoption.

Trajectory identity MSE is **0.239116475** against the required **≤ 0.02** after
400 updates. None of its 24 observations passes. Forge stops at that prerequisite
failure: 18 other quality tasks and both endurance tasks remain unmeasured.
UNKNOWN does not mean an execution error; denominators remain three smoke,
19 quality and two endurance tasks.

## Comparison and explanation

The [declared grid](../../configs/forge/searches/r1r2-modern-toy-v1.json) varies
only learning rate and paired penalty endpoints on the shared Modern GAN trainer.
All trials share one scientific source, runtime and protocol. Architecture,
task horizons, initialization, named random streams, task-declared priors and
sampling laws remain fixed across configurations. This includes trajectory's
explicit particle-cloud prior exception; its gate does not share the smoke
tasks' MoG sampling law.

All configurations retain plain Adam, beta1 zero, beta2 0.9 → 0.99, a 20% cosine
burn-in, and R1/R2 regularization every update. This transfers training settings
to existing toy hosts; it does not reproduce the paper's image architecture,
Gaussian latent prior, 10-million-image budget or EMA evaluation. The
[original baseline readout](R3GAN_BASELINE_READOUT.md) records that scope and the
earlier learning-rate 0.0002 failure. That unchanged experiment was not repeated;
its different source remains historical evidence rather than a ranked trial.

Three configurations stop at `two_pole` after 80 updates:

- LR 0.00425, gamma 1 → 0.1: movement 0.145771667 is below 0.30;
  gradient median 0.241196170 passes its ≤ 1 threshold.
- LR 0.00425, gamma 0.1 → 0.01: final movement 0.379952937 and gradient
  median 0.693737507 pass individually, but only three consecutive observations
  pass. The required sustained suffix is five, so this remains FAIL.
- LR 0.0085, gamma 0.1 → 0.01: movement 1.006283998 passes, but gradient
  median 1.020285249 exceeds 1 and the sustained gate fails.

The selected configuration passes `two_pole` with movement 0.812555611 and
gradient median 0.619865417, including an 11-observation passing suffix.
It also passes `unused_token_hold` with concept movement 0.999979496 and unused
hold 0.997301101, and `ae_gan_hold` with hold 0.064195588 and reconstruction
MSE 0.006500413. Selection used complete required smoke PASS counts and a
configuration-content hash tie-breaker. Later-tier observations did not alter
selection, and no leaderboard cell combines different configurations.

## Budget, confirmation and recommendation

Tuning used **six attempts, 770 updates and 37.819526426 paid seconds** against
its declared 3,600-second campaign ceiling. The preregistered
[winner-only confirmation campaign](../../configs/forge/campaigns/r1r2-modern-toy-confirmation-v1.json)
reused all smoke evidence and spent **6.636296235 seconds** on one 400-update
trajectory attempt. Total new paid cost is **44.455822661 seconds**, seven
attempts and 1,170 updates, with no execution errors or remaining reservations.

The compact [confirmation receipt](r1r2-modern-toy-confirmation.json) separates
the campaigns. The search report's tuning `status: PASS` and `qualified_winner`
cover only its declared tuning tier. Its frozen selection-stage
`independent_confirmation: not_performed` does not summarize the subsequent
campaign; that failed ordinary later-tier check appears in the trial's task
outcomes and the separate confirmation receipt. The configuration remains Tier 1.

Keep this frozen configuration as the current smoke-qualified R1/R2 entry and
stop its exact failed revision. Before expanding the grid, review the provisional
convergence screen and task/prior contracts against qualified reference evidence.
A short toy-host adaptation failure does not establish failure of the paper's
full recipe. Changing the recipe after examining these quality tasks requires
a new selection protocol.

## Provenance and regeneration

Execution source commit: `cd0ba9feba7aae79723f8581fe29eb4bd3c9050b`.
Scientific source digest:
`f23a47fab9058c4e16590c5c8ebf6a8bbed9f1d6b5406d4107db395096658f54`.
Selected configuration:
[`r1r2--abf642c42c5346ad096c29202e4716db535c393c113478552133c1c22761ddbd`](../../configs/forge/configurations/r1r2--abf642c42c5346ad096c29202e4716db535c393c113478552133c1c22761ddbd.json),
revision `7c08dcbf6b2e1a90368368ba328f3597a260381ca68b67f2ac28904050ce3ee1`.

The ignored local archive contains original execution envelopes, full readouts,
source snapshots, worker artifacts and logs. Its exact SHA256 and original
receipt hashes are in the [archive manifest](r1r2-modern-toy-archive.json).
Committed projections contain final metrics and provenance only; they cannot
serve as qualification inputs. Publication independently regraded the original
evidence under its recorded source and runtime.

Cached publication requires no training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

The [configuration-search guide](../../docs/forge-configuration-search.md)
documents the public plan/enqueue/run/report commands. This execution used
`search run r1r2-modern-toy-v1`, followed by the frozen winner through ordinary
Tier 3 prerequisites. Hydrate exact archived originals before a full regrade:

```sh
python reports/forge/regenerate_technique_inventory.py --source-commit cd0ba9feba7aae79723f8581fe29eb4bd3c9050b --device cpu
python reports/forge/regenerate_technique_inventory.py
```

Local execution logs are easy to tail:

```sh
tail -f runs/forge/r1r2-modern-toy-search-v1/search-run.log
tail -f runs/forge/r1r2-modern-toy-search-v1/confirmation-run.log
python -m experiments.forge logs --follow --campaign r1r2-modern-toy-search-v1
```

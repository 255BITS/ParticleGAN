# AE, VAE, DDGAN and AE-DDGAN API research

Eight new public-API toy cases cover four API model/encoding modes. They use
one global recipe, the existing selected `bispf-optimized-quintic-v1`, with
protocol seed 0 and the public deterministic initializer. No optimizer, loss,
learning-rate, prior-update or prior-regularizer search is proposed. Each task
owns its architecture, target law, initial MoG capacity/width and model objective;
the recipe retains learned prior updates and SBNP regularization.

The research proposals and controls are frozen separately for [AE](ae.md),
[hard/categorical VAE](vae.md), [DDGAN](ddgan.md), and [AE-DDGAN](ae-ddgan.md).
The [bounded protocol](protocol.json) preserves every task contract and source
hash before execution. Six basic tasks enter required Tier 1 of the existing
`discriminator_stability` view. The two bimodal chain tasks enter Tier 2 with
their own Gaussian gate dependencies, reflecting their larger target and
conditional-generation claim. No gate is weakened or task moved after observing
training. Initial placement remains provisional; this is not calibrated default
adoption.

Each task has 24 numerical observations and requires the last five to pass.
Oracles and destructive controls check the scorers before training. The full
DDGAN tasks learn real forward-pair transitions and sample four reverse steps;
direct posterior toys cannot fill these cells. AE-DDGAN separately scores its
paired encoded-input chain and an independent ancestral chain with fresh latent
and reverse noise at every step.

The standalone bounded research cohort supplies its own CPU measurements. It
does not pool with the older CUDA family row or qualify a whole recipe against
the main suite. All six basic tasks finish before deciding eligibility for the
two deeper runs. Any basic failure blocks full Tier 2 research; an explicitly
budgeted three-update software prefix still supplies an actual-training GIF for
each blocked implementation, labelled INCOMPLETE against its full task.

Reproduce after committing the scientific setup:

```sh
PYTHONPATH=. /path/to/python reports/forge/api-model-toys/run.py \
  --output /outside-git/api-model-toys-run --jobs 3
tail -F /outside-git/api-model-toys-run/*/stdout.log
```

Outputs are exclusive: an unchanged run cannot be overwritten or retried.
`--export-only` reads saved receipts and GIFs without training or sampling.
Raw stdout, metric curves, arrays and complete model/optimizer/policy/RNG
checkpoints stay outside Git. The compact supplemental publication and saved
actual-training GIFs feed the existing [Forge leaderboard](../technique-inventory.md)
and [experiments by tier](../EXPERIMENTS_BY_TIER.md); this goal has no second
generated leaderboard. Historical AE, VAE and DDGAN evidence keeps its original
source, seed, initialization, budget and serving contract.

The MoG replay tests exposed a public `UpdatePolicy` checkpoint validator that
assumed every module state entry was a tensor. Its narrow metadata fix validates
immutable `_extra_state` before mutation. The two current word declarations
receive an explicit source-only binding layer because they pin that shared file;
the [migration receipt](policy-checkpoint-migration.json) retains their original
Git blobs, task hashes and source hashes. No word run or qualification is changed
or rerun.

Full research results are pending the committed source freeze. Software checks
already cover exact checkpoint replay, evaluation isolation, independent
training streams, declared recipe ownership, full-chain execution and scorer
counterexamples. These are software evidence, not trained-quality passes.

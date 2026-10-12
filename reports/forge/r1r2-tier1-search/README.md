# R1/R2 Tier 1 repair search

This search tests global trainer policies against the [recorded baseline](../r1r2-baseline/README.md). It preserves the original task gates and separates ordinary Tier 1 qualification from the optional clock measurement. See the [saved-state diagnosis and research sources](evidence-diagnosis.md).

## Registered comparison

Round 1 compares one fresh matched scheduled baseline and six constant policies. The structural delta removes LR decay, beta2 annealing and penalty annealing together. The finite grid uses beta1=0, beta2=0.9 or 0.99, and constant R1/R2 coefficient 0.1, 0.3 or 1. Both real and fake L2 critic-input penalties run every update. Native Adam, base LR 0.0085, critic multiplier 1, learned prior, prior multiplier 1, prior regularization 0, input/output noise 0 and EMA 0 are held fixed.

Each candidate uses one global recipe across Gaussian acquisition, two-pole transport, unused-token hold, AE/GAN hold, Ring16 sustained density and five-word acquisition. Its own clock probe is a separate measurement. A solution must pass all six ordinary gates and the clock probe; task winners cannot be combined into a recipe.

Protocol seed 0, public deterministic initializer, architectures, target/data laws, actual seen batch sequences, initial prior distribution/capacity, sampling, update caps and evaluation cadence remain fixed. Scalar/vector, behavior and word adapters now record passive initial tensor hashes, ordered consumed target hashes and named RNG checkpoints. This new evidence source justifies a fresh matched baseline; it does not replace archived observations. Fixed parameter and identity fixtures keep their declared separate cohort identities.

Current task cards explicitly refresh the hashes of the instrumented adapter sources. The [source-binding receipt](protocol-source-bindings.json) records every old/new task-file and adapter-source hash, with equal before/after hashes after removing only `evaluation.sources`. All task laws, gates, cadence and evaluator algorithm revisions remain unchanged. Archived legacy task cards and their evidence retain their original bindings.

## Budget and stopping

The [round manifest](phase1.json) declares seven recipes and 17,640 seconds of full task reservations: 2,520 per recipe, comprising 2,220 required-task seconds and 300 clock seconds. Actual charged work is capped at 3,600 seconds. Each candidate completes its runnable current-tier peers before a required failure blocks further tiers. There are no seed variations, scientific retries, Tier 2 jobs or automatic refinements.

At most one separately registered second round may follow a concluded first-round readout if a specific diagnosis supports another global delta. The aggregate ceiling is twelve recipes, 30,240 reservation seconds and 5,400 charged seconds. Unspent limits are ceilings, not a request to consume them.

Artifacts remain outside Git under `/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012`, with an 8 GiB archive cap and 5 GiB free-space floor. Workers use GPUs 0 and 1, one worker per GPU, and the same CPU/thread/runtime settings. Costs under shared execution are accounting, not a speed ranking. Compact final metrics and provenance belong here; raw logs, per-update streams and checkpoints remain in the archive.

## Reproduction

Use the repository Python environment from the repository root. Materialize the immutable finite cards, prepare the registration, and commit every scientific input before admission:

```sh
python -c 'from pathlib import Path; from experiments.forge.configuration_search import materialize_search; materialize_search(Path.cwd(), "r1r2-tier1-constant-search-v1")'
python reports/forge/r1r2-tier1-search/workflow.py prepare
python reports/forge/r1r2-tier1-search/workflow.py run --source-commit COMMITTED_SHA --submit --drain
python reports/forge/r1r2-tier1-search/workflow.py collect --media
```

Tail execution with:

```sh
tail -f /mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/run.log
```

The collector checks certified numerical gates, source/recipe/task bindings, actual initialization and consumed-batch equality, charged costs and saved-training GIF provenance. Rendering adds no training or sampling draws. The repository's single current generated technique inventory remains the leaderboard; this research report does not create a second generated leaderboard or adopt defaults.

Results will be appended after the registered comparison concludes. Generated experiment memory and its compilation hash are excluded from this PR at the user's request.

# Forge image performance diagnostics

The [current technique inventory](../reports/forge/technique-inventory.md#performance)
contains the single current performance overview. Its Performance section measures
speed, memory and execution limits of complete public `GANTrainer.step` calls.
These measurements answer whether a recipe can execute a larger workload within
the stated resources. They supply no image-quality, scientific qualification or
default-adoption evidence. A completed finite update budget is the diagnostic's
numerical requirement, not a trained-image quality gate.

## Fixed workload and recipe ownership

The [diagnostic view](../configs/forge/views/dcgan_performance.json) declares six
tasks. Each uses RGB images, latent dimension 128, 256 initial prior locations,
seed 0, three warmup updates and ten measured updates:

| Task | Image size | Batch size | Base channels / channel cap |
| --- | ---: | ---: | ---: |
| `dcgan_performance_r32` | 32×32 | 16 | 64 / 512 |
| `dcgan_performance_r64` | 64×64 | 16 | 64 / 512 |
| `dcgan_performance_r128` | 128×128 | 16 | 64 / 512 |
| `dcgan_performance_r256` | 256×256 | 16 | 64 / 512 |
| `dcgan_performance_b64` | 64×64 | 64 | 64 / 512 |
| `dcgan_performance_w128` | 64×64 | 16 | 128 / 1024 |

The generator uses ConvTranspose2d, BatchNorm, ReLU and a Tanh RGB output. The
critic uses Conv2d, BatchNorm and LeakyReLU with scalar logits. All recipes use a
shared final 4×4 flatten plus Linear scalar head, allowing Atlas/E22 feature
extraction. Its forward function class and parameter count equal a final 4×4
scalar convolution, while DualNorm treats the dense head differently from a
convolution's per-offset normalization. This module layout is fixed across all
recipes. The 64-pixel, base-64 convolutional stack follows DCGAN. Larger resolutions add stages while keeping
the declared channel cap. Width and resolution are consequently different tasks;
parameter counts are recorded. The repository's public deterministic orthogonal
initializer replaces DCGAN's conventional random Normal initialization.

Standard BatchNorm remains in both networks. Critic input-gradient regularizers
therefore differentiate a batch-coupled training-mode function. The public trainer
uses its ordinary discriminator mode during generator updates. This architecture
is a declared performance workload; it does not establish the behavior of a
critic with independent per-example normalization.

Mock targets contain independent RGB pixels sampled Uniform[-1, 1]. Every update
has separate fresh critic and generator batches. The same task uses identical
architecture, initialization, target law, batch sequence, prior distribution and
capacity, budget and evaluation cadence across recipes. Constructor, target data,
training noise, prior sampling and evaluation use named, isolated streams. Saved
checkpoints contain consumed streams and the data cursor; receipts preserve initial
model and batch-sequence hashes. A small actual-training GIF is an untimed artifact
illustrating execution, and grants no quality credit.

The main six-task cohort uses the **same explicit particle-cloud prior** for every
recipe: sigma 0, no standardization. This performance-only finite-table exception
allows the selected Atlas/E22 controllers and the other selected recipes to run
under one common initial prior. The exception reason is in each task. Prior update
policy, regularizer, optimizer, losses and controller behavior remain recipe-owned.
No recipe is simplified to improve throughput. One selected global recipe per
family runs across all tasks; there is no per-resolution recipe selection.

The separate CPU development smoke task uses a small positive-width MoG workload.
It validates portability cheaply and is a separate task/prior/architecture cohort.
Its timings cannot fill a main-cohort cell. Unsupported objectives or controller
capabilities remain explicit unsupported/blocker outcomes. New tasks or revisions
are needed for another prior, dtype, architecture, budget or timing protocol.

## Timing and memory scope

Each measured update uses synchronized host wall time around the complete public
trainer update. It includes generator, discriminator, prior and recipe controller
work. Construction, mock-data creation/transfer and hashing, finite-state checks,
evaluation, GIF generation and checkpoint writes are outside the timed region.
The receipt records total run wall time and first-update/warmup measurements
separately. Images/s is `batch_size / mean_update_seconds`, where one update
includes both critic and generator/prior work; it is not the sum of all network
forward images. Median and p95 use the ten measured updates. The p95 is a short
empirical window, not a confidence bound or sustained-training guarantee. Startup
and early update costs do not establish long-horizon controller/policy costs.

CUDA peaks use PyTorch's allocated and reserved memory statistics, with the peak
counters reset after warmup and read before evaluation/checkpoint copies. They
include resident model, optimizer and current-batch memory; they omit CUDA context,
driver, other processes and non-PyTorch allocations. CPU GPU-memory fields are
unavailable. Component phase timing is unknown in v1; a total update measurement
cannot locate a Sinkhorn, SVD or gradient-regularization bottleneck by itself.

Every displayed cell is one exact run. The publication never averages multiple
devices, retries, sources or contention states. Its matched comparison key includes
the entire task specification, exact source file map, runtime/dtype, physical GPU
UUID/PCI identity, initializer, initial state and batch-sequence hashes. GPUs of
the same model retain separate cohorts. Contention snapshots before and after
execution exclude the benchmark's own process. They are point-in-time observations,
not continuous exclusivity proof. Unknown or contended measurements remain visible
and unpaired. Use one worker per physical device and record other activity.

OOM, unsupported capability, execution error, nonfinite result and incomplete
budget remain distinct from completion. Partial timing observations stay attached
to their failed/incomplete run. Missing cells remain unknown. A timeout during a
single expensive update can leave only the worker's raw error evidence; it cannot
be displayed as a completed benchmark. Soft task limits are checked between
updates; the Forge worker enforces the hard reservation deadline.

## Bounded execution and logs

The current execution registration is `dcgan-performance-cloud-v2`, frozen from
commit `58ade47791367abd46c68ede8d5e7d9fea3dbafb`. Its GPU tasks reserve 24 GiB
each, independently of the measured allocator peaks. The earlier v1 registration
retains its original task/reservation bindings; its unstarted requests were
cancelled for telemetry and reservation corrections with zero paid execution.
It cannot be replayed against the revised task contracts. Choose a new
registration ID for future execution after any source or declaration change.

Read `EXPERIMENTATION.md` and compiled memory first. Registration freezes the
currently selected complete recipes and creates ready diagnostic studies with a
finite campaign reservation. A new source, task or candidate declaration requires
a new registration ID. Planning and registration launch no training. Run from the
repository root in the project's Python environment:

```sh
python -m experiments.forge --queue-root runs/forge-scaling scaling register image-performance-next --device cuda
python -m experiments.forge --queue-root runs/forge-scaling scaling plan image-performance-next
python -m experiments.forge --queue-root runs/forge-scaling scaling enqueue image-performance-next
python -m experiments.forge --queue-root runs/forge-scaling drain --gpus 0 --workers-per-gpu 1 --campaign image-performance-next
```

Use a single GPU for a complete comparable recipe matrix. A second GPU can run a
separate cohort. The plan prints the exact per-job and campaign allowance; the
campaign cap includes every selected recipe's full reservation. Reuse compatible
completed evidence instead of rerunning unchanged tasks. Seed-only repeats are
excluded.

In another terminal:

```sh
python -m experiments.forge --queue-root runs/forge logs --follow --campaign dcgan-performance-cloud-v2
tail -F runs/forge/events.jsonl
```

Each run's stdout prints one flushed JSON line per update. Tail the run's
`stdout.log` or `updates.jsonl` using the paths in its queue receipt. Raw stdout,
event/update streams, checkpoints, tensor dumps and media remain in ignored local
storage or an artifact archive. Commit compact final receipts, source/protocol
declarations and reports; never force-add bulk logs.

## Compact publication without training

Publish a completed registration directly from its queue evidence:

```sh
python -m experiments.forge --queue-root runs/forge scaling publish dcgan-performance-cloud-v2
```

This command validates the exact frozen requests, jobs, source maps, candidate
and task bindings and independent grading receipts. It projects final core fields
from the queue's canonical raw result, because collection replaces the runner's
`result.json` with the canonical queue receipt. Compact content-addressed receipts
retain original file byte hashes, attempt and request IDs, frozen job identity,
registration family, current inventory display family, artifact root and bulk
artifact hashes. BCAP's registration-family label does not change its actual
selected recipe or current inventory ownership. Bulk logs, checkpoints and GIFs
stay in their original local/archive storage. Missing bulk originals keep their
recorded hashes and are explicit as unavailable locally.

By default publication refuses queued, running or unsubmitted cells. Use
`--allow-partial` only to request a labelled partial snapshot; pending timings stay
unknown. Hard timeouts, unsupported capability and errors without a core summary
produce compact queue-outcome receipts with no fabricated model or metric fields.
The same Performance section displays those outcomes and reasons. Repeating
publication with unchanged evidence is deterministic. It never launches or retries
training and never writes another leaderboard.

For importing already archived compact summaries, the lower-level workflow below
remains available.

Copy each run's compact final `result.json` to a stable path under
`reports/forge/scaling-receipts/`; preserve its source, recipe, initialization,
runtime and artifact hashes. The adapter annotates candidate/task identity and
Forge execution bindings. The full source file map names the actual performance
revision, which may differ from the selected recipe's older scientific source.
No old scientific evidence is regraded or substituted.

Create an entries document, for example in ignored `runs/forge-scaling/entries.json`:

```json
{
  "entries": [
    {
      "trainer_family": "bispf",
      "candidate_id": "bispf-v2",
      "task_id": "dcgan_performance_r64",
      "receipt": "reports/forge/scaling-receipts/bispf-r64.json"
    }
  ]
}
```

An optional `tasks` array of `{ "id": "...", "spec": { ... } }` declares cells
that have no receipt yet. Use the exact normalized `execution.performance` task
specifications. The reducer rejects mismatching task/source/runtime/recipe hashes,
changed receipt bytes, duplicate current cells and different global recipes
across a candidate's tasks. Choose the current receipt explicitly when replacing
a failed attempt; retain original attempt evidence in its artifact archive.

```sh
python -m experiments.forge.scaling_publication --entries runs/forge-scaling/entries.json
python -m experiments.forge.scaling_publication --check
python reports/forge/regenerate_technique_inventory.py --refresh-performance
```

These commands validate and render committed evidence without training, artifact
hydration or original-source extraction. `scaling-performance.json` is the compact
evidence index, not another leaderboard. `--refresh-performance` updates only the
existing overview's Performance block and display provenance, preserving scientific
rows, gates, archived text and family pages. Ordinary publication regeneration
also includes the section and adds filtered performance evidence to family pages.
The old scientific qualification and default-selection files remain authoritative.

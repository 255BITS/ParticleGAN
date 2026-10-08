# BCAP convolution support and four-image rerun

The original [Tier 2 readout](../bcap-tier2/README.md) records four image
`INCOMPLETE` results before training: DualNorm rejected convolution weights.
This source-bound adaptation enables the
[per-offset kernel update](../../../docs/dualnorm-convolution.md) through
`Recipe.optimizer_convolution="per_offset"`.

The single global recipe retains G/E `.012`, D `.018`, sampled-prior `.03`,
smoothing `1e-5`, zero momentum and constant learning rates. Each kernel group
and spatial offset uses a channel-matrix polar update scaled by
`sqrt(out_channels/in_channels)/(kernel_height*kernel_width)`. Transposed
convolution uses its actual input/output channel layout. Dense, vector and
sampled-prior updates retain their existing rules.

The [candidate](../../../configs/forge/ideas/bcap-dualnorm-convolution-v1.json),
[ready study](../../../configs/forge/studies/bcap-convolution-images-v1.json)
and [diagnostic view](../../../configs/forge/views/bcap_convolution_images.json)
select `img_stripes2`, `img_bars4`, `img_blobs4` and `img_intensity2`. Each runs
its unchanged 600 updates, 24 observations, exact finite-prior enumeration,
clean/live scoring and sustained gates. The campaign reserves at most 7,200
worker seconds, 1,800 per task, and completes all four peers after a failure.
Protocol seed is 0 with the public deterministic initializer and isolated,
checkpointed named RNGs. No rate annealing, tuning, seed repeat or retry is
declared.

This explicitly scoped capability-repair diagnostic retains the original
Tier 2 task definitions. Its scheduler uses diagnostic Tier 1 placement solely
to execute the selected subset; it does not promote tasks or qualify a new
source. Original Tier 1 passes, Tier 2 failures and the four setup errors stay
under their original source. No Tier 3 run or public-default adoption follows.

The image adapter retains the outputs already consumed by each scheduled
scorer. Offline publication will recompute their numerical metrics and render
target/output GIFs under guards that prohibit learned model construction,
forward passes, training and sampling. Bulk evidence stays in an ignored,
byte-verified local archive; compact final metrics and provenance are committed.

## Reproduction and logs

Run from the repository root with the project environment:

```sh
python reports/forge/bcap-convolution/run.py --stage plan
python reports/forge/bcap-convolution/run.py --stage enqueue
python reports/forge/bcap-convolution/run.py --stage drain --gpus 0,1
python reports/forge/bcap-convolution/publish.py
```

Execution is frozen from the merged develop source. Reproducing later requires
the executed source revision; publication-only changes do not authorize a rerun.

```sh
tail -f runs/forge/events.jsonl
tail -f runs/bcap-convolution/drain.log
```

The [current technique inventory](../technique-inventory.md) remains the sole
goal leaderboard. These four-image diagnostics cannot fill its ordinary
qualification cells.

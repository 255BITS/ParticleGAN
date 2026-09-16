# Running and reusing experiment grids

Run from the repository root with the environment used for training:

```bash
python experiments/run_grid.py --configs 'configs/*.yaml' \
  --python .venv/bin/python --gpus 0,1 --workers_per_gpu 2
# TOML uses the same flat experiment configuration schema:
python experiments/run_grid.py --configs 'configs/100gaussians/*.toml' \
  --trainer experiments/train_100gaussians.py --python .venv/bin/python --gpus 0
```

Both TOML and YAML inputs are supported. The runner checks every config before
launching children. Invalid TOML/YAML, unknown
keys (for trainers declaring `DEFAULTS`), nonpositive worker counts, and duplicate
or nested output directories fail the grid. Independent runner processes use a
POSIX file lock to prevent simultaneous writes to the same output directory.
An unrelated nonempty directory and directories containing source/config inputs
cannot be archived as run outputs.

Completed results are reused only when `run_grid_complete.json` matches all of:

- The full requested configuration, including the trainer's current defaults.
- The trainer, runner, shared config reader, and local Python sources under
  `lib/` and `particlegan/` (plus primary-trainer dependencies).
- The resolved Python executable path and the saved summary's SHA-256 digest.

The manifest is written atomically after a successful child exit, a nonempty
`summary.json` final block, and an exact match of its recorded `config`. Legacy
results without a completion manifest are preserved and rerun. A source edit
during a grid prevents certification of affected runs. This fingerprint covers
local source; it does not record installed dependency versions, hardware, or
hash every generated artifact. Use a fixed environment for reproducible studies.

Before a rerun, previous output files are moved intact to
`<out_dir.parent>/.run_grid_history/<out_dir.name>/<timestamp>-<id>/`. The new
attempt begins in a clean directory. `--force` uses the same archive behavior.
Failed attempts retain their logs; a summary written by a failed child is renamed
`summary.failed.json` and cannot be reused. Other queued jobs continue, failures
are logged in `results/failures.txt`, and the overall command exits nonzero.

Custom trainers must accept `--config`, report the complete effective config in
their summary, and declare a `DEFAULTS` mapping or receive every config value
explicitly in their input. Defaults can be literal values or unpack the shared
`recipe_defaults("100gaussians")` / `recipe_defaults("denoising")` helper into
a dictionary with literal overrides. The runner reads this narrow syntax using
Python's AST parser; it never imports the trainer or evaluates arbitrary trainer
expressions. This lets the primary trainers inherit package recipe defaults
without executing their training code during validation.

Regardless of input format, the runner passes an immutable, fully resolved copy
named `requested_config.yaml` to the child. This YAML file remains the canonical
runner request artifact; changing the original TOML or YAML while a run is queued
does not change that run's request.

## Sparse pipeline

```bash
PY=.venv/bin/python GPUS=0 WORKERS=2 experiments/sparse_pipeline.sh
BASE='coeff=0.3,total_steps=1000' experiments/sparse_pipeline.sh recipe sparse
```

The default stages are `recipe ucd sparse discrete champion fewshot`; `smoke` can
be requested explicitly. Generation, training, and analysis errors stop the
pipeline. Its log is `results/sparse/PIPELINE.log`.

Config precedence is trainer defaults, stage BASE (where applicable), group
overrides, then `--base`/`BASE`. Thus `BASE='coeff=0.3'` also overrides a group's
coefficient sweep. `--total_steps`/`STEPS` supplies the fallback run length; named
short/long groups keep their own length. Use `BASE='total_steps=1000'` to override
every group's length. The generator owns `seed` and `out_dir`; select seeds with
`--seeds`/`SEEDS`.

Nondefault BASE, seed lists, or step settings select a deterministic namespace
such as `sparse__<hash>` under both `configs/sparse/` and `results/sparse/runs/`.
Default invocations keep the historical stage names. Use the generator's
`--print_stage` option to get the resolved namespace without writing files, and
pass that name to `analyze_sparse.py --stage` when analyzing manually. The
pipeline does this automatically. A generated `manifest.json` lists the exact
config paths for the invocation; `run_grid.py --config_manifest <path>` excludes
stale YAML files that may also be present in the config directory. The pipeline
passes the same manifest to `analyze_sparse.py --config_manifest <path>`, so
obsolete result groups are excluded too. Manifest analysis checks each summary's
full config against the current request/defaults and exits nonzero when any
listed run is missing or mismatched. Omit the manifest for historical directory
scanning when analyzing earlier studies manually.

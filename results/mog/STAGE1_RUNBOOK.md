# Reproduce the Stage 1 optimizer pilot

The primary acceptance thresholds are frozen in
`configs/mog/stage1_criteria.json`, committed before the first pilot result.
They use the observed C0 range; the historical criterion remains recorded as
`passed_strict`, and C0-mean comparison appears in the analysis outputs.

```bash
.venv/bin/python experiments/gen_mog_configs.py --stage stage1_lr
.venv/bin/python -u experiments/run_grid.py \
  --configs 'configs/mog/stage1_lr/*.yaml' \
  --trainer experiments/train_100gaussians.py \
  --gpus 0,1 --workers_per_gpu 2 > results/mog/stage1_lr.runner.log 2>&1

.venv/bin/python experiments/analyze_mog_stage1.py --phase lr
.venv/bin/python experiments/gen_mog_configs.py --stage stage1_beta
.venv/bin/python -u experiments/run_grid.py \
  --configs 'configs/mog/stage1_beta/*.yaml' \
  --trainer experiments/train_100gaussians.py \
  --gpus 0,1 --workers_per_gpu 2 > results/mog/stage1_beta.runner.log 2>&1

.venv/bin/python experiments/analyze_mog_stage1.py --phase final
```

Inspect progress without selecting from incomplete cells:

```bash
tail -F results/mog/stage1_lr.runner.log
tail -F results/mog/stage1_beta.runner.log
tail -F results/mog/stage1/lr_n100_m0p1_s1/log.txt
.venv/bin/python experiments/analyze_mog_stage1.py --phase progress
```

The grid runner certifies source hashes, complete configs, and summaries.
Reusing completed runs requires the same training sources and config. The six
beta=0 configs point to their existing LR-sweep outputs and are reused rather
than retrained. Source archives are stored with each run. The analysis script
requires all 30 LR runs before choosing multipliers and all 36 unique runs
before writing the final Stage 1 report.

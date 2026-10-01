# Coordinator commands

Preparation is static (source/data hashes, AST syntax/formula checks and read-only GPU inventory); it runs no initialization, training, replay or quality test:

```bash
/tmp/pr38-default-env/bin/python -B /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/prepare.py
```

Root's serialized launcher captures logs and executes the following CLI. Scripts install the frozen GPU0/thread/determinism environment before scientific imports. Run one command at a time, and serialize this lane with original-harness screens.

```bash
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/run_training.py --problem toy --variant E22
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/run_training.py --problem toy --variant CB64-RA
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/run_training.py --problem mnist --variant E22
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/run_training.py --problem mnist --variant CB64-RA
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/replay.py E22
/tmp/pr38-default-env/bin/python -B -u /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/replay.py CB64-RA
```

After all saved learned/replay evidence exists, generate report/results/manifest (no quality execution):

```bash
/tmp/pr38-default-env/bin/python -B /ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/summarize.py
```

Training outputs: `training/<problem>/<variant>/config.json`, `metrics.jsonl`, ten `checkpoint-*.pt`, `result.json` or `error.json`, and MNIST evaluator/sample receipts. Replay outputs: `replay/<problem>/<variant>/branch-{0,1}.pt` and `result.json`/`error.json`, plus aggregate `replay-<variant>.json`. Reporting outputs: `results.json`, `REPORT.md`, `artifact-manifest.json`.

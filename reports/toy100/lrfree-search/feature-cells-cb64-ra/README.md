# CB64-RA experimental backend

Frozen E22-derived package with 64 rank-8 feature cells and bounded real-anchor parents. This archive makes the tested experimental implementation and configuration available in Git.

**Status: experimental.** The completed CPU diagnostics found a reaction-cost improvement and passing correctness checks, alongside mass, geometry and learned-model quality regressions. The restored-GPU retest is complete: 9/13 portability tests and 0/3 native tests pass; learned MNIST recall regresses. Evidence validity passed 660/660 checks. See the [GPU report](cuda/REPORT.md).

- [Configuration](configs/overrides-CB64-RA.json) and [backend source](pkg-CB64-RA/particlegan/feature_cells.py)
- [Usage and checkpoint instructions](USAGE.md)
- [Frozen CPU validation report](REPORT.md), [leaderboard](leaderboard.json) and [audit](review/audit.json)
- [Implementation protocol](implementation/PROTOCOL.md) and [focused contract tests](implementation/test_integration.py)
- [Copied-file manifest](ARCHIVE.json)

The report is preserved from the CPU phase and describes CUDA access as unavailable at that time. CUDA access was restored on 2026-09-29; the new GPU study is separate at `/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929`.

## Load the archived package

Select `pkg-CB64-RA` before importing `particlegan` in a fresh Python process. Load `configs/overrides-CB64-RA.json`, supply your generator and scalar-head critic, and set particle count, latent dimension and batch size for your problem. In the usage example, replace the original experiment root with this archive directory. The package retains `knn` as its default backend; the experimental JSON explicitly selects `feature_cells`.

## Reproduction artifacts

All copied files match their frozen originals byte for byte. Checkpoints, datasets, logs and large raw geometry buffers remain in the original experiment directory recorded in `ARCHIVE.json`. Some links in preserved phase reports refer to those local-only artifacts.

For the focused CPU contracts from this directory:

```bash
/tmp/pr38-default-env/bin/python implementation/test_integration.py
```

The recorded result is 8/8 passing contracts. The GPU retest runs the unchanged config on the original CUDA harness, paired E22/CB64-RA learned training and checkpoint replay. Its completed runners, compact receipts and independent audit are preserved under `cuda/`. Corrected CB64-RA2 diagnosis and validation are a separate experiment.

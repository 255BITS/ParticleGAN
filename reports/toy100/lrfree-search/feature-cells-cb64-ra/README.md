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

The recorded result is 8/8 passing contracts. The GPU retest runs the unchanged config on the original CUDA harness, paired E22/CB64-RA learned training and checkpoint replay. Its completed runners, compact receipts and independent audit are preserved under `cuda/`.

## Corrected candidates and joint quality target

[The correction archive](fixes/README.md) contains the subsequent experimental
packages, matching configs, focused diagnostics and CUDA results. The target
requires one package passing both the unchanged learned toy and full canonical
grid gates. No completed candidate qualifies yet.

RA8 passes the final toy: precision 0.965332, all 25 modes and mass TV 0.052114.
All ten saved training states match RA7 exactly apart from the declared serving
metadata and performance counters. Its full grid is independently VALID/FAIL;
center, radial and covariance errors remain. The actual positive serving lease
also passes a separate exact CUDA sample/reload check. These findings are an
experimental checkpoint, and the default package remains unchanged. See the
[toy/grid leaderboard](fixes/quality/REPORT.md) for final gates and the
[usage guide](fixes/USAGE.md) for package and checkpoint scope.

RA9 retains the passing toy with exact training parity and uses a real-fit
sample cap to permit128 cells on grid. Its complete grid is VALID/FAIL:
all five terminal clouds pass every check except center RMS (0.211–0.222
sigma, bound0.20). The independent100k holdout fully passes. The final grid
has precision0.98210, all100 modes, TV0.03075, radial KS0.01856 and maximum
covariance ratio1.61258. This is still an experimental failure of the joint
target; small saved-state diagnostics are investigating the remaining mean
bias. See the [completed RA9 receipt](fixes/quality/results/CB64-RA9.json).

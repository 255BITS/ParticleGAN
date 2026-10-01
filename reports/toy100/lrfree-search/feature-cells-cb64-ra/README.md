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
grid gates. RA11 is the first candidate to pass both original CUDA quality
gates. The complete19-job study is independently VALID: all three native tests
pass and portability is8/13 PASS.
Exact CUDA replay passes for toy and MNIST. The completed MNIST comparison
regresses severely to active feature distance40.54441 and zero recall,
compared with E22's0.54449 and84.72% recall. RA11 leads the toy/grid target;
it is not recommended as a general base package.

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

RA10 adds a bounded conditional feature-mean copy phase within the existing
shared 5% budget. Independent CPU integration and short CUDA mechanics pass;
the original final CUDA toy passes and all ten checkpoints are VALID. The full
grid is VALID/FAIL: final center RMS improves to0.19825 sigma, but maximum
covariance ratio1.77312 exceeds1.7. All five terminal covariance checks fail;
the first four also fail center RMS and the frozen holdout fails coverage.
Final precision is0.97945, all100 modes and TV0.03185. Historical abbreviated
action lists receive count/scalar checks; full lineage/reset/row-ID validity
is checked in the final saved state. The fixed saved-output diagnostic found
that observation noise substantially reduces nonlinear feature residuals
despite a small paired raw mean increment and no learned-group transitions.
See the [completed RA10 receipt](fixes/quality/results/CB64-RA10.json).

A fixed CPU prototype measuring raw output means accepted914 grid copies,
reduced EMA squared conditional mean error40.4% and centered covariance trace5.0%,
and conservatively vetoed the new phase on the toy. RA11 is selected from
this result with genuine backend schema10. Affected state controls and short
CUDA mechanics pass. Its original final CUDA toy passes, with all ten saved
checkpoints VALID. Full Grid100 validation passes all five terminal checks and
the independent100k holdout. Final grid precision is0.98250, all100 modes,
TV0.03280, center RMS0.16661 sigma, radial KS0.01376 and maximum covariance
ratio1.34756. Holdout precision is0.98453 and center RMS0.13973 sigma.
RA11 is the validated joint quality winner. MNIST and five portability failures
prevent a general base-package recommendation. See the [validated joint quality
receipt](fixes/quality/results/CB64-RA11.json), [completed validation](fixes/quality/results/CB64-RA11-regressions.json),
and [fixed MNIST diagnosis](fixes/quality/ra11/mnist-diagnosis/counts/REPORT.md).

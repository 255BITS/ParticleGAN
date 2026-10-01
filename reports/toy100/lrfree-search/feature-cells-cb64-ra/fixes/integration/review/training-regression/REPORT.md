# Count recovery proposal, frozen v4

This private proposal changes ordinary transport only when the unchanged isolation guard rejects a flagged set larger than Q times the table population. In that branch it replaces only flagged rows in cells with existing exact categorical excess evidence. Births require existing categorical deficit evidence, actual supported vacancies, and distinct eligible supported parents. Supported rows never supply deaths in this branch. The per-call ordinary budget remains floor(.05 times N); topology-group target caps reserve supported mass. The original guard and support/count laws are unchanged.

When the isolation guard passes, the v3 action arrays and planning RNG remain exact. The package advertises `reference_topology_vacancies_unique_parents_v4` for checkpoint compatibility. Its only difference from frozen RA2 is `particlegan/feature_cells.py`; `COUNT-RECOVERY.patch` is the reviewable change.

The saved CPU toy reconstruction yields four count-certified flagged replacements at each saved step: v3 yielded zero at 1000 and one at 2000. There are zero supported deaths, four distinct parents, and zero isolation actions in those broad-flag cases. These reconstructed snapshots preserve their existing projections, comparison evidence and CPU planning streams; they are not reconstructed CUDA random states.

Four focused recovery regressions pass, along with all eight existing stability contracts. The standalone runner's CPU check passes seven cases, including the original nominal and rare-hole inputs with 46 unchanged isolation repairs, saved toy 1000/2000, no eligible parent, and flag-count guard boundaries 51/52. It checks budgets, unique children/parents, unchanged guard behavior, supported targets, exact v3 actions/RNG while the guard passes, and rare-survivor preservation. All receipts report no CUDA context and unchanged package inputs.

`READY.json` freezes package sources, patch, runner, v3 method reference and the existing evidence. No prior evidence was regenerated during this freeze. This proposal remains limited to fixed-input accounting: it has no learned-quality qualification and leaves the separately diagnosed within-cell support blind spot unresolved.

Root may run the fixed-input CUDA contract once in its single physical GPU0 queue:

```bash
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/recovery_gpu_check.py --package-root /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/pkg-count-recovery --output /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/training-regression/recovery-gpu-frozen --device cuda:0
```

The runner also accepts a composed package root, records its source hashes before and after, and uses the frozen v3 method for within-device comparisons. Root owns composition and canonical matched toy/MNIST/replay tests. A separate support-aware count proposal must use a new private directory and must not modify this v4 artifact.

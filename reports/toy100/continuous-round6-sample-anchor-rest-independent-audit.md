# Sample-anchor rest guard: independent audit

The frozen guard preserves the previous sample-anchor candidate exactly on the converged path. Across the three saved-state branches, all 44 guarded updates have `nonconverged_fit_rested=false`. After removing only method/source metadata and that new Boolean, the six branch JSON results are identical to the original sample-anchor results: supports, grades, accepted-state hashes, full update records, correction costs and fit traces, applied rates, Adam counters, RNG, and noise receipts. The exact raw results and 24 source files are in the [saved44 archive](continuous-evidence/round6-sample-anchor-rest-independent/saved44/manifest.json).

Two forced-failure tests exercise the new branch. In one actual native cold update, a naturally converged fit was followed by deterministic perturbations to both a G weight and prior `z`, then marked nonconverged. The guard restored every G/prior parameter bitwise to its pre-G value. D, both Adam states, EMA, RNG, noise, and the native curvature record matched the original PR84 update; each optimizer still took one moment step at fixed rates 0.00425 D, 0.00425 G, and 0.0085 prior. This stops before EMA/checkpoint and is an integration test, not a cold episode.

The second test resumes the archived 1324 pre-step state through update 1332. Its first eight corrections match the frozen success branch exactly. At 1332 the native G proposal improves the native-batch anchor objective from 0.0025639977 to 0.0024872871, so the old fallback would retain the native proposal if the joint fit failed. With forced nonconvergence, the new guard instead selects `rest` and restores every G/prior parameter exactly to pre-G; the final sampled objective equals 0.0025639977. This specifically validates the changed behavior. The [forced-run receipt](continuous-evidence/round6-sample-anchor-rest-independent/forced/manifest.json) includes both tests, generated host source, and the log.

The audit introduces no warm or cold qualification. The guard handles a returned nonconverged fit status; an exception still aborts the episode. The sampled group estimator, additional anchor objective, and nonlinear fit remain outside the original PR84 game, and the fixed-center free-output argument does not establish a stochastic neural invariant region.

Run the focused tests with one CPU thread and the frozen original capture:

```bash
SAMPLE_ANCHOR_REST_ROOT=/ml2/hypergan/ParticleGAN-continuous-learning SAMPLE_ANCHOR_REST_CAPTURE=/ml2/hypergan/ParticleGAN-stationary-diagnosis/artifacts/continuous-learning/stationary-failure-replay/capture-v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m pytest -q tests/test_sample_anchor_rest_independent.py
```

All three focused tests pass. The [audit driver](pr84_sample_anchor_rest_independent_audit.py) independently checks source hashes before running either failure test.

No release candidate. Executed 9/16 proposals in batches 3/2/2/1/1; all 9 failed warm_probe_200. No cold, own-state continuation, full19, native100 or saved-sample qualification gates were run.

| Rank | Proposal / change from H | Dense passing / observed | First failing step | HQ at failure | Timed seconds | Verdict |
|---|---|---:|---:|---:|---:|---|
| reference | Pinned H | 54 / 55 | 1255 | 0.780518 | 1.371 | FAIL |
| 1 | [s02_mixup_0p1](signal_attempt/s02_mixup_0p1/warm/summary.json): Mixup coefficient 0.1 | 54 / 55 | 1255 | 0.812012 | 1.477 | FAIL |
| 2 | [s07_local_curvature_0p001](signal_attempt/s07_local_curvature_0p001/warm/summary.json): Add local curvature 0.001, displacement 0.1 | 44 / 45 | 1245 | 0.736328 | 1.389 | FAIL |
| 3 | [s04_r1r2_0p3](signal_attempt/s04_r1r2_0p3/warm/summary.json): R1+R2 coefficient 0.3 | 27 / 28 | 1228 | 0.899414 | 1.084 | FAIL |
| 4 | [s08_g075_r1_only](signal_attempt/s08_g075_r1_only/warm/summary.json): G/prior rates x0.75; real-only R1 0.6 | 25 / 26 | 1226 | 0.839600 | 1.091 | FAIL |
| 4 | [s09_g075_confidence_r2](signal_attempt/s09_g075_confidence_r2/warm/summary.json): G/prior rates x0.75; confidence-weighted R2, scale 0.25 | 25 / 26 | 1226 | 0.881836 | 1.124 | FAIL |
| 6 | [s01_r1r2_1p2](signal_attempt/s01_r1r2_1p2/warm/summary.json): R1+R2 coefficient 1.2 | 17 / 18 | 1218 | 0.862549 | 0.920 | FAIL |
| 6 | [s06_antithetic1](signal_attempt/s06_antithetic1/warm/summary.json): Antithetic score pair, noise 0.05 | 17 / 18 | 1218 | 0.883545 | 1.162 | FAIL |
| 8 | [s03_score_l2_0p01](signal_attempt/s03_score_l2_0p01/warm/summary.json): Add score L2 0.01 | 10 / 11 | 1211 | 0.884766 | 0.997 | FAIL |
| 9 | [s05_noise_0p1](signal_attempt/s05_noise_0p1/warm/summary.json): Fixed input noise 0.1 | 3 / 4 | 1204 | 0.880371 | 0.934 | FAIL |

Code: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_runner.py; /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_variants.py; /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/run-signal.sh.
Results and gate coverage: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/RESULTS.md.
Artifacts: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_attempt/<candidate>/warm/ (actual source archive, config/options, metrics/losses, final model/optimizer/RNG state, raw terminal samples, work/rate receipt).
Ledger: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/tests.jsonl.
Summary: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_attempt/summary.json.
Source/state parity audit: /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_attempt/audit.json.
Tests: 70 passing unit-test executions, 37 distinct tests, zero failures; exact-control/source audit PASS. Two control warm executions reproduce first failure1255/HQ0.780517578125 with bitwise-identical full state; all125 pinned sources unchanged. Setup errors:0. Full suite skipped because no viable candidate.
Work: 231 proposal dense checks, 462 total G+D Adam commits, 10.178 summed timed probe seconds, excluding process startup/source snapshots/final receipts. s01–s07 rates: G/D.0015, prior.003. s08/s09 fixed rates: G.001125, D.0015, prior.00225. All Adam(0,.999). Nine proposals retired; unused budget is intentional.

Replay from `/ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo`:
```bash
bash reports/toy100/h_stability/run-signal.sh --declaration reports/toy100/h_stability/signal_attempt/batch-01.json --candidate s02_mixup_0p1 --output reports/toy100/h_stability/signal_attempt/replay-s02 --ledger reports/toy100/h_stability/signal_attempt/replay-tests.jsonl | tee reports/toy100/h_stability/signal_attempt/replay-s02.log
```
Tail: `tail -F /ml2/hypergan/gan-attempts/stability-20260924T172714Z/stability_signal/20260924T172714Z-1520130/repo/reports/toy100/h_stability/signal_attempt/s0*.log`.
Regression replay: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 /tmp/pr38-default-env/bin/python -m pytest -q tests/test_h_signal_variants.py tests/test_regularizers.py tests/test_continuous_candidates.py tests/test_legacy_noise_adapters.py tests/test_legacy_noise_remaining.py`.

Remaining failures: H's six older gates and own-state stability remain unresolved; native gates remain unrun. AE has no encoder adversarial path; unused-token symmetry implies retention>=.85 forces movement<=.30. Two supervisor-directed extensions used verified peer rate evidence, without importing peer gate passes; both failed at1226. No auxiliary targets were restored. Retain H only as the acquisition baseline; no further coefficient sweep is justified by these results.

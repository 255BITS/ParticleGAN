# RA9 completed grid — canonical VALID, quality FAIL

The original complete grid schedule finished7000 updates with34 observations,
five terminal20k clouds and an independent100k holdout. The unchanged
canonical collector accepted source/init/prior-range/data/stream/fixture validity.
Raw and canonical quality are bothFAIL. No gate was reinterpreted.

| Terminal step | Center RMS / sigma | Original bound | Coverage | Accuracy |
|---|---:|---:|---|---|
| 6000 | 0.222497747 | <=0.20 | PASS | FAIL |
| 6250 | 0.215073156 | <=0.20 | PASS | FAIL |
| 6500 | 0.212694353 | <=0.20 | PASS | FAIL |
| 6750 | 0.211112518 | <=0.20 | PASS | FAIL |
| 7000 | 0.211927465 | <=0.20 | PASS | FAIL |

The original holdout passed every requirement: P0.98173, massTV0.01909, center0.192687033sigma, radialKS0.014529296, absolute covariance-trace bias0.016773572. The terminal center failures retain the complete gateFAIL.

All98 saved grid artifacts were hashed and verified unchanged. Exactly1/16
canonical screens completed; the remaining15 stayPENDING with unverified
fixtures and no quality verdict. Learned toy separately passed its original
gate, so the combined toy-and-grid target remains unmet.

After completion, only the owned CPU watcherPID860452/start166179427 was
stopped with oneSIGTERM after two exact command/start/all-thread-nochild
checks. It terminated. The parked RA4 queue and every other watcher/GPU
process were untouched. No regression jobs were started.

`receipt.json` records the independent read-only auditPASS and distinct
original qualityFAIL. The copied canonical receipt retains its original
fields and SHA; `GRID-ARTIFACT-MANIFEST.json` binds saved artifact bytes.
`STOPPED-AT-GRID-FAILURE.json` records ownership checks and the pending state.
`FROZEN.json` is the authoritative post-exit seal of closed logs, sources and
saved evidence. This audit made no draws, emissions, training, scoring calls,
model imports or production edits.

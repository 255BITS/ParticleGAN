# Draw-free generator-step publication

This exporter consumes root's separately completed `combine_studies` certification. Root must finish both immutable family studies and perform the explicit CPU-only capacity/retained-trace recertification before binding a card and supplying its trusted SHA256. `bind-certification` only binds the completed result; it does not run or establish certification.

The fixed new contrast is `{lr: 0.00265625, prior_lr_mult: 3.0, d_lr_mult: 4.5}` for Atlas and E22, each with the original eight required cases. Publication retains all 16 cells, zero-update capacity outcomes, scientific FAIL and unavailable execution states, and separate original/first-window hold grades beside every unchanged GIF. Native scope is 24 post-update checks of 20,000 noisy served outputs; clean diagnostics remain separate and this cohort provides no independent 100,000-output certification.

The original 15,360-second campaign includes exactly one prior debit of 59.116680497769266 seconds: 54.3587717928458 seconds for the two previous full-600 scientific failures and 4.757908704923466 seconds for a separate startup ERROR. The exporter verifies the prior published result, certification, combined result, two family studies, public receipts, artifacts, source snapshots and durable costs. New paid intervals and conservative interruption reservations are separate; prior failures grant no new-cell credit.

Before exporting, copy and commit `publish_generator_step.py` at `reports/forge/generator-step-20261003/publish_generator_step.py` in the publication checkout, with `test_generator_step_publication.py` in `tests/`. Keep the scientific checkout frozen at its original run source. The exporter checks both its own current bytes against its publication commit and the exact scientific source/snapshot manifests, including the new runner, new binder, unchanged delegated critic helper and pinned ring discovery JSON. Current capacity artifacts and all explicitly hash-bound construction inputs are consumed as bytes without opening model states.

Run these commands from a fresh process with no previously imported ParticleGAN checkout. Replace `COMMITTED_PUBLICATION_CHECKOUT` with the checkout containing the frozen exporter, and `TRUSTED_CARD_SHA256` with the independently supplied root certification-card digest. Output paths must be new and outside the source and raw family/snapshot trees.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
/COMMITTED_PUBLICATION_CHECKOUT/reports/forge/generator-step-20261003/publish_generator_step.py bind-certification \
  --combined /ml2/hypergan/forge-generator-step-20261003/combined.json \
  --scientific-root /ml2/hypergan/ParticleGAN-generator-step-20261003 \
  --output /ml2/hypergan/forge-generator-step-20261003/certification.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
/COMMITTED_PUBLICATION_CHECKOUT/reports/forge/generator-step-20261003/publish_generator_step.py publish \
  --certification /ml2/hypergan/forge-generator-step-20261003/certification.json \
  --certification-sha256 TRUSTED_CARD_SHA256 \
  --publisher-root /COMMITTED_PUBLICATION_CHECKOUT \
  --output /ml2/hypergan/forge-generator-step-20261003/publication-v1
```

The publisher emits `results.json`, a readable 16-cell `README.md`, byte-identical goal GIFs with paired verdict captions, and a full hash-bound `input-index.json`. A zero exit means evidence publication succeeded; it grants no scientific PASS, fastest-family, shipping/default or old-cell qualification. The publisher itself performs zero model constructions/restores, sampling calls, numeric rescoring, optimizer updates, GPU/queue/lease operations or training. It rechecks all raw inputs after export.

Software controls use only temporary synthetic source/evidence/GIF bytes and a stubbed scientific predicate oracle. They preserve the real file, artifact, snapshot, cardinality, publication, cost and source-binding logic; no scientific training or Git mutations occur.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python -m pytest -q \
/ml2/hypergan/generator-step-publication-build-20261003/test_generator_step_publication.py
```

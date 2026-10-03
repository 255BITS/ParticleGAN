# Draw-free critic contrast publication

The publisher consumes root's separately produced `combine_studies` JSON and an
explicit trusted certification SHA. It does not execute that certifier. Root's
certification includes CPU capacity restore/sample replay and pure retained-array
gate verification, with zero ordinary optimizer updates. The publisher separately
checks source Git blobs, snapshots, receipts, artifacts, recorded flag grades and
supervisor costs, and copies the original GIF bytes.

Freeze these files in Git before actual export:

- `publish_critic_balance.py` → `reports/forge/critic-balance-20261003/publish_critic_balance.py`
- `test_critic_balance_publication.py` → `tests/test_critic_balance_publication.py`

From a new CPU-only process, after root has run the original frozen certifier:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/critic-balance-20261003/publish_critic_balance.py \
  bind-certification --combined /EXTERNAL/certified-combined.json \
  --scientific-root /FROZEN/SCIENTIFIC/CHECKOUT --output /EXTERNAL/new-certification.json
```

Keep the printed certification SHA as the explicit trust anchor. Binding a file
is an attestation of the already completed root verification, not proof that the
verifier was run. The publication requires that exact anchor:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python reports/forge/critic-balance-20261003/publish_critic_balance.py \
  publish --certification /EXTERNAL/new-certification.json \
  --certification-sha256 TRUSTED_SHA --publisher-root /COMMITTED/PUBLISHER/CHECKOUT \
  --output /EXTERNAL/NEW-PUBLICATION
```

The card schema is `particlegan_critic_balance_publication_certification_v1`:

```json
{
  "schema": "particlegan_critic_balance_publication_certification_v1",
  "combined": {"path": "/EXTERNAL/certified-combined.json", "sha256": "SHA", "bytes": 123},
  "certifier": {
    "root": "/FROZEN/SCIENTIFIC/CHECKOUT",
    "commit": "FULL_COMMIT",
    "files_sha256": {"relative/source.py": "SHA"},
    "discovery_inputs_sha256": {"relative/discovery.json": "SHA"},
    "function": "combine_studies"
  },
  "verification": {
    "cpu_only": true,
    "ordinary_training_updates": 0,
    "capacity_sampler_replay": true,
    "numeric_trace_rescore": true
  },
  "engineering_carryover": "Exact object from the frozen combined spec, or null"
}
```

The v2 spec's original Atlas startup error is bound through its original study,
request, terminal and log. Its measured cost is projected once, independently of
the 16 scientific cells and the two duplicate spec copies. The remaining Atlas
quota plus original error is 7,680 seconds; the pair's combined ceiling remains
15,360 seconds.

Output is a new external directory with compact `results.json`, readable
`README.md`, copied `media/<family>/<case>/goal.gif`, and `input-index.json`.
The full index stays external; compact results bind its hash/count. Return code
0 means evidence publication succeeded and does not imply scientific PASS.

Software controls:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_critic_balance_publication.py
```

Synthetic tests stub only scientific predicates. They exercise actual file
hashes, source/snapshot binding, saved-grade projection, durable costs, media
frame/copy identity and output ownership without training or model draws.

# Raw evidence and independent verification

The [results](results.json), [questions](QUESTIONS_AND_MEDIA.md), diagnosis and two original goal GIFs are committed for review. Bulk evidence is retained separately in the [Forge archive card](archive.json). Availability is **LOCAL_ONLY**: remote replication is NOT_PERFORMED, retention owner UNASSIGNED, and retention date UNDECLARED.

Archive: `/ml2/hypergan/forge-critic-balance-archive-20261003/critic-balance-capacity16-learning2-v1.tar.gz`, **64,385,103 bytes**, SHA-256 **`292061adad594577d9186207234929afa42437cdf9e5539128d9ca4865a7b533`**.

Its **10,411 regular files** include the original sixteen CPU capacity constructions, all pinned donor inputs, the original pre-training ERROR, both corrected full600 FAIL runs, retained model/optimizer/RNG states and sample arrays, original GIFs, certification and publication outputs, the 3,396-file original and 3,397-file corrected source snapshots, the committed snapshot preflight, and only these three attempts' durable supervisor artifacts. All **6,971 publication inputs** are present. Unrelated queue state, leases, credentials and mailboxes are excluded. The older broad-mode continuation evidence belongs to PR #266's separate `e160ac…` archive.

Every member was checked against its original before packing, individually verified against the archive, and rechecked afterward. The existing Forge resolver independently inspected every member and hydrated the 17-member bootstrap. The [verification receipt](archive-verification.json) records PASS, unchanged originals, zero training updates and zero sampler calls. The full index is **`3f8e917f81bbaa8c9bd3e85613c4b23bd0297532837acc3b5a023f4a456bc41a`**, with original paths, sizes and SHA-256 hashes.

These existing commands launch no training and change no grades:

```bash
python -m experiments.forge artifacts inspect \
  reports/forge/critic-balance-20261003/archive.json
python -m experiments.forge artifacts hydrate \
  reports/forge/critic-balance-20261003/archive.json \
  --destination /tmp/pg-critic-evidence-bootstrap
```

Hydration destinations must be fresh. Default hydration restores the bootstrap and full index. To independently inspect and hydrate every indexed member, use the existing resolver API:

```python
import hashlib
import json
from pathlib import Path
from experiments.forge.artifact_resolver import inspect_archive, hydrate_archive

root = Path.cwd()
card = json.loads((root / "reports/forge/critic-balance-20261003/archive.json").read_text())
data = Path("/tmp/pg-critic-evidence-bootstrap/full-index.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == card["full_index"]["sha256"]
index = json.loads(data)
expanded = {**card, "files": index["files"] + [card["full_index"]]}
checked = inspect_archive(root, expanded)
assert len(checked["files"]) == card["archive_member_count"]
hydrate_archive(root, expanded, "/tmp/pg-critic-evidence-full")
```

On another machine, mount a mirror containing `sha256/<archive-sha256>.tar.gz` and pass `--mirror <directory>` to the CLI or `mirrors=[directory]` to the API. The resolver checks the complete hash and size. No remote location is assumed here.

Hydration restores original bytes under isolated member paths. Scientific JSON retains its original absolute paths. Re-running scientific certification also requires the pinned Git sources/runtime and an isolated mount layout matching the index's `original_path` entries; rewriting JSON paths changes its evidence identity. Archive integrity verification and scientific certification are separate. Root performed the latter at the frozen science source before this publication, including CPU capacity sampler replay and retained-trace rescoring; publication and archive verification perform neither.

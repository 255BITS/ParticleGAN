# Raw evidence and independent verification

The [results](results.json), [questions](QUESTIONS_AND_MEDIA.md), diagnosis and two original goal GIFs are committed for review. Bulk evidence is retained separately in the [Forge archive card](archive.json). Availability is **LOCAL_ONLY**: remote replication is NOT_PERFORMED, retention owner UNASSIGNED, and retention date UNDECLARED.

Archive: `/ml2/hypergan/forge-generator-step-archive-20261003/generator-step-capacity16-learning2-v1.tar.gz`, **111,262,244 bytes**, SHA-256 **`271cf1c21a341d33d520311160c3d9c03f3ffcb6094dd544602e2084089c8949`**.

Its **17,366 regular files** include the sixteen new CPU capacity constructions and all explicit donor inputs, both full600 generator-step FAIL runs, the nested earlier critic cohort and original pre-training ERROR, retained model/optimizer/RNG states and sample arrays, original GIFs, separate certifications and publications, complete 3,396/3,397/3,400-file source snapshots and their copied-source preflights. Only these five completed attempts' supervisor artifacts are included: two current scientific failures, two prior scientific failures, and one prior engineering error. No prior outcome fills a current test cell. All **10,466 publication inputs** are present. Unrelated queue state, leases, credentials and mailboxes are excluded. The older broad-mode continuation evidence belongs to PR #266's separate `e160ac…` archive.

Every member was checked against its original before packing, individually verified against the archive, and rechecked afterward. The existing Forge resolver independently inspected every member and hydrated the 29-member bootstrap. The [verification receipt](archive-verification.json) records PASS, unchanged originals, zero training updates and zero sampler calls. The full index is **`62c42dd3a03570ed023d332e5a629191c30306296478d219c0a64881b65a3696`**, with original paths, sizes and SHA-256 hashes.

These existing commands launch no training and change no grades:

```bash
python -m experiments.forge artifacts inspect \
  reports/forge/generator-step-20261003/archive.json
python -m experiments.forge artifacts hydrate \
  reports/forge/generator-step-20261003/archive.json \
  --destination /tmp/pg-generator-evidence-bootstrap
```

Hydration destinations must be fresh. Default hydration restores the bootstrap and full index. To independently inspect and hydrate every indexed member, use the existing resolver API:

```python
import hashlib
import json
from pathlib import Path
from experiments.forge.artifact_resolver import inspect_archive, hydrate_archive

root = Path.cwd()
card = json.loads((root / "reports/forge/generator-step-20261003/archive.json").read_text())
data = Path("/tmp/pg-generator-evidence-bootstrap/full-index.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == card["full_index"]["sha256"]
index = json.loads(data)
expanded = {**card, "files": index["files"] + [card["full_index"]]}
checked = inspect_archive(root, expanded)
assert len(checked["files"]) == card["archive_member_count"]
hydrate_archive(root, expanded, "/tmp/pg-generator-evidence-full")
```

On another machine, mount a mirror containing `sha256/<archive-sha256>.tar.gz` and pass `--mirror <directory>` to the CLI or `mirrors=[directory]` to the API. The resolver checks the complete hash and size. No remote location is assumed here.

Hydration restores original bytes under isolated member paths. Scientific JSON retains its original absolute paths. Re-running scientific certification also requires the pinned Git sources/runtime and an isolated mount layout matching the index's `original_path` entries; rewriting JSON paths changes its evidence identity. Archive integrity verification and scientific certification are separate. Root performed the latter at the frozen science source before this publication, including CPU capacity sampler replay and retained-trace rescoring; publication and archive verification perform neither.

The current publication cost separates 54.87757138675079 seconds of new science, 54.3587717928458 seconds of prior science and 4.757908704923466 seconds of prior engineering, each once. Cumulative campaign paid time is 113.99425188452005 seconds; interruption reserve is zero and the original ceiling remains 15,360 seconds. No future candidate is included in this frozen archive.

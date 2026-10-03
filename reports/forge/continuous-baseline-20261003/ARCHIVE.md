# Raw evidence and independent verification

The [results](results.json) and [27 GIFs](README.md) are committed for review. The bulk evidence is retained separately in the [Forge archive card](archive.json). Its availability is **LOCAL_ONLY**: no remote copy or retention assignment has been made.

The archive is `159261059` bytes with SHA-256 `e160acbcc6b98e1ea7a4ff928911033515774d6781d0e83ebeedc5a565d892d0`. It contains 11,572 regular files: the full Atlas19 outputs, both preserved H1 startup attempts, both H2 continuations, their original C6 parents, all three complete frozen source snapshots, only these 23 attempts' terminal supervisor records, supplemental retained-cloud exports, the offline publication and its input files. Other workers' queue state, leases, credentials and mailboxes are excluded.

Every member was individually hashed before export, checked against the completed archive and rechecked against its original afterward. The existing Forge resolver then independently verified all 11,572 member hashes and hydrated the 40-member bootstrap. The compact [verification receipt](archive-verification.json) records that result. The archive's `full-index.json` binds each member's SHA-256, size and original path; its hash is `38bb5904845be87d53143c735cb8dd9724532833657f3f82aebf2782093e92e7`.

Use the existing resolver from a checkout containing this report. These commands launch no training and change no grades:

```bash
python -m experiments.forge artifacts inspect \
  reports/forge/continuous-baseline-20261003/archive.json
python -m experiments.forge artifacts hydrate \
  reports/forge/continuous-baseline-20261003/archive.json \
  --destination /tmp/pg-evidence-bootstrap
```

Both destinations in this example must be fresh. The default hydration restores the bootstrap, including the full index. To verify and hydrate every individually declared original, expand that index with the existing resolver API:

```python
import hashlib
import json
from pathlib import Path
from experiments.forge.artifact_resolver import inspect_archive, hydrate_archive

root = Path.cwd()
card = json.loads((root / "reports/forge/continuous-baseline-20261003/archive.json").read_text())
index_bytes = Path("/tmp/pg-evidence-bootstrap/full-index.json").read_bytes()
assert hashlib.sha256(index_bytes).hexdigest() == card["full_index"]["sha256"]
index = json.loads(index_bytes)
expanded = {**card, "files": index["files"] + [card["full_index"]]}
verification = inspect_archive(root, expanded)
assert len(verification["files"]) == card["archive_member_count"]
hydrate_archive(root, expanded, "/tmp/pg-evidence-full")
```

On another machine, place the exact archive in a mounted content-addressed mirror as `sha256/<archive-sha256>.tar.gz`, and pass `--mirror <directory>` to the CLI or `mirrors=[directory]` to the API. A mirror lookup verifies the full archive hash and size; its basename alone does not establish identity. Optional existing location configuration can record the actual storage owner and retention date. None has been assumed here.

Hydration restores original bytes under isolated archive member paths. Original scientific JSON retains its hash-bound absolute paths. Replaying the original certifiers on another machine additionally requires the pinned runtime and an isolated mount/container layout matching the index's `original_path` entries. Rewriting those JSON paths would change original evidence. Archive/member verification and scientific recertification are separate operations; the frozen certifiers already recertified all 19 baseline and two continuation protocols before this publication.

The outer native dispatcher session reported exit 143 after its complete 19/19 log. Every scientific child has a retained zero-return-code terminal supervisor receipt, and the full result files independently passed certification. The outer-session termination cause is unknown. It does not replace those completed scientific results, and no scientific child was rerun.

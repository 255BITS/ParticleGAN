# Closed KA2/K3P prior1 cohort: raw evidence and verification

Bulk evidence is retained separately through the [Forge archive card](archive.json).
Availability is **LOCAL_ONLY**; remote replication is **NOT_PERFORMED**, retention
owner **UNASSIGNED**, and retention date **UNDECLARED**. The archive changes no
qualification or scientific outcome.

Archive:
`/ml2/hypergan/forge-ka2-k3p-defaults-archive-20261003/ka2-k3p-defaults-capacity16-learning2-v1.tar.gz`,
**131,261,240 bytes**, SHA256
**`5271659901bd29e11fddb59486bbdf3ed7a4631da3fa519c62d6222fb2ed7c15`**.

Its **20,897 safe regular members** contain every one of the **13,945 consumed
publication inputs**; the full current named KA2/K3P raw cohort; the earlier
generator-step and critic-v2 raw cohorts; the original pre-model bootstrap
ERROR; all retained capacity/model/optimizer/RNG states, sample arrays, original
GIFs, certifications and publications; four complete canonical source snapshots;
the current frozen readable-media review, retained pair diagnosis, eight-question
explanation and unexecuted next-rate proposal; root's relevant software,
capture, copied-source preflight, plan, run, combination and publication logs;
and committed publication reproducer/tests/docs. All source bytes and consumed
inputs were checked before and after packing.

The source snapshots retain these original, distinct scientific identities:

| Source commit | Snapshot files | Cohort |
|---|---:|---|
| `92167813cb2b04af0c4e0395984c503b7fb7d7fe` | 3,396 | original bootstrap ERROR |
| `a956c6fc447bbe1314ff43ec224825a0e941eb55` | 3,397 | critic-v2 pair |
| `488b792e2fb875894f017cf7043420f2bb66190f` | 3,400 | generator-step pair |
| `26ff278c3796d775969391adc0bde52e3af11149` | 3,404 | named KA2/K3P prior1 pair |

Only **seven completed own supervisors** are included: two current scientific
attempts, four prior scientific attempts and one prior engineering error. The
archive excludes unrelated worker/queue state, credentials, live/future PRIOR2
paths and the unfinished common-index V2. PR266's historical Atlas19/H2 archive
remains separate and is not charged to this campaign. Explicit retained fast
parameter inputs are provenance for capacity construction, never transferred
prior grades or training state.

The current candidate remains `.006375/1/1` for `lr/prior_lr_mult/d_lr_mult` in
each named family. All **16 zero-update capacity outcomes are SUPPORTED**.
Learning retains **2 full original/study FAIL and 14 UNKNOWN** after both
600-update intensity2 failures. Cold snapshot capacity does not imply training
reachability, full-suite solvability, noisy-native terminal capacity, a winning
shared configuration or default adoption.

New paid scientific intervals total **27.193673191126436 seconds**. The separately
bound prior science **109.23634317959659** and original engineering
**4.757908704923466** sum to **113.99425188452005** seconds. Each is debited once:
cumulative paid time is **141.1879250756465 seconds**, reserve **0**, and the
unchanged campaign ceiling **15,360 seconds**. Remaining cap is
**15,218.812074924354 seconds**. Archive verification launches no new scientific
run and adds no reservation.

Every tar member was verified individually against its original bytes. The
existing `experiments.forge.artifact_resolver` independently inspected and
hydrated the **42-member bootstrap**, recovered its bound full index, then
inspected **all 20,897 members** against the expanded index. The
[verification receipt](archive-verification.json) records PASS, unchanged
originals and zero model constructions, draws, scorer calls or optimizer
updates. Full index SHA256:
**`aa9684cde781ff356c6a9c65775240db7b9d591c8b10da569aa7c7d1f3ea14bc`**.
Bulk inspection/hydration receipts and easy-to-tail verification log remain
outside Git at `/ml2/hypergan/forge-ka2-k3p-defaults-archive-20261003` and
`/ml2/hypergan/ka2-k3p-defaults-archive-forge-verification-20261003.log`.

Existing commands launch no training or rescoring:

```bash
python -m experiments.forge artifacts inspect \
  reports/forge/ka2-k3p-defaults-20261003/archive.json
python -m experiments.forge artifacts hydrate \
  reports/forge/ka2-k3p-defaults-20261003/archive.json \
  --destination /tmp/pg-ka2-k3p-prior1-bootstrap
```

Destinations must be fresh. Default hydration restores the bootstrap/full index.
To inspect or hydrate every independently indexed member with the same existing
resolver API:

```python
import hashlib
import json
from pathlib import Path
from experiments.forge.artifact_resolver import inspect_archive, hydrate_archive

root = Path.cwd()
card = json.loads((root / "reports/forge/ka2-k3p-defaults-20261003/archive.json").read_text())
data = Path("/tmp/pg-ka2-k3p-prior1-bootstrap/full-index.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == card["full_index"]["sha256"]
index = json.loads(data)
expanded = {**card, "files": index["files"] + [card["full_index"]]}
checked = inspect_archive(root, expanded)
assert len(checked["files"]) == card["archive_member_count"]
hydrate_archive(root, expanded, "/tmp/pg-ka2-k3p-prior1-full")
```

A mirror must contain `sha256/<archive-sha256>.tar.gz`; use `--mirror` or the API
`mirrors` argument. No remote mirror is presumed here. Hydration preserves
original JSON and isolated archive member paths. Scientific replay would also
require the exact recorded runtime/Git sources and a new isolated layout matching
the full index's `original_path` entries; rewriting evidence JSON is not an
equivalent replay. Root separately certified the scientific inputs before
publication. This archive verifies bytes and availability only.

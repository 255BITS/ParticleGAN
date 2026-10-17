# Final evidence helper v2

The root invokes `finalize_evidence_v2.py` after the corrected moving/native
groups and the actual RA14 full suite close. This stdlib helper reuses the
hash-pinned sealed v1 validators. It reads source, JSON and logs and writes
exclusively to a new child directory under local `release-prep`. It rejects
existing output directories and preserves all original preparation files.

```sh
python -B /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/release-prep/finalize_evidence_v2.py --output /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/release-prep/final-v2-attempt1
```

The required original19 tasks are explicitly routed: the13 completed portability
receipts and scoreboard results come from the retained interrupted
`validation-ra14/scoreboard-all.json`; the3 moving and3 full native receipts come
from `validation-ra14-r2/scoreboard-moving.json` and `scoreboard-native.json`.
The actual suite receipt/log come from the original `validation-ra14` lane.
Each receipt is checked against its own source freeze and the shared package
and config. The old ALL scoreboard remains ERROR; its closed portability scope
is used without changing or relabeling the interrupted attempt.

The helper binds all three pinned source freezes and their actual source/input
file hashes, verifies the identical93 external frozen inputs, and proves the
exact two adapter corrections by source transformation: moving diagnostics use
`trainer.policy.surprise`; native preparation imports and prepends the frozen
harness before runpy. The only freeze-source change binds the correction
receipt. Package/config, model/optimizer arithmetic, hosts, original scorers,
budgets, seeds, streams and gates remain governed by the shared frozen sources.

Original moving500 baseline/error, native0-update import error, failed ALL/native
scoreboards and the separate frozen `validation-ra14-moving-r2` preparation
that was never launched remain retained and hash-bound. These runtime errors
are recorded independently of completed quality PASS/FAIL verdicts.

Missing or uncompleted prerequisites produce `PENDING.json`, `ARCHIVE-APPEND.json`,
`INPUTS.json` and `FROZEN.json`, with no final qualification table. Only all19
corresponding receipts, their scope closures and the completed actual suite
allow `QUALIFICATION.json` and `QUALIFICATION.md`. Quality PASS/FAIL and evidence
VALID/INVALID remain separate. Source/consistency defects force overall INVALID
and exit2; exit0 means a pending or complete record was written. A completed
quality FAIL is retained as an actual outcome.

Native detail uses the unchanged collector's original34 observations, five20k
terminal records and paired-cloud shape receipts plus independent100k holdout.
Moving detail retains baseline500 and both postturn1000/1500 records with the
original30-degree schedule. Reused validators check the original saved-JSON
Boolean equations. No cloud is opened or rescored and no new quality gate is
introduced. Actual RA14 pytest counts and skip reasons are parsed from its
closed log, with no expected count substituted. Historical RA13's closed suite
log and skip reasons remain labeled RA13.

Fresh learned training remains RA13 (2000 updates per fixture): Toy's original
gate PASS with all9 postupdate metric/LR records matching RA11; all10 MNIST
checkpoint metric/LR records match corrected E22 with no formal MNIST numerical
gate. RA14 inherits those metrics through the closed helper-only restoration
bridge and executes the original two10-update CUDA continuation branches per
fixture. RA14 has zero fresh training updates. RA13 strict replay FAIL and RA14
r1 zero-update bridge failure remain preserved; closed RA14 replay is PASS for
both fixtures.

`ARCHIVE-APPEND.json` lists final small JSON/JSONL/log evidence and adapter
source/protocol files absent by path from the sealed510-file preparation
inventory, including all original failures, corrected lane artifacts, and the
unlaunched separate preparation. Any candidate artifact already in the original
inventory must retain its original bytes. The root separately archives the new
final output directory. PT/PTH/checkpoints, datasets, NPZ/NPY clouds, images and
GIFs remain local. The helper performs no archive copying, staging, commits,
repository writes, GPU operations, tensor loads, model calls, updates, sampling
or scorer jobs.

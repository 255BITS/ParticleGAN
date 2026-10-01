# Final evidence helper

`finalize_evidence.py` is a root-invoked stdlib helper. It reads source, JSON and
closed logs only. It writes exclusively to a new child directory of local
`release-prep`; it rejects existing output directories and any repository output
path. Earlier preparation files and frozen evidence remain unchanged.

```sh
python -B /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/release-prep/finalize_evidence.py --output /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/release-prep/final-attempt1
```

## Pending and complete records

Before completion, the helper writes `PENDING.json`, `ARCHIVE-APPEND.json`,
`INPUTS.json` and a local `FROZEN.json`. Every absent or uncompleted gate/full-suite
prerequisite remains pending. It produces no final qualification table. A pending
archive list is a snapshot that must be rebound after completion.

After the scoreboard and all 19 corresponding receipts close, and the actual
RA14 full-suite receipt/log close, it writes `QUALIFICATION.json` and
`QUALIFICATION.md`. Quality PASS/FAIL and evidence VALID/INVALID are separate;
consistency/source defects force overall qualification INVALID. Process exit0
means a pending or complete report was written; exit2 means identified evidence
defects. Quality FAIL by itself is preserved as a valid completed outcome.

## Scope

The helper verifies the exact 13 portability +3 moving +3 static native task set,
receipt/scoreboard/result agreement, before/after source identities, actual final
package source maps and both source digest conventions, shared config, closed
RA13 training/RA14 replay provenance, and actual pytest summary counts/skip
reasons. It does not substitute an expected test count.

Native detail comes from the unchanged collector's original 34 observations,
five20k terminal metric records and paired-cloud shape receipts plus independent
100k holdout. Moving detail retains baseline500 and both postturn1000/1500
records, the original30-degree schedule and original saved-JSON Boolean rule.
No scorer or cloud is opened. The existing collectors remain the artifact
validity authority; this helper checks their final reporting consistency.

RA13's original fresh2000-update training remains labeled RA13. RA14 inherits
that evidence through the closed restoration-only source bridge and executes
only the original continuation replays. MNIST has no invented numerical gate;
the exact ten-record corrected-E22 comparison is stated explicitly. Original
RA12 false-fire/quality failures, RA13 strict replay failures and RA14 r1
zero-update bridge preparation failure remain archived.

`ARCHIVE-APPEND.json` lists final JSON/JSONL/log files absent by path from the
sealed510-file inventory. It excludes PT/PTH checkpoints, NPZ/NPY clouds,
datasets, images and GIFs. The root separately archives this helper/protocol and
its final output files. No copy, staging, commit, GPU operation, tensor load,
model call, update, sample or rescoring is performed by this helper.

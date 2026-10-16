"""Write reports/forge/wallclock-reruns.json from the rerun attempts (real PASS/FAIL verdicts only).

Only cells whose earlier verdict was a wall-clock hit are listed; the same-runtime control
(Confidence LR, earlier PASS) is reported in the readout but never enters the overlay.
Usage: build_overlay.py STATE_ROOT REPO_ROOT
"""
import glob
import hashlib
import json
import os
import sys

ROOT, REPO = sys.argv[1], sys.argv[2]
# Retained receipts keep their raw terminal status in the evidence file only (see hygiene #428).
QUOTED = {"bcap-default-baseline-direction-v1": {
    "raw_terminal_status": "timeout", "seconds": 900.2391388670076,
    "evidence": "reports/forge/test-performance-evidence/pr399-retained-pr399-paired-tier1-12.json"}}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


records = []
for spec_path in sorted(glob.glob(f"{ROOT}/specs/*.json")):
    spec = json.load(open(spec_path))
    directory = f"{ROOT}/attempts/{os.path.basename(spec_path)[:-5]}"
    if spec["rerun_of"]["status"] not in {"TIMEOUT", "INCOMPLETE"} or not os.path.exists(f"{directory}/result.json"):
        continue
    result = json.load(open(f"{directory}/result.json"))
    record = {"key": spec["rerun_of"]["key"], "candidate_id": spec["candidate_id"], "task": spec["task"],
              "family": spec["family"], "source_digest": spec["source_digest"],
              "candidate_revision": spec["candidate_revision"], "execution_sha256": spec["execution_sha256"],
              "evaluation_sha256": spec["evaluation_sha256"], "timeout_seconds": spec["timeout_seconds"],
              "status": result["status"], "reason": result.get("reason", ""), "seconds": result["seconds"],
              "updates": result.get("updates"), "attempt": directory,
              "result_sha256": sha(f"{directory}/result.json"), "log_sha256": sha(f"{directory}/run.log"),
              "superseded_attempt": spec["rerun_of"]["attempt_id"], "superseded_status": spec["rerun_of"]["status"],
              "lease": "one job of this workflow per GPU (gpu-lease.sh)", "seed": 0}
    if spec["candidate_id"] in QUOTED:
        record["original_hit"] = QUOTED[spec["candidate_id"]]
    records.append(record)
value = {"schema": "forge_wallclock_reruns_v1", "qualification_input": False, "formulations_modified": False,
         "note": "Same-key solo reruns of wall-clock hits under the unchanged declared limit. A record replaces "
                 "the earlier hit only when its status is PASS or FAIL. Raw logs stay outside Git.",
         "readout": "reports/forge/dig2-20261016/wallclock-reruns.md", "records": records}
with open(f"{REPO}/reports/forge/wallclock-reruns.json", "w") as stream:
    json.dump(value, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps([(r["candidate_id"], r["task"], r["status"]) for r in records]))

"""Amendment A1: hold specs continuing from each PASSing smoke rerun's own checkpoint.

The checkpoint binding is made by the attempt-coverage runner's own ``prerequisite_bindings`` with
the rerun as a new-attempt parent. Usage: make_hold_specs.py STATE_ROOT HOLD_ROOT
"""
import hashlib
import json
import os
import shutil
import sys

from experiments.forge import attempt_coverage as runner

ROOT, HOLD = sys.argv[1], sys.argv[2]
FILL = "/ml2/hypergan/attempt-fill-20261015/state.json"
ORDER = ["bcap-default-baseline-direction-v1", "signed-confidence-lr-gaussian-v1", "positive-evidence-lr-gaussian-v1"]
DROP = {"status", "attempt_id", "directory", "pid", "process_identity", "gpu", "slot", "seconds", "updates",
        "reason", "grade", "result_sha256", "log_sha256", "runtime", "announced", "evidence", "origin",
        "attempted", "training_gif_sha256", "media_error", "media_note", "original_attempt", "block_kind"}
state = json.load(open(FILL))
for d in ("specs", "requests", "attempts"):
    os.makedirs(f"{HOLD}/{d}", exist_ok=True)
for i, candidate in enumerate(ORDER):
    smoke_dir = next(d for d in sorted(os.listdir(f"{ROOT}/attempts")) if d.endswith(f"{candidate}--five_word_joint_smoke"))
    smoke_dir = f"{ROOT}/attempts/{smoke_dir}"
    if not os.path.exists(f"{smoke_dir}/result.json"):
        print("skip", candidate, "smoke rerun not finished"); continue
    smoke_result = json.load(open(f"{smoke_dir}/result.json"))
    if smoke_result["status"] != "PASS":
        print("skip", candidate, smoke_result["status"]); continue
    smoke_spec = json.load(open(f"{smoke_dir}/request.json"))
    job = dict(next(j for j in state["jobs"].values() if j["candidate_id"] == candidate and j["task"] == "five_word_joint_hold"))
    target = f"{HOLD}/requests/{os.path.basename(job['request_path'])}"
    shutil.copyfile(job["request_path"], target)
    digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
    assert job.get("request_sha256") in (None, digest)
    spec = {k: v for k, v in job.items() if k not in DROP}
    spec.update(request_path=target, request_sha256=digest, status="PENDING",
                rerun_of={"status": job["status"], "attempt_id": job.get("attempt_id") or job.get("original_attempt"),
                          "seconds": job.get("seconds"), "origin": job.get("origin"), "key": job["key"],
                          "reason": job.get("reason")})
    parent = {"family": job["family"], "task": "five_word_joint_smoke", "status": "PASS", "origin": "new_attempt",
              "directory": smoke_dir, "attempt_id": os.path.basename(smoke_dir),
              "candidate_revision": smoke_spec["candidate_revision"]}
    synthetic = {"jobs": {"parent": parent, "hold": spec}}
    spec["prerequisites"] = runner.prerequisite_bindings(spec, synthetic)
    spec["dependency_blockers"] = runner.dependency_blockers(spec, synthetic)
    assert "five_word_joint_smoke" in spec["prerequisites"] and not spec["dependency_blockers"], spec["dependency_blockers"]
    json.dump(spec, open(f"{HOLD}/specs/{i:02d}-{candidate}--five_word_joint_hold.json", "w"), indent=1, sort_keys=True)
    print(i, candidate, job["status"], job.get("reason"), spec["timeout_seconds"])

"""Build rerun specs for wall-clock-hit cells from the attempt-fill state (read-only on the fill).

Each spec is the fill's own job record (same frozen source snapshot, recipe revision, task card,
timeout and request bytes); only the attempt bookkeeping is reset and the request file is copied
into this lead's state root (its SHA-256 is re-checked by the worker).
"""
import hashlib
import json
import os
import shutil
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/ml2/hypergan/dig2-20261016/wallclock-reruns"
FILL = "/ml2/hypergan/attempt-fill-20261015/state.json"
ORDER = [
    ("bcap-default-baseline-direction-v1", "five_word_joint_smoke"),
    ("signed-confidence-lr-gaussian-v1", "five_word_joint_smoke"),
    ("positive-evidence-lr-gaussian-v1", "five_word_joint_smoke"),
    ("generator-confidence-lr-bcap-direction-v1", "five_word_joint_smoke"),
    ("kinetic_transport_local_v2", "five_word_joint_smoke"),
    ("kinetic-local-dualnorm-g-history-v1", "five_word_joint_smoke"),
    ("confidence-lr-finite-cap-gaussian-v1", "five_word_joint_smoke"),
    ("kinetic-local-dualnorm-g-history-reference8-u-v1", "ring16_acquisition"),
]
DROP = {"status", "attempt_id", "directory", "pid", "process_identity", "gpu", "slot", "seconds", "updates",
        "reason", "grade", "result_sha256", "log_sha256", "runtime", "announced", "evidence", "origin",
        "attempted", "training_gif_sha256", "media_error", "media_note", "original_attempt"}
state = json.load(open(FILL))
for d in ("specs", "requests", "attempts"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
for i, (candidate, task) in enumerate(ORDER):
    jobs = [j for j in state["jobs"].values() if j["candidate_id"] == candidate and j["task"] == task]
    assert len(jobs) == 1, (candidate, task, len(jobs))
    job = dict(jobs[0])
    source = job["request_path"]
    target = f"{OUT}/requests/{os.path.basename(source)}"
    shutil.copyfile(source, target)
    digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
    assert job.get("request_sha256") in (None, digest), (candidate, "request bytes differ")
    spec = {k: v for k, v in job.items() if k not in DROP}
    spec.update(request_path=target, request_sha256=digest, dependency_blockers=[], prerequisites={},
                status="PENDING",
                rerun_of={"status": job["status"], "attempt_id": job.get("attempt_id") or job.get("original_attempt"),
                          "seconds": job.get("seconds"), "origin": job.get("origin"), "key": job["key"]})
    name = f"{OUT}/specs/{i:02d}-{candidate}--{task}.json"
    json.dump(spec, open(name, "w"), indent=1, sort_keys=True)
    print(i, candidate, task, job["status"], job.get("timeout_seconds"), job.get("dependencies"),
          job.get("diagnostic_continuation"), job["source_root"][-20:])

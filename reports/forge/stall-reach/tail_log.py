"""Append one line per terminal Forge job (both arms) to ARCHIVE/run.log; exits when the queue is idle."""
import json
from pathlib import Path
import sys
import time

archive = Path(sys.argv[1])
state_path = archive / "queue/queue/state.json"
progress = json.loads((archive / "progress.json").read_text())
roles = {request: role for role, request in progress["requests"].items()}
seen = set()
with open(archive / "run.log", "a", buffering=1) as log:
    while True:
        state = json.loads(state_path.read_text())
        for key, job in sorted(state["jobs"].items()):
            if key in seen or job["status"] != "terminal":
                continue
            seen.add(key)
            result = job.get("result") or {}
            definition = job.get("definition", {})
            owners = ",".join(roles.get(s, s[:8]) for s in job.get("subscribers", []))
            for row in result.get("task_results") or [{"task_id": definition.get("task_id")}]:
                cost = row.get("cost") or result.get("cost") or {}
                log.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), arm=owners, task=row.get("task_id"),
                                          raw=row.get("raw_status") or row.get("status") or result.get("status"),
                                          reason=(row.get("reason") or result.get("reason") or "")[:120],
                                          wall=round(cost.get("wall_seconds") or 0, 1))) + "\n")
        busy = any(j["status"] == "running" for j in state["jobs"].values()) or any(
            s["status"] in ("queued", "running") for s in state["submissions"].values())
        if not busy:
            log.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), event="queue_idle")) + "\n")
            break
        time.sleep(15)

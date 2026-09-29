"""Task child entrypoint: one dispatch path for CLI run and queue workers."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import traceback

from .contracts import atomic_json, canonical, read_json, utc_now


def execute(path: Path) -> int:
    resolved = read_json(path)
    request, job, worker = resolved["request"], resolved["job"], resolved["worker"]
    output = path.parent
    started = time.monotonic()
    common = {"timestamp": utc_now(), "candidate": request["candidate"]["id"],
              "revision": request["candidate_revision"], "task": job["task_id"], "attempt": worker["attempt"],
              "gpu": worker["device"], "campaign": request["campaign_id"]}
    print(canonical({**common, "event": "started"}), flush=True)
    exit_code = 0
    try:
        import torch
        torch.set_num_threads(job["resources"].get("cpu_threads", 1))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from .api import CapabilityError
        blockers = request["tasks"][job["task_id"]].get("preflight_blockers", [])
        if blockers:
            raise CapabilityError(blockers)
        from .adapters import run_task
        execution_job = {**job, "prerequisites": resolved.get("prerequisites", {})}
        result = run_task(request, execution_job, output, "cpu" if worker["device"] == "cpu" else "cuda:0")
        if not isinstance(result, dict):
            raise TypeError("adapter must return a raw evidence receipt")
    except Exception as exc:
        from .api import CapabilityError
        if isinstance(exc, CapabilityError):
            result = {"applicability": {"status": "unsupported", "reason": str(exc)},
                      "error": {"type": type(exc).__name__, "message": str(exc)}}
        else:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        traceback.print_exc()
        exit_code = 1
    result.setdefault("execution_path", request["candidate"].get("execution_path", "public_trainer"))
    result.setdefault("api_version", request["candidate"].get("api_version", "forge-api-v1"))
    result.setdefault("api_changes", request["candidate"].get("api_changes", []))
    result.setdefault("claim_contract", request["candidate"]["claim_contract"])
    result.setdefault("cost", {})["adapter_seconds"] = time.monotonic() - started
    atomic_json(output / "raw-result.json", result)
    print(canonical({**common, "timestamp": utc_now(), "event": "recorded", "exit_code": exit_code,
                     "cost": result["cost"], "result": str(output / "raw-result.json")}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(execute(Path(sys.argv[1]).resolve()))

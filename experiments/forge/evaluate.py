"""Independent evaluator child, imported entirely from the frozen source tree."""
from pathlib import Path
import sys
import time

from .contracts import atomic_json, read_json, stable_hash
from .views import grade_result
from .telemetry import MemoryProbe


def evaluate(path: Path):
    started = time.monotonic()
    memory = MemoryProbe()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resolved = read_json(path)
    request, job = resolved["request"], resolved["job"]
    raw = read_json(path.parent / "raw-result.json")
    grades = {}
    for task_id in job.get("task_ids", [job["task_id"]]):
        member = raw.get("task_results", {}).get(task_id, raw)
        from .atlas_existing_mog import is_candidate as existing_mog_candidate
        evidence = member.get("evidence", member.get("metrics")) if isinstance(member, dict) else None
        if (existing_mog_candidate(request["candidate"])
                and isinstance(member, dict) and not member.get("error")
                and member.get("gate_status", member.get("status")) not in {"BLOCKED", "ERROR", "error", "timeout", "cancelled", "INVALID", "INCOMPLETE"}
                and (not isinstance(evidence, dict) or "existing_mog717" not in evidence)):
            grades[task_id] = {"status": "INVALID", "reason": "Track B actual owner/control receipt is missing"}
        else:
            grades[task_id] = grade_result(request["tasks"][task_id], member)
    result = {"schema_version": 1, "grades": grades, "raw_hash": stable_hash(raw),
              "source_digest": request["source"]["digest"],
              "telemetry": {"independent_grading_seconds": time.monotonic() - started,
                            "memory": memory.snapshot(), "scope": "Independent CPU evaluator process, including evaluator artifact reads."}}
    atomic_json(path.parent / "graded-result.json", result)
    return result


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]).resolve())

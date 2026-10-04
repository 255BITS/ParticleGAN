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
        task = request["tasks"][task_id]
        grade = grade_result(task, member)
        if task.get("task_cohort") == "word_joint_policy_min11_rates_v1":
            from .word_joint_rate_policy_contracts import validate_request, validate_result_binding
            try:
                validate_request(request, task)
                if grade["status"] in {"PASS", "FAIL"}:
                    validate_result_binding(request, task, member)
            except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
                grade = {"status": "INVALID", "reason": str(error)}
        grades[task_id] = grade
    result = {"schema_version": 1, "grades": grades, "raw_hash": stable_hash(raw),
              "source_digest": request["source"]["digest"],
              "telemetry": {"independent_grading_seconds": time.monotonic() - started,
                            "memory": memory.snapshot(), "scope": "Independent CPU evaluator process, including evaluator artifact reads."}}
    atomic_json(path.parent / "graded-result.json", result)
    return result


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]).resolve())

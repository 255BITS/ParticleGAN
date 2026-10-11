"""Interim grades of finished jobs (no qualification; publish uses certified receipts)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.views import grade_result  # noqa: E402

archive = Path(sys.argv[1])
state = json.loads((archive / "queue/queue/state.json").read_text())
roles = {r: k for k, r in json.loads((archive / "progress.json").read_text())["requests"].items()}
for request_id, entry in state["submissions"].items():
    for definition in entry["request"]["jobs"]:
        job = state["jobs"].get(definition["compatibility_key"], {})
        for row in (job.get("result") or {}).get("task_results", []):
            grade = grade_result(entry["request"]["tasks"][row["task_id"]], row)
            guards = (row.get("evidence") or {}).get("guards", {})
            read = guards.get("generator_critic_read")
            read = read[0] if isinstance(read, list) and read else read
            print(roles.get(request_id), row["task_id"], grade.get("gate_status", grade.get("status")),
                  round((row.get("cost") or {}).get("wall_seconds") or 0, 1),
                  {k: read.get(k) for k in ("observes", "widened", "stalled", "max_width")} if read else "")

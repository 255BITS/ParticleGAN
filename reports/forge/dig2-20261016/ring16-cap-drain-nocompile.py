"""Drain one campaign exactly as `python -m experiments.forge drain`, but without the
post-completion compile_memory hook (a reporting step that rewrites reports/forge boards and
took >20 min per request on this checkout). Execution, grading and receipts are unchanged."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from experiments.forge.queue import Queue, drain  # noqa: E402

root = Path(os.getcwd())
queue_root = Path(sys.argv[1])
gpu, campaign = sys.argv[2], sys.argv[3]
queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=None)
drain(queue, [gpu], workers_per_gpu=1, campaign=campaign)
print({"status": "drained", "logs": str(queue_root / "events.jsonl")})

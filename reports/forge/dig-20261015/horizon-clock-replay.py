"""Diagnostic replay of one frozen Forge ring16 request with the progress/LR trajectory logged.

Usage: python horizon-clock-replay.py <attempt request.json> <out_dir> <device>
Runs the same adapter (run_task) on the frozen request; it is NOT evidence, only a trace of
p, the critic order-accuracy EMA, the applied G LR multiplier and noise levels every 50 updates.
"""
import json
import sys
from pathlib import Path

import torch

from experiments.forge.adapters import run_task
from particlegan.policy import UpdatePolicy

request = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
trace = open(out / "trace.jsonl", "w")
original = UpdatePolicy.begin_step


def begin_step(self, real, *args, **kwargs):
    noise = original(self, real, *args, **kwargs)
    step = self.completed_steps
    if step % 50 == 0:
        p = None if getattr(self, "progress", None) is None else self.progress.state_dict()
        g_lr = self.optimizers[0].param_groups[0]["lr"] / self.initial_lrs[0][0]
        trace.write(json.dumps({"step": step, "progress": p, "g_lr_scale": g_lr,
                                "input_sigma": float(noise.input_sigma),
                                "output_sigma": float(noise.output_sigma)}) + "\n")
        trace.flush()
    return noise


UpdatePolicy.begin_step = begin_step
job, request = request["job"], request["request"]
assert job["task_id"] == "ring16_acquisition", job["task_id"]
result = run_task(request, job, out / "run", sys.argv[3])
live = result["evidence"]["live"]
print(json.dumps({k: live.get(k) for k in ("modes", "mass_tv", "hq", "component_covariance_error",
                                            "component_min_eigen_ratio")}))

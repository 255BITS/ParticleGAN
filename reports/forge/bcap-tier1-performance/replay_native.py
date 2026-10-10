#!/usr/bin/env python3
"""Actual pre-change source versus current: aligned complete word state proof."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from verify_profile_parity import differences


BASE = "79f7ddb512f0bc1e1457257e85c97adbe6d11bd1"
PROBE = r'''
from copy import deepcopy
import json, sys
from pathlib import Path
import torch
from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge.api import task_formulation_context
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
# Align otherwise unconsumed ambient process state. Public named initializer
# and all consumed streams still use the unchanged protocol seed zero.
torch.manual_seed(0)
root=Path.cwd()
candidate=json.loads((root/'configs/forge/ideas/bcap-develop-integration-combined-v1.json').read_text())
task=json.loads((root/'configs/forge/tasks/five_word_joint_smoke.json').read_text())
context=task_formulation_context(candidate,task,{'seed':0},device='cuda:0',root=root)
fixture=WordFixture(device='cuda:0',seed=0,recipe_name=None,max_steps=20001,components=context)
initial=deepcopy({'fixture':fixture.state_dict(),'streams':context.streams.state_dict()})
outputs=[fixture.step() for _ in range(8)]
observation=fixture.observe()
torch.save({'initial':initial,'outputs':outputs,'observation':observation,
 'final':{'fixture':fixture.state_dict(),'streams':context.streams.state_dict()}},sys.argv[1])
'''


def main():
    import os
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    old_root = args.output / "original-source"
    old_root.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", BASE, "particlegan", "experiments", "benchmarks",
                                       "configs", "lib"], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(old_root, filter="data")
    packets = {}
    for name, source in (("original", old_root), ("candidate", root)):
        output = args.output / (name + ".pt")
        environment = dict(os.environ, PYTHONPATH=str(source), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run([sys.executable, "-c", PROBE, str(output)], cwd=source, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        packets[name] = torch.load(output, map_location="cpu", weights_only=False)
    delta = differences(packets["original"], packets["candidate"])
    receipt = dict(scope="software_diagnostic_only", qualification_input=False,
        original_source_commit=BASE, updates=8, protocol_seed=0,
        original_execution_cap=20001, original_schedule_horizon=20000,
        whole_state_outputs_observation_exact=not delta, differences=delta,
        baseline_archive_sha256=hashlib.sha256(archive).hexdigest(),
        probe_sha256=hashlib.sha256(PROBE.encode()).hexdigest(),
        packets={name: hashlib.sha256((args.output / (name + ".pt")).read_bytes()).hexdigest()
                 for name in packets})
    (args.output / "replay.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    assert not delta


if __name__ == "__main__":
    main()

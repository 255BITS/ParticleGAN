"""Observe the complete declared transfer suite, using its frozen references.

Published solvable architectures are explicit for the ten canonical vector
and image cases. Other stresses retain their original cosine reference. The
three former reserved cases are declared seen audit data in this run.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.compare_defaults import (RECIPES, candidate, effective_spec,
                                                         optimizer_defaults, plan, skip_constructor,
                                                         smooth_constructor)
from benchmarks.transfer_suite.protocol import test_verdict
from .capture import Capture, write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reference",choices=["frozen","public_v2"],default="frozen")
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[2]
    manifest=suite.manifest()
    published={j["spec"]["name"]:j for j in plan()}
    write(args.output / "manifest.json", dict(base_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                tasks=manifest["tasks"], reference=args.reference, interpretation="previously reserved cases now seen by this audit; no new holdout claim",
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((root/"particlegan").glob("*.py"))}))
    summaries=[]
    for original in manifest["tasks"]:
        name=original["name"]
        output=args.output/name
        output.mkdir(parents=True,exist_ok=False)
        spec=deepcopy(published[name]["spec"] if name in published else original)
        print(json.dumps(dict(event="BASE_START",name=name,steps=spec["steps"])),flush=True)
        started=time.perf_counter()
        if args.reference=="frozen":
            if spec["runner"]=="legacy":
                result=baseline.run_toy(name,baseline.Candidate("locked_shared"))
                write(output/"result.json",result)
                summary=dict(name=name,kind="behavior",spec=spec,verdict=test_verdict(spec,result),
                             live=result.get("live"),error=result.get("error"),frames=len(result["observations"]),
                             sampling="original live numerical behavior; curves only",artifact=str(output),
                             result_sha256=hashlib.sha256((output/"result.json").read_bytes()).hexdigest())
            else:
                with ExitStack() as stack:
                    card=spec.get("research_discriminator")
                    if card:
                        create=skip_constructor(card) if card.get("skip")=="raw_linear" else smooth_constructor(card)
                        stack.enter_context(patch.object(vector_tasks,"SimpleMLPDiscriminator",create))
                    cap=Capture(output)
                    with cap.installed():
                        suite.run_episode(spec,vector_tasks.fixed_policy(),fixed=True,allow_reserved=True)
                    summary=cap.records[0]
        elif name in published:
            recipe=RECIPES["proposed"]
            spec=effective_spec(spec,recipe)
            applied=[]
            with optimizer_defaults(recipe,applied),ExitStack() as stack:
                if spec["runner"]=="legacy":
                    control=evaluate.FixedControl(vector_tasks.fixed_policy(),spec["steps"])
                    with bridge.control_host_schedules(control):
                        result=baseline.run_toy(name,candidate(recipe))
                    result["seconds"]=time.perf_counter()-started
                    write(output/"result.json",result)
                    summary=dict(name=name,kind="behavior",spec=spec,verdict=test_verdict(spec,result),
                                 live=result.get("live"),error=result.get("error"),frames=len(result["observations"]),
                                 sampling="original live numerical behavior; curves only",artifact=str(output),
                                 result_sha256=hashlib.sha256((output/"result.json").read_bytes()).hexdigest())
                else:
                    card=spec.get("research_discriminator")
                    if card:
                        create=skip_constructor(card) if card.get("skip")=="raw_linear" else smooth_constructor(card)
                        stack.enter_context(patch.object(vector_tasks,"SimpleMLPDiscriminator",create))
                    cap=Capture(output)
                    with cap.installed():
                        suite.run_episode(spec,vector_tasks.fixed_policy(),fixed=True,allow_reserved=True)
                    summary=cap.records[0]
        else:
            cap=Capture(output)
            with cap.installed():
                suite.run_episode(spec,vector_tasks.fixed_policy(),fixed=True,allow_reserved=True)
            summary=cap.records[0]
        summary.update(architecture=published[name]["architecture"] if name in published else "declared stress architecture",
                       seconds=time.perf_counter()-started)
        write(output/"summary.json",summary)
        summaries.append(summary)
        write(args.output/"index.json",summaries)
        print(json.dumps(dict(event="BASE_DONE",name=name,status=summary["verdict"]["status"],seconds=summary["seconds"])),flush=True)


if __name__=="__main__":
    main()

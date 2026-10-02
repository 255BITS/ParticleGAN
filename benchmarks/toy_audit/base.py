"""Observe the complete declared transfer suite, using its frozen references.

Published solvable architectures are explicit for the ten canonical vector
and image cases. Other stresses retain their original cosine reference. The
three former reserved cases are declared seen audit data in this run.
Identical frozen bars4 evidence is shared explicitly; --separate-execution
retains the historical one-execution-per-declaration path.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import torch
import numpy as np

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite import suite, vector_tasks, image_tasks
from benchmarks.transfer_suite.compare_defaults import (RECIPES, candidate, effective_spec,
                                                         optimizer_defaults, plan, skip_constructor,
                                                         smooth_constructor)
from benchmarks.transfer_suite.protocol import test_verdict
from .capture import Capture, write


ALIAS_VERSION = "frozen-image-bars4-alias-v1"
ALIAS_PROOF = "reports/toy_audit/frozen-image-alias.json"
ALIAS_PROOF_SHA256 = "2eb6ca8010e6db70d92050026ace5fa6f7c1aa853ccb47592eab90ffab55faa2"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_alias_proof(root):
    """Unknown/missing provenance always falls back to separate execution."""
    try:
        path = root / ALIAS_PROOF
        if _sha256(path) != ALIAS_PROOF_SHA256:
            return None
        proof = json.loads(path.read_text())
        return proof if proof["version"] == ALIAS_VERSION else None
    except (OSError, ValueError, KeyError):
        return None


def runtime_identity():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torch_git_revision=torch.version.git_version, machine=platform.machine(),
                device=str(image_tasks.host_device()), default_device=str(torch.get_default_device()),
                default_dtype=str(torch.get_default_dtype()), torch_threads=torch.get_num_threads())


def source_runtime_matches(root, proof):
    """Bind hidden defaults to the exact source/runtime whose identity is known."""
    if proof is None:
        return False
    try:
        sources = proof["source_sha256"]
        package_files = {str(p.relative_to(root)) for p in (root / "particlegan").glob("*.py")}
        if package_files != {name for name in sources if name.startswith("particlegan/")}:
            return False
        if any(_sha256(root / name) != expected for name, expected in sources.items()):
            return False
        return runtime_identity() == proof["runtime"]
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def frozen_image_identity(spec, proof):
    """Compare ordered execution laws, never just a common pattern name."""
    metadata = set(proof["metadata_fields"])
    expected = proof["effective_spec"]
    if set(spec) - metadata - set(expected):
        raise ValueError("unknown mathematical fields")
    resolved = proof["implicit_defaults"] | {k:v for k,v in spec.items() if k not in metadata}
    # JSON equality distinguishes booleans/integers and retains every gate/key.
    if json.dumps(resolved, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("effective recipe changed")
    target = image_tasks.templates(spec).detach().cpu().numpy()
    target_identity = dict(shape=list(target.shape), dtype=target.dtype.str,
                           ordered_sha256=hashlib.sha256(target.tobytes(order="C")).hexdigest())
    if target_identity != proof["target"]:
        raise ValueError("ordered target law changed")
    if image_tasks.evaluation_steps(spec) != proof["evaluation_steps"]:
        raise ValueError("evaluation cadence changed")
    return dict(effective_spec=resolved, target=target_identity,
                execution_contract=proof["execution_contract"])


def shared_frozen_summary(spec, summaries, *, root, reference, separate_execution=False):
    """Reuse one complete observed capture without creating a second capture."""
    if reference != "frozen" or separate_execution or spec["name"] != "img_residual_bars4":
        return None
    proof = load_alias_proof(root)
    if not source_runtime_matches(root, proof):
        return None
    canonical = next((s for s in summaries if s["name"] == proof["canonical_name"]), None)
    if (canonical is None or canonical.get("execution_status") == "ALIAS" or canonical.get("error")
            or canonical.get("spec", {}).get("name") != proof["canonical_name"]):
        return None
    try:
        identity = frozen_image_identity(spec, proof)
        if identity != frozen_image_identity(canonical["spec"], proof):
            return None
        source = Path(canonical["artifact"])
        result_path, cloud_path = source / "result.json", source / "observations.npz"
        if (_sha256(result_path) != canonical["result_sha256"]
                or _sha256(cloud_path) != canonical["capture_sha256"]):
            return None
        result = json.loads(result_path.read_text())
        if (result.get("error") or result["spec"] != canonical["spec"]
                or result["protocol"] != proof["observed_protocol"]
                or result["policy"] != vector_tasks.fixed_policy()
                or result["ablation"] != "none" or result["fixed"] is not True):
            return None
        verdict = test_verdict(spec, result)
        if not verdict["convergence"]["complete"]:
            return None
        with np.load(cloud_path) as arrays:
            shape = (len(proof["evaluation_steps"]), spec["particles"], 1, 8, 8)
            if (arrays["steps"].tolist() != proof["evaluation_steps"]
                    or arrays["live"].shape != shape or arrays["ema"].shape != shape
                    or not np.isfinite(arrays["live"]).all() or not np.isfinite(arrays["ema"]).all()
                    or arrays["templates"].dtype.str != proof["target"]["dtype"]
                    or hashlib.sha256(arrays["templates"].tobytes(order="C")).hexdigest()
                        != proof["target"]["ordered_sha256"]):
                return None
        summary = deepcopy(canonical)
        summary.update(name=spec["name"], spec=deepcopy(spec), verdict=verdict,
                       execution_status="ALIAS", independently_trained=False,
                       independent_qualification_evidence=False, seconds=0.,
                       alias=dict(version=ALIAS_VERSION, canonical_name=canonical["name"],
                                  reason="Identical frozen effective recipe; shared observed evidence",
                                  observed_spec=deepcopy(canonical["spec"]), source_artifact=str(source),
                                  source_result=str(result_path), source_capture=str(cloud_path),
                                  source_result_sha256=canonical["result_sha256"],
                                  source_capture_sha256=canonical["capture_sha256"],
                                  source_training_seconds=canonical["seconds"],
                                  proof_sha256=ALIAS_PROOF_SHA256, execution_identity=identity))
        return summary
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reference",choices=["frozen","public_v2"],default="frozen")
    ap.add_argument("--separate-execution",action="store_true",
                    help="Reproduce historical separate executions, including identical frozen bars4 jobs")
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[2]
    manifest=suite.manifest()
    published={j["spec"]["name"]:j for j in plan()}
    write(args.output / "manifest.json", dict(base_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                tasks=manifest["tasks"], reference=args.reference, interpretation="previously reserved cases now seen by this audit; no new holdout claim",
                execution_reuse=dict(version=ALIAS_VERSION, separate_execution=args.separate_execution,
                                     scope="frozen img_residual_bars4 may share img_bars4 evidence; declarations remain complete"),
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
        summary=shared_frozen_summary(spec,summaries,root=root,reference=args.reference,
                                      separate_execution=args.separate_execution)
        if summary is not None:
            print(json.dumps(dict(event="BASE_ALIAS",name=name,canonical=summary["alias"]["canonical_name"],
                                  source_artifact=summary["artifact"],independently_trained=False)),flush=True)
        elif args.reference=="frozen":
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
        summary.update(architecture=published[name]["architecture"] if name in published else "declared stress architecture")
        if summary.get("execution_status") == "ALIAS":
            summary["reuse_seconds"]=time.perf_counter()-started
        else:
            summary.update(execution_status="EXECUTED",seconds=time.perf_counter()-started)
        write(output/"summary.json",summary)
        summaries.append(summary)
        write(args.output/"index.json",summaries)
        print(json.dumps(dict(event="BASE_DONE",name=name,status=summary["verdict"]["status"],seconds=summary["seconds"])),flush=True)


if __name__=="__main__":
    main()

"""Bounded original-source coverage; no recipe repair or substitute loop."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

import torch

from .source_conditional_routes_observer import TransitionCapture, owner_state

REVISION = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
ENTRIES = {
    "source-family-04": ("trajectory", "discrete", "configs/trajectory/default.yaml"),
    "source-family-05": ("trajectory", "continuous", "configs/trajectory/diversity/confirm_10k/mlp_continuous.yaml"),
    "source-family-06": ("transition", "discrete", "configs/transition/default.yaml"),
    "source-family-07": ("transition", "continuous", "configs/transition/default.yaml"),
}


class PrefixComplete(BaseException):
    pass


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hashes(source):
    paths = []
    for directory in ("particlegan", "lib", "experiments"):
        paths.extend((source / directory).rglob("*.py"))
    paths.extend((source / "configs/trajectory").rglob("*.yaml"))
    paths.extend((source / "configs/transition").rglob("*.yaml"))
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load(source, family):
    sys.path.insert(0, str(source))
    path = source / f"experiments/train_{family}.py"
    spec = importlib.util.spec_from_file_location("original_route_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(module, cfg, directory, *, observe, prefix):
    directory.mkdir(exist_ok=False)
    run_cfg = dict(cfg, out_dir=str(directory / "run"))
    if "live_log" in run_cfg:
        run_cfg["live_log"] = str(directory / "live.log")
    capture = TransitionCapture(module, directory) if observe else None
    target = module.train.__code__
    tree = ast.parse(Path(module.__file__).read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    lines = {node.lineno for node in ast.walk(fn) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "step"}
    boundaries, completed = {}, 0

    def trace(frame, event, arg):
        return local_trace if event == "call" and frame.f_code == target else None

    def local_trace(frame, event, arg):
        nonlocal completed
        if event == "line" and frame.f_lineno in lines:
            local = frame.f_locals
            step = local.get("step", 0)
            completed = step
            if prefix:
                if step not in boundaries:
                    if capture:
                        capture.observe(step, local)
                    boundaries[step] = owner_state(local)
                if step == 3:
                    raise PrefixComplete()
            elif capture and (step in (0, 1, 10, 25, 50, 100) or step % 250 == 0 or step == cfg["steps"]):
                if not capture.rows or capture.rows[-1]["step"] != step:
                    capture.observe(step, local)
        return local_trace

    started = time.perf_counter()
    result, error, error_type, status = None, None, None, None
    sys.settrace(trace)
    try:
        result = module.train(run_cfg)
        status = "COMPLETE"
    except PrefixComplete:
        status = "PREFIX_COMPLETE"
    except TimeoutError:
        status, error, error_type = "INCOMPLETE", traceback.format_exc(), "TimeoutError"
    except Exception as exc:
        status, error, error_type = "BLOCKED", traceback.format_exc(), type(exc).__name__
    finally:
        sys.settrace(None)
        if capture:
            capture.save()
    receipt = dict(status=status, completed_updates=completed, boundaries=boundaries,
                   observer_states=len(capture.rows) if capture else 0,
                   observer_seconds=capture.seconds if capture else 0.,
                   original_budget=cfg["steps"], external_prefix_stop=3 if prefix else None,
                   elapsed_seconds=time.perf_counter() - started, error=error, error_type=error_type)
    if result is not None:
        write(directory / "original-summary.json", result)
        receipt["original_summary_sha256"] = hashlib.sha256((directory / "original-summary.json").read_bytes()).hexdigest()
    if error:
        print(error, flush=True)
    for name in ("observations.jsonl", "observations.npz"):
        if (directory / name).exists():
            receipt[name + "_sha256"] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    write(directory / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", choices=ENTRIES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second entry wall cap exhausted")))
    signal.alarm(120)
    torch.set_num_threads(1)
    family, geometry, config = ENTRIES[args.entry]
    hashes = file_hashes(args.source)
    receipt = dict(format="route_source_coverage_v1", catalog_id=args.entry, family=family,
                   geometry=geometry, source_revision=REVISION, source_root=str(args.source),
                   source_sha256=hashes, input_config_path=config, wall_cap_seconds=120,
                   runtime=dict(python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1),
                   qualification_credit="none", seed_tuning=False, full_campaign_count=0,
                   gate="Original trainer reports endpoint diagnostics; it declares no aggregate convergence acceptance gate.",
                   original_scientific_status="NOT EVALUATED", added_gate_status="NOT EVALUATED")
    own_root = Path(__file__).resolve().parents[2]
    receipt["observer_sources"] = {p: hashlib.sha256((own_root / p).read_bytes()).hexdigest() for p in
                                  ("benchmarks/toy_audit/source_conditional_routes_run.py",
                                   "benchmarks/toy_audit/source_conditional_routes_observer.py")}
    try:
        module = load(args.source, family)
        cfg = {**module.DEFAULTS, **module.read_config(str(args.source / config))}
        receipt["original_cli_config"] = cfg.copy()
        cfg["geometry_mode"] = geometry
        if "device" in cfg:
            cfg["device"] = "cpu"
        receipt["execution_config"] = cfg.copy()
        receipt["problem_selection"] = "Original checked-in config" if family == "trajectory" or geometry == "discrete" else "Source-supported continuous geometry, same transition defaults and budget"
        receipt["resolved_recipe"] = module.training_recipe(cfg).to_dict()
        write(args.out / "frozen-inputs.json", receipt)
        print(json.dumps(dict(event="route_source_frozen", entry=args.entry, budget=cfg["steps"], seed=cfg["seed"],
                              geometry=geometry, phase="prefix_baseline")), flush=True)
        baseline = execute(module, cfg, args.out / "prefix-baseline", observe=False, prefix=True)
        receipt["prefix_baseline"] = baseline
        if baseline["status"] != "PREFIX_COMPLETE":
            receipt.update(fresh_execution_status=baseline["status"], original_scientific_status=baseline["status"],
                           blocked_phase="original-source baseline prerequisite", error=baseline["error"])
        else:
            print(json.dumps(dict(event="route_source_phase", entry=args.entry, phase="prefix_observed")), flush=True)
            observed = execute(module, cfg, args.out / "prefix-observed", observe=True, prefix=True)
            receipt["prefix_observed"] = observed
            same = observed["status"] == "PREFIX_COMPLETE" and baseline["boundaries"] == observed["boundaries"]
            receipt["prefix_parity"] = dict(passed=same, boundaries=[0, 1, 2, 3],
                                           exact_matches=4 if same else 0, software_updates=6)
            if not same:
                receipt.update(fresh_execution_status="BLOCKED", original_scientific_status="BLOCKED",
                               blocked_phase="observer parity prerequisite", error=observed["error"] or "Prefix hashes differ")
            else:
                print(json.dumps(dict(event="route_source_phase", entry=args.entry, phase="full_original_budget")), flush=True)
                paid = execute(module, cfg, args.out / "training", observe=True, prefix=False)
                receipt["training"] = paid
                receipt["full_campaign_count"] = 1
                receipt["fresh_execution_status"] = paid["status"]
                receipt["original_scientific_status"] = "NO DECLARED ACCEPTANCE GATE" if paid["status"] == "COMPLETE" else paid["status"]
                receipt["error"] = paid["error"]
    except TimeoutError:
        receipt.update(fresh_execution_status="INCOMPLETE", original_scientific_status="INCOMPLETE", error=traceback.format_exc())
    except Exception:
        receipt.update(fresh_execution_status="BLOCKED", original_scientific_status="BLOCKED", error=traceback.format_exc())
    finally:
        signal.alarm(0)
    receipt["source_unchanged"] = hashes == file_hashes(args.source)
    receipt["wall_seconds"] = time.perf_counter() - started
    write(args.out / "receipt.json", receipt)
    print(json.dumps(dict(event="route_source_finished", entry=args.entry, status=receipt["fresh_execution_status"],
                          full_campaigns=receipt["full_campaign_count"], seconds=receipt["wall_seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

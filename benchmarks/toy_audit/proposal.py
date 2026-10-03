"""Observe pinned new-problem proposals without editing develop's configs.

The proposal directory is an overlay, never a replacement for particlegan.
An archived recipe, when requested, is a separately labelled historical arm.
All raw outputs belong outside Git.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
import traceback

import numpy as np
import torch

from .capture import write


def finite_json(value):
    """Keep diagnostic infinities explicit without writing invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def archived_import(script):
    original = script.read_text()
    lines = original.splitlines(keepends=True)
    matches = [n for n in ast.parse(original).body if isinstance(n, ast.ImportFrom)
               and n.module == "particlegan" and any(a.name == "get_recipe" for a in n.names)]
    if len(matches) != 1:
        raise ValueError("expected one recipe import")
    node = matches[0]
    remaining = [a.name + (" as " + a.asname if a.asname else "")
                 for a in node.names if a.name != "get_recipe"]
    replacement = "from particlegan import " + ", ".join(remaining) + "\n" if remaining else ""
    replacement += "from benchmarks.gan_v3 import gan_v3_recipe as get_recipe\n"
    lines[node.lineno-1:node.end_lineno] = [replacement]
    adapted = "".join(lines)
    script.write_text(adapted)
    return dict(change="Explicit archived GAN-v3 factory; no current-public-recipe convergence claim",
                original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pr", type=int, choices=[22, 153, 196, 224], required=True)
    ap.add_argument("--mechanism", default="mcar_p50")
    ap.add_argument("--arm", default="gan")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--archived-recipe", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = args.output / "proposal"
    shutil.copytree(args.source, overlay)
    import particlegan
    # Fix package identity before the original sources insert their own ROOT.
    sys.path.insert(0, str(overlay))
    scripts = {22:"experiments/train_circle_transition.py", 153:"experiments/train_animation_transition.py",
               196:"experiments/misgan_toy.py", 224:"examples/e22_stiff_game_reopen.py"}
    script = overlay / scripts[args.pr]
    receipt = dict(pr=args.pr, package=str(Path(particlegan.__file__).resolve()),
                   torch=torch.__version__, python=sys.version,
                   source_sha256={str(p.relative_to(overlay)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(overlay.rglob("*.py"))},
                   scope="new test source on pinned develop; no production config repairs")
    if args.archived_recipe:
        receipt["adapter"] = archived_import(script)
    try:
        mod = runpy.run_path(str(script))
        torch.set_num_threads(1)
        if args.pr == 224:
            results = {}
            for label, factor, cancel in [("native",1.6,False), ("cancel_one_release",1.6,True), ("safe_geometry",.8,False)]:
                fixture = mod["make_fixture"](contracted_stiff_factor=factor)
                rows = []
                for step in range(1, mod["HORIZON"]+1):
                    row = mod["advance"](fixture, step, cancel_first_release=cancel)
                    row["residual"] = fixture.generator.bias.detach().tolist()
                    rows.append(row)
                results[label] = rows
            write(args.output / "observations.json", finite_json(results))
            receipt["final"] = {k:dict(initial_game=v[0]["game_before"], peak_game=max(r["game_after"] for r in v),
                                       final_game=v[-1]["game_after"], releases=[r["step"] for r in v if r["next_scale"]>r["previous_scale"]])
                                for k,v in results.items()}
        else:
            cfg = mod["DEFAULTS"].copy()
            run = args.output / "run"
            cfg.update(out_dir=str(run), **({"live_log":str(args.output/"live.log")} if args.pr != 196
                                           else {"log_path":str(args.output/"live.log")}))
            if args.pr == 22:
                import yaml
                cfg.update(yaml.safe_load((overlay/"configs/circle/radial_hold.yaml").read_text()))
                cfg.update(out_dir=str(run), live_log=str(args.output/"live.log"), device=args.device)
            elif args.pr == 153:
                cfg.update(arm=args.arm, device=args.device, data_dir=str(args.output/"data"))
            else:
                cfg.update(arm="misgan", mechanism=args.mechanism)
                frames, imputations, observed = [], [], []
                train_globals = mod["train"].__globals__
                original_generation = train_globals["generation_metrics"]
                original_imputation = train_globals["imputation_metrics"]
                def generation(problem, samples):
                    value = original_generation(problem, samples)
                    frames.append(problem.to2d(samples).detach().cpu().numpy().copy())
                    return value
                def imputation(problem, samples, post):
                    value = original_imputation(problem, samples, post)
                    # Retain identical ambiguous examples at every evaluator call.
                    ids = (problem.m_test.sum(1)<=1).nonzero().flatten()[:16]
                    if len(ids):
                        flat = samples[:, ids].reshape(-1, samples.shape[-1])
                        imputations.append(problem.to2d(flat).detach().cpu().numpy().reshape(samples.shape[0],len(ids),2).copy())
                        observed.append(ids.detach().cpu().numpy())
                    return value
                train_globals.update(generation_metrics=generation, imputation_metrics=imputation)
            write(args.output/"config.json", cfg)
            result = mod["train"](cfg)
            if args.pr == 196:
                np.savez_compressed(args.output/"observations.npz", live=np.asarray(frames,dtype=np.float32),
                                    imputed=np.asarray(imputations,dtype=np.float32), ids=np.asarray(observed))
            receipt["summary"] = str(run/"summary.json")
        receipt["status"] = "COMPLETE"
    except Exception:
        receipt.update(status="ERROR", error=traceback.format_exc())
        print(receipt["error"], flush=True)
    write(args.output/"replay.json", receipt)
    print(json.dumps(dict(event="PROPOSAL_DONE",pr=args.pr,status=receipt["status"],output=str(args.output))),flush=True)
    return int(receipt["status"]=="ERROR")


if __name__ == "__main__":
    raise SystemExit(main())

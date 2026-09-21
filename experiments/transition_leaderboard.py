#!/usr/bin/env python
"""Persistent transition benchmark across reviewed model/trainer revisions.

Pins completed artifacts and requires the same data, evaluation code, MoG recipe
(except critic conditioning), normalization and reference observations. Unlike
analyze_transition.py, model-source hashes may differ and are reported explicitly.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_transition import preference_audit

PROTOCOL_NODES = {
    "lib/transition.py": ("Transitions", "TransitionScaler", "metrics", "residual", "shuffle_blocks"),
    "lib/trajectory.py": ("Routes",),
    "lib/toy_metrics.py": ("sliced_w1",),
    "experiments/train_transition.py": ("evaluate",),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path):
    path = Path(path)
    summary = json.loads((path/"summary.json").read_text())
    return path, summary


def protocol(path, run):
    code = {}
    with zipfile.ZipFile(path/"source.zip") as archive:
        for filename, names in PROTOCOL_NODES.items():
            source = archive.read(filename)
            if hashlib.sha256(source).hexdigest() != run["provenance"]["sources"][filename]:
                raise ValueError(f"Archived source hash mismatch: {filename}")
            nodes = {node.name: node for node in ast.parse(source).body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            for name in names:
                code[filename+":"+name] = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
    references = {}
    for split in ("train", "test"):
        h = hashlib.sha256()
        with np.load(path/f"{split}_samples.npz") as samples:
            if not np.isfinite(samples["x"]).all():
                raise ValueError("Nonfinite generated samples")
            for key in ("real", "c", "geom", "tick", "group"):
                x = np.ascontiguousarray(samples[key])
                h.update(json.dumps([key, str(x.dtype), list(x.shape)]).encode())
                h.update(x.tobytes())
        references[split] = h.hexdigest()
    cfg = run["config"]
    recipe = {k: v for k, v in run["recipe"].items() if k != "conditioning"}
    if recipe["prior_kind"] != "mog" or recipe["num_particles"] != 1024:
        raise ValueError("This leaderboard requires 1024-component MoG")
    if run["real_draws"] != 2*cfg["steps"]*cfg["batch_size"]:
        raise ValueError("Unexpected real-data update budget")
    return dict(code=code, references=references, recipe=recipe,
                data={k: cfg[k] for k in ("seed", "length", "geometry_mode", "z_dim", "num_particles",
                                          "steps", "batch_size", "eval_per_context", "normalization_samples")},
                normalization=json.loads((path/"normalization.json").read_text()),
                prior=json.loads((path/"prior.json").read_text()), torch=run["environment"]["torch"],
                real_draws=run["real_draws"])


def register(board, name, path, note=""):
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in name):
        raise ValueError("IDs must use lowercase letters, digits, hyphens or underscores")
    path, run = load_run(path)
    resolved = path.resolve()
    relative = str(resolved.relative_to(ROOT))
    current = protocol(path, run)
    if board.get("protocol") is None:
        board["protocol"] = current
        board["generator_budget"] = run["parameters"]["G"]
    if current != board["protocol"]:
        differences = [key for key in current if current[key] != board["protocol"][key]]
        raise ValueError(f"Incompatible benchmark protocol: {differences}")
    if abs(run["parameters"]["G"]/board["generator_budget"]-1) > .05:
        raise ValueError("Generator parameter count differs from baseline by more than 5%")
    encoder_count = run["parameters"].get("E", 0)
    if bool(encoder_count) != bool(run["config"].get("encoder", False)):
        raise ValueError("Encoder configuration/parameter count mismatch")
    if encoder_count:
        if "encoder_budget" not in board:
            board["encoder_budget"] = encoder_count
        if encoder_count != board["encoder_budget"]:
            raise ValueError("Encoder cohort requires the same E parameter budget")
    entry = dict(id=name, path=relative, summary_sha256=digest(path/"summary.json"),
                 source_zip_sha256=digest(path/"source.zip"), note=note)
    for previous in board["entries"]:
        if previous["id"] == name or previous["path"] == relative:
            if previous == entry:
                return
            raise ValueError("ID or run already registered; baseline entries are immutable")
    board["entries"].append(entry)


def verified_runs(board):
    runs = []
    variants = set()
    for entry in board["entries"]:
        path, run = load_run(ROOT/entry["path"])
        if digest(path/"summary.json") != entry["summary_sha256"] or digest(path/"source.zip") != entry["source_zip_sha256"]:
            raise ValueError(f"Pinned artifacts changed: {entry['id']}")
        if protocol(path, run) != board["protocol"]:
            raise ValueError(f"Protocol changed: {entry['id']}")
        if abs(run["parameters"]["G"]/board["generator_budget"]-1) > .05:
            raise ValueError("Generator budget changed")
        e_count = run["parameters"].get("E", 0)
        if bool(e_count) != bool(run["config"].get("encoder", False)):
            raise ValueError("Encoder configuration/parameter count mismatch")
        if e_count and e_count != board.get("encoder_budget"):
            raise ValueError("Encoder budget changed")
        cfg = {"encoder": False, "shared_state_critic": False, "encoder_width": 128,
               "real_encoding_weight": 1., "synthetic_reconstruction_weight": 1., "d_conditioning": "ucd", "g_class_scale": 1.0, "g_context_scale": 1.0, **run["config"]}
        # Include future explicit experiment factors, while ignoring output/log settings.
        variant = json.dumps({k: v for k, v in cfg.items() if k not in
                              ("out_dir", "live_log", "log_interval", "save_checkpoint", "device")}, sort_keys=True)
        if variant in variants:
            raise ValueError("Duplicate experimental configuration; no repeat runs on this board")
        variants.add(variant)
        runs.append((entry, path, run))
    return sorted(runs, key=lambda item: item[2]["final"]["test"]["joint_sw1"])


def render(board, out):
    runs = verified_runs(board)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = board["protocol"]["recipe"]
    lines = ["# Transition GAN leaderboard", "",
             "Primary score: conditional joint SW1 on the fixed held-out geometries, lower is better. "
             "Transition residual and class separation are required companion diagnostics. "
             "These geometries now serve as a development benchmark because we reuse them for model selection; "
             "this leaderboard is not an untouched generalization test.", "",
             f"Every entry uses {recipe['num_particles']:,} MoG components, bcap, "
             f"{recipe['total_steps']:,} updates, batch {recipe['batch_size']}, and training seed "
             f"{board['protocol']['data']['seed']}. Generator parameter counts stay within 5% of baseline. "
             "E and D capacity and wall time are reported; added encoders/critics cost extra compute. "
             "No seed-only repeats.", "",
             "The registry pins summary/source archives and checks identical data/evaluation functions, "
             "recipe settings except critic conditioning, normalization, prior initialization metadata, "
             "and reference samples. Model/trainer revisions are allowed and recorded; this does not "
             "claim identical critic initialization across different architectures.", "",
             "Encoder rows form a paired-supervision cohort: E adds reconstruction/prediction losses and "
             "synthetic composition. They reuse the same real training draws and frozen prior-sampling benchmark, "
             "but are not adversarial-only or total-capacity-matched comparisons. E counts must match within "
             "that cohort; the historical G budget check is unchanged. Prediction metrics are separate below.", "",
             "| Rank | Run | Critic conditioning | G class gain | G geometry/time gain | Joint SW1 ↓ | Interp. | Extrap. | Residual ↓ | G / E / D parameters | Train seconds |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for rank, (entry, path, run) in enumerate(runs, 1):
        cfg, m = run["config"], run["final"]["test"]
        name = entry["id"]
        conditioning = cfg.get("d_conditioning", "ucd")
        lines.append(f"| {rank} | {name} | {conditioning} | {cfg.get('g_class_scale', 1.):g} | "
                     f"{cfg.get('g_context_scale', 1.):g} | {m['joint_sw1']:.5f} | "
                     f"{run['final']['interpolation']['joint_sw1']:.5f} | {run['final']['extrapolation']['joint_sw1']:.5f} | "
                     f"{m['consistency_mean']:.5f} | {run['parameters']['G']:,} / {run['parameters'].get('E', 0):,} / {run['parameters']['D']:,} | {run['train_seconds']:.1f} |")
        audit = preference_audit(path, run)
        rows.append(dict(id=name, path=entry["path"], note=entry["note"], conditioning=conditioning,
                         g_class_scale=cfg.get("g_class_scale", 1.),
                         g_context_scale=cfg.get("g_context_scale", 1.),
                         cohort="paired encoder" if cfg.get("encoder", False) else "adversarial only",
                         inference=run.get("inference"), architecture=cfg["architecture"], critics=cfg["critic_mode"], parameters=run["parameters"],
                         metrics={k: v for k, v in m.items() if k != "contexts"}, preference=audit,
                         interpolation=run["final"]["interpolation"]["joint_sw1"],
                         extrapolation=run["final"]["extrapolation"]["joint_sw1"],
                         train_seconds=run["train_seconds"], summary_sha256=entry["summary_sha256"],
                         source_zip_sha256=entry["source_zip_sha256"]))
        for filename in ("viewer.html", "transitions.png", "config.yaml"):
            shutil.copy2(path/filename, out/f"{name}_{filename}")
    floor = runs[0][2]["reference_floor"]["test"]
    lines += ["", f"Reference-vs-reference joint SW1 floor: **{floor['joint_sw1']:.5f}**.", "",
              "## Marginals and preference", "",
              "Upper-side frequency is measured near the route midpoint, using the observed analytic centerline. "
              "Targets are 0.8 for class 0 and 0.3 for class 1. Similar frequencies across classes indicate "
              "weak class conditioning; these frequencies alone do not establish support validity.", "",
              "| Run | State SW1 ↓ | Action SW1 ↓ | Next SW1 ↓ | Upper class 0 | Upper class 1 | Coverage | Precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]
        a, b = row["preference"]["classes"]
        lines.append(f"| {row['id']} | {m['state_sw1']:.5f} | {m['action_sw1']:.5f} | {m['next_state_sw1']:.5f} | "
                     f"{a['generated_upper']:.3f} | {b['generated_upper']:.3f} | {m['coverage']:.3f} | {m['precision']:.3f} |")
    encoded = [row for row in rows if row["inference"] is not None]
    if encoded:
        lines += ["", "## Encoder cohort: prediction and composition", "",
                  "Real-input prediction supplies st/at and measures G3 next-state error in physical units. "
                  "Synthetic SW1 evaluates G1/G2 -> E -> G3. Neither replaces the original prior-sample rank. "
                  "Here action is displacement: the analytic st + at control has zero prediction error.", "",
                  "| Run | Next-state L2 ↓ | p95 ↓ | Synthetic joint SW1 ↓ | Synthetic residual ↓ | E used / effective components |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in encoded:
            inf = row["inference"]["test"]
            paired, syn = inf["paired"], inf["synthetic"]
            routing = paired["real_routing"]
            lines.append(f"| {row['id']} | {paired['next_l2']:.5f} | {paired['next_l2_p95']:.5f} | "
                         f"{syn['joint_sw1']:.5f} | {syn['consistency_mean']:.5f} | "
                         f"{routing['used']} / {routing['effective']:.1f} |")
    lines += ["", "## Experiment notes and artifacts", ""]
    for row in rows:
        name = row["id"]
        lines += [f"- **{name}:** {row['note']} [Config]({name}_config.yaml) · [Viewer]({name}_viewer.html) · [Plot]({name}_transitions.png)"]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    (out/"leaderboard.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    return out/"README.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/transition/leaderboard.json")
    parser.add_argument("--out-dir", default="reports/transition/leaderboard")
    parser.add_argument("--add", nargs=2, action="append", default=[], metavar=("ID", "RUN_DIR"))
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    registry = Path(args.registry)
    board = json.loads(registry.read_text()) if registry.exists() else dict(version=1, entries=[])
    for name, path in args.add:
        register(board, name, path, args.note)
    if not board["entries"]:
        parser.error("Register at least one completed run")
    verified_runs(board)
    if args.add:
        registry.parent.mkdir(parents=True, exist_ok=True)
        temporary = registry.with_suffix(".tmp")
        temporary.write_text(json.dumps(board, indent=2, allow_nan=False)+"\n")
        temporary.replace(registry)
    print(render(board, args.out_dir))


if __name__ == "__main__":
    main()

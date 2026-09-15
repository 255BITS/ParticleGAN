#!/usr/bin/env python
"""Export certified, equal-budget trajectory experiments and a visual gallery."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, seen = [], set()
    for root in args.roots:
        for path in sorted(Path(root).glob("*/summary.json")):
            s = json.loads(path.read_text())
            cert = json.loads((path.parent / "run_grid_complete.json").read_text())
            if cert["summary_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or canonical(cert["config"]) != canonical(s["config"]) or cert["provenance"] != s["provenance"]:
                raise ValueError(f"Invalid completion certificate: {path}")
            name = path.parent.name
            if name in seen:
                raise ValueError(f"Duplicate run name: {name}")
            seen.add(name)
            runs.append((name, path.parent, s))
    if not runs:
        raise ValueError("No completed runs")
    protocols = {(s["config"]["steps"], s["config"]["batch_size"], s["config"]["eval_per_context"]) for _, _, s in runs}
    sources = {canonical(s["provenance"]) for _, _, s in runs}
    if len(protocols) != 1 or len(sources) != 1:
        raise ValueError("Runs must share budget, evaluation protocol, and source provenance")
    runs.sort(key=lambda item: item[2]["final"]["test"]["conditional_sw1"])
    rows = []
    for name, path, s in runs:
        test, train = s["final"]["test"], s["final"]["train"]
        contexts = test["contexts"]
        variances = [v for r in contexts[:6] for v in r["coefficient_variance_ratio"] if v is not None]
        rows.append(dict(name=name, test_sw1=test["conditional_sw1"], route_tv=test["route_tv"],
                         test_valid=test["valid"], interpolation_valid=statistics.mean(r["valid"] for r in contexts[:6]),
                         extrapolation_valid=statistics.mean(r["valid"] for r in contexts[6:]),
                         collision=test["collision"], interpolation_variance_ratio=statistics.median(variances) if variances else None,
                         train_valid=train["valid"], train_sw1=train["conditional_sw1"],
                         train_seconds=s["train_seconds"], real_draws=s["real_draws"]))
        dest = out / name
        dest.mkdir(exist_ok=True)
        for file in ("summary.json", "config.yaml", "metrics.jsonl", "provenance.json", "run_grid_complete.json", "viewer.html", "test_routes.png", "route_mass.png", "particle_probe.png"):
            if (path / file).exists():
                shutil.copyfile(path / file, dest / file)
        # One animation suffices for the report; all raw runs retain their own.
        if name == "learned":
            shutil.copyfile(path / "futures.gif", dest / "futures.gif")
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2)+"\n")
    lines = ["# Trajectory comparison", "", "Sorted by held-out conditional SW1; this is not a universal ranking. Interpolation and extrapolation are reported separately. Variance ratios use valid paths only; target 1.", "",
             "| Recipe | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Collision ↓ | Interp. variance ratio | Training s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        v = r["interpolation_variance_ratio"]
        vs = f"{v:.2f}" if v is not None else "missing"
        lines.append(f"| {r['name']} | {r['test_sw1']:.4f} | {r['route_tv']:.4f} | {r['interpolation_valid']:.1%} | {r['extrapolation_valid']:.1%} | {r['collision']:.1%} | {vs} | {r['train_seconds']:.1f} |")
    floor = runs[0][2]["reference_floor"]["test"]
    lines.extend(["", f"Real-vs-real test floor: SW1 {floor['conditional_sw1']:.4f}; route TV {floor['route_tv']:.4f}; validity {floor['valid']:.1%}; collision {floor['collision']:.1%}.", "",
                  "All runs have matching source fingerprints and completion certificates. Matched updates/sample exposure do not mean matched wall time or parameter count. Single-seed configuration comparisons; no significance claim.", ""])
    (out / "TABLE.md").write_text("\n".join(lines))
    options = "".join(f'<option value="{html.escape(r["name"])}">{html.escape(r["name"])}</option>' for r in rows)
    page = f'''<!doctype html><meta charset="utf-8"><title>Trajectory experiment gallery</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:24px auto;background:#eef2f7}}iframe{{width:100%;height:1050px;border:0}}select{{padding:10px}}</style>
<h1>ParticleGAN: possible futures</h1><p>Choose a recipe, then change scene, route preference, and physical time in the viewer. Scene 4 extrapolates beyond the training geometries; scenes 1–3 interpolate.</p>
<select id="recipe">{options}</select><iframe id="viewer" title="Trajectory viewer"></iframe>
<script>const r=document.getElementById('recipe'),v=document.getElementById('viewer');function show(){{v.src=r.value+'/viewer.html'}}r.onchange=show;show();</script>'''
    (out / "index.html").write_text(page)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

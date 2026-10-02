"""PR236 goal media from exact observations; explicit future public-API replay.

Ordinary export never trains or runs a model forward. --reproduce invokes the
original public caller once, with its unchanged 500-update/60-second protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from .recent_toy_media import Inputs, finite, finite_health, sha, write

ROOT = Path(__file__).resolve().parents[2]
HEAD = "e71265fd7b77ae2ab70ccbf8b7f0049c3d539b84"
CARD = "docs/routed_generator_batch_20261002_results.json"
CARD_SHA = "cb1639d21790915f2a6387e4d65bcfc42fdf5fb1314c6b68b724ee7833c49498"
DRIVER = "examples/routed_generator_batch.py"
PROTOCOL = "examples/routed_generator_batch_protocol.json"
ARMS = ("G16", "G64")
STEPS, SECONDS, LIMIT = 500, 60, .9
MEDIA_STEPS = (0, 1, *range(50, 501, 50))
STREAMS = {"d_data", "d_gaussian", "g_data", "g_gaussian"}


def load_card():
    if sha(ROOT / CARD) != CARD_SHA:
        raise ValueError("public PR236 card differs from the reviewed exact head")
    return json.loads((ROOT / CARD).read_text())


def active_sources(card):
    """Publication's unchanged caller verifies its exact 37 bound sources."""
    path = ROOT / PROTOCOL
    if sha(path) != card["publication"]["protocol_sha256"]:
        raise ValueError("publication protocol source identity differs")
    protocol = json.loads(path.read_text())
    hashes = protocol["source_hashes"]
    if not hashes or DRIVER not in hashes or len(hashes) != card["publication"]["protected_source_file_count"]:
        raise ValueError("missing complete publication source manifest")
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(ROOT / relative) != expected:
            raise ValueError(f"publication source drift: {name}")
    return {"protocol_sha256": sha(path), "source_hashes": deepcopy(hashes)}


def original_gates(arms, *, streams_match, source_match):
    """The original frozen endpoint laws, independent of recorded booleans."""
    checks = {"completed_fixed_budget": all(a["steps"] == STEPS for a in arms.values()),
              "caller_streams_match": streams_match, "frozen_sources_match": source_match}
    for name, arm in arms.items():
        h, control = arm["health"], arm["control"]
        checks[name + "_finite_native_health"] = (
            h["finite_steps"] == STEPS and h["min_dense_rows"] == 128
            and h["ka2_applied_calls"] > 0 and h["ownership"] and h["frozen_host"]
            and control["rows"]["counters"]["updates"] == STEPS
            and control["counters"]["evals"] > 0 and control["counters"]["probes"] > 0)
        checks[name + "_converges_10_percent"] = (
            arm["endpoint"]["live_excess_error"] <= LIMIT * arm["initial"]["live_excess_error"])
    checks["G64_improves_10_percent"] = (
        arms["G64"]["endpoint"]["live_excess_error"] <= LIMIT * arms["G16"]["endpoint"]["live_excess_error"])
    return checks


def verify(root, card, inputs, *, fresh=False):
    root = Path(root)
    protocol_sha = card["publication"]["protocol_sha256"] if fresh else card["protocol_sha256"]
    protocol = inputs.read(root / "protocol.json", protocol_sha)
    source_sha = (protocol["source_hashes"][DRIVER] if fresh else
                  card["publication"]["original_scientific_driver_sha256"])
    inputs.bind(root / "source.py", source_sha)
    if (protocol["external_steps_per_arm"] != STEPS or protocol["external_total_seconds"] != SECONDS
            or protocol["threads"] != 1 or protocol["arms"] != {
                "G16": {"D_batch": 16, "G_batch": 16}, "G64": {"D_batch": 16, "G_batch": 64}}
            or protocol["primary_metric"]["threshold_both_arm_convergence_ratio"] != LIMIT
            or protocol["primary_metric"]["threshold_G64_over_G16"] != LIMIT):
        raise ValueError("changed fixed public batch, budget or numerical gate")
    result = inputs.read(root / "result.json", None if fresh else card["artifact_hashes"]["result.json"])
    if (set(result["arms"]) != set(ARMS) or result["protocol_sha256"] != protocol_sha
            or result["failure"] is not None or result["status"] not in {"passed", "failed"}):
        raise ValueError("missing complete original scientific protocol")
    for key in ("fixture_sha256", "teacher_sha256", "initial_hashes"):
        if result.get(key) != card[key]:
            raise ValueError(f"changed or missing fixed {key} identity")
    finite([result["elapsed_seconds"]])
    if not 0 < result["elapsed_seconds"] <= SECONDS:
        raise ValueError("original fixed wall budget was exceeded")
    evaluations, panel_histories = {}, {}
    for name in ARMS:
        arm = result["arms"][name]
        streams = arm["caller_stream_hashes"]
        if (set(streams) != STREAMS or any(not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value) for value in streams.values())):
            raise ValueError("exactly four valid caller RNG source hashes are required")
        path = inputs.bind(root / f"{name}.jsonl", None if fresh else card["artifact_hashes"][f"{name}.jsonl"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (arm["steps"] != STEPS or [r["step"] for r in rows] != list(range(1, STEPS + 1))
                or any(r["d_batch"] != 16 or r["g_batch"] != int(name[1:]) or r["dense_rows"] != 128 for r in rows)):
            raise ValueError("partial or changed native update coverage")
        for row in rows:
            finite([row[k] for k in ("loss_g", "loss_d", "penalty", "sigma")])
            if len(row["quadratic_weights"]) != 2:
                raise ValueError("both learned channel-energy weights are required")
            finite(row["quadratic_weights"])
            finite_health(row)
        panels = [row["caller_panel_sha256"] for row in rows]
        if (any(not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value) for value in panels)
                or hashlib.sha256("\n".join(panels).encode()).hexdigest() != arm["caller_panel_history_sha256"]):
            raise ValueError("actual caller panel history differs from its summary binding")
        panel_histories[name] = panels
        points = [{"step": 0, **arm["initial"]},
                  *[{"step": r["step"], **r["evaluation"]} for r in rows if "evaluation" in r]]
        if [p["step"] for p in points] != list(MEDIA_STEPS):
            raise ValueError("missing or reordered actual evaluation observations")
        for point in points:
            finite([point[k] for k in ("live_excess_error", "served_excess_error", "irreducible_population_error")])
            if (point["live_excess_error"] < 0 or point["served_excess_error"] < 0
                    or point["irreducible_population_error"] != .0225
                    or point["served_source"] not in {"fast", "averaged"}):
                raise ValueError("invalid original clean/served excess observation")
        if arm["initial"]["live_excess_error"] <= 0:
            raise ValueError("positive initial excess required for frozen relative gates")
        if {k: v for k, v in points[-1].items() if k != "step"} != arm["endpoint"]:
            raise ValueError("endpoint differs from last actual observation")
        if any(arm["health"].get(k) is not True for k in
               ("ownership", "frozen_host", "critic_score_active_gradients", "generator_live_gradients")):
            raise ValueError("native ownership/gradient health failed")
        finite_health(arm["health"])
        if not fresh:
            published = card["arms"][name]
            if (arm["initial"] != published["initial"] or arm["endpoint"] != published["endpoint"]
                    or points[1:] != published["evaluation_points"]):
                raise ValueError("numeric observations differ from original public evidence")
        evaluations[name] = points
    a, b = (result["arms"][name] for name in ARMS)
    streams_match = (panel_histories["G16"] == panel_histories["G64"]
                     and a["caller_stream_hashes"] == b["caller_stream_hashes"]
                     and a["caller_panel_history_sha256"] == b["caller_panel_history_sha256"])
    checks = original_gates(result["arms"], streams_match=streams_match, source_match=True)
    if (result["checks"] != checks or any(type(v) is not bool for v in result["checks"].values())
            or result["status"] != ("passed" if all(checks.values()) else "failed")):
        raise ValueError("recorded verdict contradicts original frozen gates")
    health_keys = [k for k in checks if not k.endswith("_converges_10_percent") and k != "G64_improves_10_percent"]
    if not all(checks[k] for k in health_keys):
        raise ValueError("full native health/caller protocol is required before goal media")
    return {"scientific_status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "cohort": "fresh exact-protocol cohort" if fresh else "original frozen V2",
            "fixture_sha256": result["fixture_sha256"], "teacher_sha256": result["teacher_sha256"],
            "initial_hashes": deepcopy(result["initial_hashes"]),
            "evaluations": evaluations, "actual_updates": list(MEDIA_STEPS),
            "native_updates_per_arm": STEPS, "quality_gate_scope": "Fixed500 endpoint only; not sustained dominance or equal compute",
            "actual_task": deepcopy(card["real100"]["quality_gate"])}


def reproduce(path, card):
    """Explicit future execution: one unchanged original API child, no retries."""
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("reproduction requires a new artifact directory")
    before = active_sources(card)
    path.mkdir(parents=True)
    target = path / "attempt"
    command = [sys.executable, DRIVER, "--output", str(target)]
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    with (path / "caller.log").open("w") as log:
        child = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=SECONDS)
    if child.returncode not in (0, 1):
        raise ValueError("API child failed; no complete media evidence")
    if active_sources(card) != before:
        raise ValueError("API sources changed during reproduction")
    write(path / "reproduction.json", {"source": before, "command": command,
          "returncode": child.returncode, "elapsed_seconds": time.monotonic() - started,
          "cap_seconds": SECONDS, "scope": "Separate fresh protocol; original scientific evidence remains immutable"})
    return target


def render(data, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    curves = data["evaluations"]
    images = []
    for index, step in enumerate(MEDIA_STEPS):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))
        for arm, color in (("G16", "#75869b"), ("G64", "#be375d")):
            points = curves[arm][:index + 1]
            x, y = [p["step"] for p in points], [p["live_excess_error"] for p in points]
            axes[0].plot(x, y, marker=".", label=arm, color=color)
            axes[1].plot(x, [v / curves[arm][0]["live_excess_error"] for v in y], marker=".", color=color, label=arm)
        axes[0].set(ylim=(0, max(p["live_excess_error"] for c in curves.values() for p in c) * 1.1),
                    title="Actual clean population excess", ylabel="Excess MSE (evaluation only)")
        axes[0].set_yscale("symlog", linthresh=1e-6)
        axes[0].legend(fontsize=8)
        axes[1].axhline(LIMIT, linestyle="--", color="#333333", label="500 endpoint limit")
        axes[1].set(ylim=(0, 1.08), title="Both arms improve at least 10%", ylabel="Excess / own initial excess")
        ratios = [b["live_excess_error"] / a["live_excess_error"] for a, b in zip(curves["G16"], curves["G64"])]
        axes[2].plot(MEDIA_STEPS[:index + 1], ratios[:index + 1], marker=".", color="#be375d")
        axes[2].axhline(LIMIT, linestyle="--", color="#333333", label="500 endpoint limit")
        axes[2].axhline(1, color="#bbbbbb", linewidth=.8)
        axes[2].set(ylim=(0, max(1.08, max(ratios) * 1.08)), title="G64 improves at least 10% vs G16", ylabel="G64 / G16 excess")
        for ax in axes:
            ax.set(xlim=(0, STEPS), xlabel="Actual API updates per arm")
            ax.grid(alpha=.15)
        fig.suptitle("Does G64 improve conditional-mean recovery under fixed +/- nuisance?", fontsize=11)
        fig.tight_layout(rect=(0, .25, 1, .94))
        hundred = 100 * (ratios[MEDIA_STEPS.index(100)] - 1)
        footer = (f"Toy {data['scientific_status']} ({data['cohort']}) | actual update {step}/500 per arm | gates evaluated at500 only\n"
                  f"D16 fixed; G64 uses four times the G examples. At100 G64 change={hundred:+.2f}% (positive is worse).\n"
                  "Separate actual-task gate FAIL: gain0.000822 < required0.001. No continuation/default promotion.\n"
                  "Retained measured curves only: no model inference, training, rescoring or interpolated frames.")
        fig.text(.025, .025, footer, fontsize=8, va="bottom")
        buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=100); plt.close(fig)
        buffer.seek(0); images.append(Image.open(buffer).convert("RGB"))
    images[0].save(path, save_all=True, append_images=images[1:], duration=[300] * (len(images) - 1) + [1800], loop=0, optimize=False)
    with Image.open(path) as gif:
        if gif.n_frames != len(MEDIA_STEPS):
            raise ValueError("goal GIF lost an actual observation")
    for image in images: image.close()
    return {"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size, "frames": len(MEDIA_STEPS)}


def export(source, output, *, fresh=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("media output must be separate from immutable input artifacts")
    card, inputs = load_card(), Inputs()
    if fresh:
        proof = inputs.read(source.parent / "reproduction.json")
        if proof["source"] != active_sources(card) or proof["cap_seconds"] != SECONDS or proof["returncode"] not in (0, 1):
            raise ValueError("missing exact public-API reproduction identity")
    data = verify(source, card, inputs, fresh=fresh)
    if fresh and proof["returncode"] != int(data["scientific_status"] == "FAIL"):
        raise ValueError("API child exit code contradicts its full scientific result")
    files = ("benchmarks/toy_audit/batch_toy_media.py", "benchmarks/toy_audit/recent_toy_media.py")
    hashes = {name: sha(ROOT / name) for name in files}
    numeric = json.dumps(data, sort_keys=True, allow_nan=False)
    if not inputs.unchanged(): raise ValueError("input archive changed before rendering")
    output.mkdir(parents=True, exist_ok=False)
    artifact = render(data, output / "goal.gif")
    if (not inputs.unchanged() or hashes != {name: sha(ROOT / name) for name in files}
            or numeric != json.dumps(data, sort_keys=True, allow_nan=False)):
        raise ValueError("source, original evidence or numeric observations changed during export")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"schema": "particlegan_pr236_goal_media_v1", "reviewed_head": HEAD,
        "cohort": "fresh_exact_protocol" if fresh else "original_frozen_V2",
        "definition_rating": "4/5 bounded conditional-nuisance endpoint diagnostic",
        "frozen_176_campaign_changed": False, "training_or_model_rescoring_during_export": False,
        "raw_inputs": inputs.files, "raw_files_unchanged": True, "data": data, "media": artifact,
        "exporter_source": {"commit": commit, "files_sha256": hashes}}
    write(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduce", action="store_true", help="Explicit future execution of the fixed public caller before media export")
    args = parser.parse_args(argv)
    try:
        source = reproduce(args.input, load_card()) if args.reproduce else args.input
        receipt = export(source, args.output, fresh=args.reproduce)
        print(json.dumps({"scientific_status": receipt["data"]["scientific_status"], "media": receipt["media"]}))
        return int(receipt["data"]["scientific_status"] == "FAIL")
    except (ValueError, KeyError, FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"INCOMPLETE: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

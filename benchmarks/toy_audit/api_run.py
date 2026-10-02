"""Run public-API toy variants and render their actual target/output states.

Example: python -m benchmarks.toy_audit.api_run --list
Raw checkpoints, arrays and progress logs belong outside Git. The compact
receipt and final GIF can be published without pretending a short run is a
completed default-budget quality test.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import subprocess
import sys
import textwrap
import time

import numpy as np
import torch

from . import api_contract as contract


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, (np.generic,)):
        return json_value(value.item())
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_value(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(part) for part in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def isolated_evaluation():
    """An evaluation cannot advance ambient training RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def source_identity():
    root = contract.ROOT
    paths = set((root / "particlegan").glob("*.py"))
    paths.update((root / "benchmarks/toy_audit").glob("api_*.py"))
    paths.update((root / "benchmarks/toy_audit/api_sources").glob("*.py"))
    # Native example hosts are also loaded with runpy; that does not retain a
    # module in sys.modules, so bind their code explicitly.
    paths.update((root / "examples").glob("e22_*.py"))
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
                paths.add(path)
    sources = {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          text=True, capture_output=True).stdout.strip()
    return {"commit": head, "files_sha256": sources}


def _points(values):
    values = contract.array(values)
    if values.ndim == 1:
        return np.column_stack((np.arange(len(values)), values))
    if values.ndim > 2 and values.shape[-1] == 2:
        return values.reshape(-1, 2)
    values = values.reshape(len(values), -1)
    if values.shape[1] == 1:
        return np.column_stack((np.arange(len(values)), values[:, 0]))
    return values[:, :2]


def _line_paths(values):
    values = contract.array(values)
    if values.ndim > 2 and values.shape[-1] == 2:
        return list(values.reshape(-1, values.shape[-2], 2))
    return [_points(values)]


def _image_grid(values, limit=8):
    values = contract.array(values)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[1] not in (1, 3, 4):
        raise ValueError("image goal views require NCHW grayscale/RGB values")
    images = np.moveaxis(values[:limit], 1, -1)
    if images.shape[-1] == 1:
        images = images[..., 0]
    spacer_shape = (images.shape[1], 1, *images.shape[3:])
    spacer = np.full(spacer_shape, .5)
    joined = []
    for index, image in enumerate(images):
        if index:
            joined.append(spacer)
        joined.append(image)
    return np.concatenate(joined, axis=1)


def _view_limits(records, roles=("target", "samples")):
    bounds = {}
    for record in records:
        for index, view in enumerate(record["views"]):
            if view["kind"] in {"image", "text"}:
                continue
            for role in roles:
                points = _points(view[role])
                points = points[np.isfinite(points).all(1)]
                if len(points):
                    low, high = points.min(0), points.max(0)
                    if index in bounds:
                        low, high = np.minimum(bounds[index][0], low), np.maximum(bounds[index][1], high)
                    bounds[index] = (low, high)
    return bounds


def render_gif(case, records, path, *, full_budget, requested_steps, final_verdict=None):
    """Reference and actual outputs share fixed axes across real observations."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image

    views_count = max(len(record["views"]) for record in records)
    columns = min(3, views_count)
    rows = math.ceil(views_count / columns)
    fixed = _view_limits(records)
    reference = _view_limits(records, roles=("target",))
    annotations = []
    frames = []
    for record in records:
        fig, axes = plt.subplots(rows, columns, figsize=(4.25 * columns, 3.05 * rows + 1.45),
                                 squeeze=False)
        for index, ax in enumerate(axes.flat):
            if index >= len(record["views"]):
                ax.set_visible(False)
                continue
            view = dict(record["views"][index])
            target, samples = contract.array(view["target"]), contract.array(view["samples"])
            if case["id"] == "api-circle-controller" and view["kind"] == "line":
                low, high = reference[index]
                margin = .08 * np.maximum(high - low, .1)
                view["xlim"] = [float(low[0] - margin[0]), float(high[0] + margin[0])]
                view["ylim"] = [float(low[1] - margin[1]), float(high[1] + margin[1])]
                points = _points(samples)
                outside = int(((points < low - margin) | (points > high + margin)).any(1).sum())
                view["caption"] = (view.get("caption", "") +
                    f" Axes show the reference band; {outside}/{len(points)} retained predicted states lie outside. The gate scores the complete rollout.")
                annotations.append({"step": record["step"], "view": index,
                                    "reference_camera": {"xlim": view["xlim"], "ylim": view["ylim"]},
                                    "outside_view_states": outside, "retained_view_states": len(points)})
            if view["kind"] == "text":
                reference = "Desired words\n" + "\n".join(view["target_labels"][:8])
                generated = "Actual decoded output\n" + "\n".join(view["sample_labels"][:8])
                ax.text(.03, .93, reference, transform=ax.transAxes, va="top",
                        fontsize=7.5 if rows > 1 else 10, family="monospace", color="#586a80")
                ax.text(.53, .93, generated, transform=ax.transAxes, va="top",
                        fontsize=7.5 if rows > 1 else 10, family="monospace", color="#aa254c")
                ax.set_axis_off()
            elif view["kind"] == "image":
                top, bottom = _image_grid(target), _image_grid(samples)
                width = max(top.shape[1], bottom.shape[1])
                def pad(image):
                    return np.pad(image, ((0, 0), (0, width - image.shape[1]),
                                          *((0, 0),) * (image.ndim - 2)), constant_values=.5)
                top, bottom = pad(top), pad(bottom)
                spacer = np.full((1, width, *top.shape[2:]), .5)
                grid = np.concatenate((top, spacer, bottom))
                if grid.ndim == 3:
                    vmin, vmax = view.get("vmin", 0), view.get("vmax", 1)
                    grid = np.clip((grid - vmin) / max(vmax - vmin, 1e-12), 0, 1)
                ax.imshow(np.ma.masked_invalid(grid), cmap="gray", vmin=view.get("vmin", 0),
                          vmax=view.get("vmax", 1), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([top.shape[0] / 2, top.shape[0] + 1 + bottom.shape[0] / 2],
                              view.get("row_labels", ["Desired", "API output"]), fontsize=8)
            elif view["kind"] == "bar":
                a, b = target.reshape(-1), samples.reshape(-1)
                if len(a) != len(b):
                    raise ValueError("bar goal reference and output must have matching bins")
                x = np.arange(len(a))
                ax.bar(x - .18, a, width=.36, color="#9ba8b6", label="Desired")
                ax.bar(x + .18, b, width=.36, color="#ce476a", label="API output")
                ax.legend(fontsize=7)
                if index in fixed:
                    ax.set_ylim(min(0, float(fixed[index][0][1]) * 1.08),
                                max(.05, float(fixed[index][1][1]) * 1.08))
            else:
                for values, color, label in ((target, "#91a0b0", "Desired/reference"),
                                              (samples, "#ce476a", "API output")):
                    points = _points(values)
                    points = points[np.isfinite(points).all(1)]
                    if view["kind"] == "line":
                        for path_index, actual_path in enumerate(_line_paths(values)[:12]):
                            actual_path = actual_path[np.isfinite(actual_path).all(1)]
                            ax.plot(actual_path[:, 0], actual_path[:, 1], color=color,
                                    label=label if path_index == 0 else None, linewidth=1.1, alpha=.75)
                    else:
                        ax.scatter(points[:, 0], points[:, 1], color=color, label=label,
                                   s=4, alpha=.55, rasterized=True)
                if index in fixed:
                    low, high = fixed[index]
                    span = np.maximum(high - low, .1)
                    ax.set_xlim(*(view.get("xlim") or [low[0] - .06 * span[0], high[0] + .06 * span[0]]))
                    ax.set_ylim(*(view.get("ylim") or [low[1] - .06 * span[1], high[1] + .06 * span[1]]))
                ax.legend(fontsize=7, loc="best")
            if case["id"] in {"api-routes-discrete", "api-routes-continuous"}:
                raw_geometry = view.get("caption", "").partition("obstacle center/radius=")[2]
                if raw_geometry:
                    geometry = ast.literal_eval(raw_geometry)
                    if len(geometry) != 3 or not all(math.isfinite(x) for x in geometry) or geometry[2] <= 0:
                        raise ValueError("retained route obstacle geometry is invalid")
                    ax.add_patch(Circle(geometry[:2], geometry[2], facecolor="#d9cbb5",
                                        edgecolor="#7a6040", alpha=.6))
                    annotations.append({"step": record["step"], "view": index,
                                        "reference_obstacle": {"center": geometry[:2], "radius": geometry[2]}})
            if view.get("xlim"):
                ax.set_xlim(*view["xlim"])
            if view.get("ylim"):
                ax.set_ylim(*view["ylim"])
            if view.get("yscale"):
                ax.set_yscale(view["yscale"])
            ax.set_title(textwrap.fill(view["title"], 45), fontsize=10)
            ax.set_xlabel(view.get("xlabel", ""), fontsize=8)
            ax.set_ylabel(view.get("ylabel", ""), fontsize=8)
            invalid = int((~np.isfinite(samples)).sum())
            if invalid:
                ax.text(.5, .5, f"FAIL: {invalid} nonfinite output values", transform=ax.transAxes,
                        ha="center", color="#a31436", bbox={"facecolor": "white", "alpha": .9})
            if view.get("caption"):
                ax.text(.5, -.23, textwrap.fill(view["caption"], 65), transform=ax.transAxes,
                        ha="center", va="top", fontsize=7)
        fig.suptitle(textwrap.fill(case["goal"], 43 * columns), fontsize=11, y=.98)
        budget = "default protocol" if full_budget else f"short run; default {case['default_steps']} updates"
        instant = "PASS" if record["passed"] else "FAIL"
        metrics = "; ".join(f"{name}={value:.4g}" for name, value in list(record["metrics"].items())[:6])
        footer = f"{case['id']} | update {record['step']}/{requested_steps} | metric {instant} | {budget}\n{metrics}"
        if final_verdict is not None:
            footer = f"Default test {final_verdict} | " + footer
        fig.text(.02, .035, textwrap.fill(footer, 47 * columns, replace_whitespace=False), fontsize=8)
        fig.subplots_adjust(left=.23 if columns == 1 else .085, right=.98,
                            top=.80, bottom=.34, hspace=.85, wspace=.32)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(pixels).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)
    durations = [260] * len(frames)
    durations[-1] = 1800
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    with Image.open(path) as decoded:
        if decoded.n_frames != len(records):
            raise ValueError("GIF discarded an actual observation")
    for frame in frames:
        frame.close()
    return {"default_verdict_displayed": final_verdict,
            "goal_annotations": annotations, "numeric_observations_changed": False}


def run_case(case, output, *, device="cpu", recipe_name=None, steps=None,
             eval_samples=None, frames=9, seed=24002):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    steps = case["default_steps"] if steps is None else steps
    samples = case["eval_samples"] if eval_samples is None else eval_samples
    if type(samples) is not int or samples < 1:
        raise ValueError("evaluation sample count must be positive")
    boundaries = contract.evaluation_steps(steps, frames)
    metric_count = contract.metric_observations(case)
    metric_steps = contract.evaluation_steps(steps, min(steps, metric_count) + 1)
    evaluation_steps = sorted(set(boundaries) | set(metric_steps))
    required_terminal = case.get("terminal_observations", 5)
    receipt = {"schema": "particlegan_api_toy_run_v1", "case": case,
               "recipe": None, "api_components": [],
               "source": source_identity(), "seed": seed,
               "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "device": str(device), "cuda": torch.version.cuda,
                           "torch_threads": torch.get_num_threads()},
               "protocol": {"updates": steps, "default_updates": case["default_steps"],
                            "evaluation_samples": samples, "default_evaluation_samples": case["eval_samples"],
                            "evaluation_steps": evaluation_steps,
                            "metric_evaluation_steps": metric_steps, "media_steps": boundaries,
                            "metric_observations": metric_count, "media_frames": frames,
                            "terminal_observations": required_terminal},
               "historical_results_changed": False, "observations": []}
    full = (steps >= case["default_steps"] and samples >= case["eval_samples"]
            and len(metric_steps) - 1 >= metric_count)
    records, arrays = [], {}
    started = time.monotonic()
    completed = 0
    fixture = None
    try:
        fixture = contract.build(case, device=device, seed=seed, recipe_name=recipe_name, max_steps=steps)
        receipt["recipe"] = json_value(fixture.recipe.to_dict())
        receipt["api_components"] = list(fixture.api_components)
        receipt["source"] = source_identity()
        for step in range(steps + 1):
            if step:
                fixture.step()
                completed = step
            if step not in evaluation_steps:
                continue
            with isolated_evaluation():
                record = contract.validate_observation(fixture.observe(n=samples, seed=seed + 10000))
            record["step"] = step
            for index, view in enumerate(record["views"]):
                for role in ("target", "samples"):
                    view[role] = contract.array(view[role]).copy()
                    arrays[f"step{step}_view{index}_{role}"] = view[role]
            records.append(record)
            compact = {key: value for key, value in record.items() if key != "views"}
            compact["views"] = [{key: value for key, value in view.items()
                                 if key not in {"target", "samples"}} for view in record["views"]]
            receipt["observations"].append(compact)
            print(json.dumps({"case": case["id"], "step": step, "metric_passed": record["passed"],
                              "failed_bounds": record["failed_bounds"][:5]}, allow_nan=False), flush=True)
        terminal = [record for record in records if record["step"] > 0
                    and record["step"] in metric_steps][-required_terminal:]
        sustained = len(terminal) == required_terminal and all(record["passed"] for record in terminal)
        failed = list(records[-1]["failed_bounds"])
        if not full:
            failed.append("default budget, evaluation draw count or metric cadence not completed")
        if not sustained:
            failed.append(f"last {required_terminal} post-update metric observations do not all pass")
        receipt.update(status="COMPLETE", metric_passed=records[-1]["passed"],
                       sustained_metric_passed=sustained, default_protocol_complete=full,
                       passed=full and sustained, verdict="PASS" if full and sustained else "FAIL",
                       failed_bounds=failed, completed_updates=completed)
    except Exception as error:
        receipt.update(status="ERROR", passed=False, verdict="FAIL",
                       failed_bounds=[f"API execution or metric error: {type(error).__name__}: {error}"],
                       completed_updates=completed, default_protocol_complete=False)
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["artifacts"] = {}
    receipt["gif_frames"] = 0
    if records:
        try:
            np.savez_compressed(output / "observations.npz", **arrays)
            gif = output / "goal.gif"
            media_records = [record for record in records if record["step"] in boundaries]
            render_gif(case, media_records, gif, full_budget=full, requested_steps=steps,
                       final_verdict=receipt["verdict"])
            torch.save(fixture.state_dict(), output / "final-state.pt")
            receipt["artifacts"] = {name: {"sha256": file_hash(output / name),
                                           "bytes": (output / name).stat().st_size}
                                     for name in ("goal.gif", "observations.npz", "final-state.pt")}
            receipt["gif_frames"] = len(media_records)
        except Exception as error:
            receipt.update(passed=False, verdict="FAIL", status="ERROR")
            receipt["failed_bounds"].append(f"goal media/state error: {type(error).__name__}: {error}")
    receipt["source_unchanged"] = all((contract.ROOT / path).is_file() and
                                        file_hash(contract.ROOT / path) == expected
                                        for path, expected in receipt["source"]["files_sha256"].items())
    if not receipt["source_unchanged"]:
        receipt.update(passed=False, verdict="FAIL", status="ERROR")
        receipt["failed_bounds"].append("source changed during execution")
    write_json(output / "receipt.json", receipt)
    return receipt


def inventory(cases):
    historical = json.loads((contract.ROOT / "reports/toy_audit/catalog.json").read_text())["cases"]
    expected = [case["id"] for case in historical] + ["pr231"]
    return {"schema": "particlegan_api_toy_inventory_v1", "coverage": contract.coverage(cases, expected),
            "cases": list(cases.values()),
            "contract": "Public ParticleGAN execution, fixed binary metrics, actual goal-reference/output GIFs"}


def _execute_request(request):
    case, output, options = request
    torch.set_num_threads(1)
    result = run_case(case, output, **options)
    return {"id": case["id"], "verdict": result["verdict"], "status": result["status"],
            "metric_passed": result.get("metric_passed", False),
            "default_protocol_complete": result["default_protocol_complete"],
            "completed_updates": result["completed_updates"],
            "receipt": str(Path(output) / "receipt.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List executable variants and goals")
    parser.add_argument("--inventory", type=Path, help="Write the complete compact coverage ledger")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recipe", default="auto")
    parser.add_argument("--steps", type=int, help="Explicitly bounded run; short runs never qualify the default budget")
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--jobs", type=int, default=1, help="Independent CPU case workers; each uses one Torch thread")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cases = contract.discover()
    if args.list:
        for case in cases.values():
            print(f"{case['id']}\t{case['goal']}")
    if args.inventory:
        write_json(args.inventory, inventory(cases))
    selected = list(cases) if args.all else args.case
    if not selected:
        if args.list or args.inventory:
            return 0
        parser.error("choose --list, --inventory, --case or --all")
    if not args.output:
        parser.error("--output is required for actual execution")
    if args.jobs < 1 or (args.jobs > 1 and torch.device(args.device).type != "cpu"):
        parser.error("--jobs must be positive; parallel case workers require CPU")
    unknown = set(selected) - set(cases)
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    requests = [(cases[name], args.output / name,
                 dict(device=args.device, recipe_name=None if args.recipe == "auto" else args.recipe,
                      steps=None if args.steps is None else min(args.steps, cases[name]["default_steps"]),
                      eval_samples=args.eval_samples, frames=args.frames)) for name in selected]
    results = []
    if args.jobs == 1:
        iterator = map(_execute_request, requests)
        for result in iterator:
            results.append(result)
            write_json(args.output / "summary.json", {"cases": results})
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_execute_request, request) for request in requests]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                write_json(args.output / "summary.json", {"cases": results})
                print(json.dumps({"finished_case": result["id"], "verdict": result["verdict"],
                                  "status": result["status"], "finished_cases": len(results),
                                  "total_cases": len(requests)}), flush=True)
    write_json(args.output / "summary.json", {"cases": results})
    return 0 if all(result["verdict"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
